use wgpu::hal::auxil::db;

#[derive(Debug)]
pub struct Forward<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    buffer_a: &'a wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    //pub round_num: wgpu::Buffer,
    // pub fft_len_buf: wgpu::Buffer,
    pub fft_len: u32,
}

impl<'a> Forward<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        src: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let pipeline_forward = prepare_cs_model(device);

        let data_len = src.size();

        let buffer_a = src;

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data_len,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // let round_num = device.create_buffer(&wgpu::BufferDescriptor {
        // label: None,
        // size: (std::mem::size_of::<u32>()) as u64,
        // usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        // mapped_at_creation: false,
        // });

        // let fft_len_buf = device.create_buffer(&wgpu::BufferDescriptor {
        // label: None,
        // size: (std::mem::size_of::<u32>()) as u64,
        // usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        // mapped_at_creation: false,
        // });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_forward = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_forward.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline: pipeline_forward,
            bind_group: bind_group_forward,
            fft_len,
            buffer_a,
            buffer_b,
            //round_num,
            // fft_len_buf,
        }
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let bind_group_forward = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffer_b.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group_forward, &[]);

        let x = (self.fft_len / 2 / 32).max(1);
        let y = (self.buffer_a.size() / 8 / self.fft_len as u64) as u32;
        let z = 1;

        // dbg!(self);

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());

        for i in 0..(self.fft_len as f32).log2().round() as u32 {
            cpass.set_push_constants(4, &i.to_le_bytes());
            cpass.dispatch_workgroups(x, y, z);
        }

        if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
            self.buffer_a
        } else {
            &self.buffer_b
        }
    }
}

fn prepare_cs_model(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/fft.wgsl"
        ))),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });

    // Instantiates the pipeline.
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    })
}

pub struct Inverse<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    buffer_a: &'a wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    // pub round_num: wgpu::Buffer,
    //pub fft_len_buf: wgpu::Buffer,
    pub fft_len: u32,
}

impl<'a> Inverse<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        src: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let pipeline_inverse = prepare_cs_model_inverse(device);

        let data_len = src.size();

        let buffer_a = src;

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data_len,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_forward = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_inverse.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline: pipeline_inverse,
            bind_group: bind_group_forward,
            fft_len,
            buffer_a,
            buffer_b,
        }
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let bind_group_inverse = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffer_b.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group_inverse, &[]);

        let x = (self.fft_len / 2 / 32).max(1);
        let y = (self.buffer_a.size() / 8 / self.fft_len as u64) as u32;
        let z = 1;

        // dbg!(self);

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        let round_num = self.fft_len.trailing_zeros();
        cpass.set_push_constants(8, &round_num.to_le_bytes());

        for i in 0..(self.fft_len as f32).log2().round() as u32 {
            cpass.set_push_constants(4, &i.to_le_bytes());
            cpass.dispatch_workgroups(x, y, z);
        }

       
       

        if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
            self.buffer_a
        } else {
            &self.buffer_b
        }
    }
}

fn prepare_cs_model_inverse(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/ifft.wgsl"
        ))),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..12,
        }],
    });

    // Instantiates the pipeline.
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    })
}
