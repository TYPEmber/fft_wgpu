use num_complex::Complex;
use std::f64::consts::PI;
use std::result;
use wgpu::hal::auxil::db;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Forward<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pub buffer_a: &'a wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    twiddle_buffer: wgpu::Buffer,
    //pub round_num: wgpu::Buffer,
    // pub fft_len_buf: wgpu::Buffer,
    pub fft_len: u32,
    pub data_len: u32,
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
        let data_len_u32 = data_len as u32;
        let buffer_a = src;

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data_len,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let n = fft_len as usize;
        let mut twiddles = Vec::with_capacity(n / 2);

        for k in 0..n / 2 {
            let theta = -2.0 * PI * (k as f64) / (n as f64);
            twiddles.push(Complex::new(theta.cos() as f32, theta.sin() as f32));
        }

        let twiddle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Twiddle Buffer"),
            contents: bytemuck::cast_slice(&twiddles),
            usage: wgpu::BufferUsages::STORAGE,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: twiddle_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline: pipeline_forward,
            bind_group: bind_group_forward,

            buffer_a,
            buffer_b,
            twiddle_buffer,
            fft_len,
            data_len: data_len_u32, //round_num,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.twiddle_buffer.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group_forward, &[]);

        let x = (self.fft_len / 2 / 512).max(1); //每个x对应一组fft运算
        //let x =self.data_len/self.fft_len;
        let y = (self.buffer_a.size() / 8 / self.fft_len as u64) as u32; //一个data中有2个u32，一个u32有4个byte
        //let y=1;
        let z = 1;

        // dbg!(self);

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        //let i: u32 = 0;
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
            wgpu::BindGroupLayoutEntry {
                binding: 2, // 新增Twiddle绑定
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let bind_group_inverse = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            bind_group: bind_group_inverse,
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

pub struct Normalize<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    buffer_a: &'a wgpu::Buffer,
    buffer_b: &'a wgpu::Buffer,
    // pub round_num: wgpu::Buffer,
    //pub fft_len_buf: wgpu::Buffer,
    pub fft_len: u32,
}

impl<'a> Normalize<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        buffer1: &'a wgpu::Buffer,
        buffer2: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let pipeline_normalize = prepare_cs_model_normalize(device);

        // let data_len = buffer1.size();

        let num_rounds = fft_len.trailing_zeros();

        let (buffer_a, buffer_b) = if num_rounds % 2 == 0 {
            (buffer1, buffer2)
        } else {
            (buffer2, buffer1)
        };
        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_normalize = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_normalize.get_bind_group_layout(0),
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
            pipeline: pipeline_normalize,
            bind_group: bind_group_normalize,
            fft_len,
            buffer_a,
            buffer_b,
        }
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let bind_group_normalize = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        cpass.set_bind_group(0, &bind_group_normalize, &[]);

        let x = (self.fft_len / 32).max(1);
        let y = (self.buffer_a.size() / 8 / self.fft_len as u64) as u32;
        let z = 1;

        // dbg!(self);

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        cpass.dispatch_workgroups(x, y, z);

        self.buffer_b
    }
}

fn prepare_cs_model_normalize(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/normalize.wgsl"
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
            range: 0..4,
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

pub struct Onlyinverse<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    buffer_a: &'a wgpu::Buffer,
    buffer_b: &'a wgpu::Buffer,
    // pub round_num: wgpu::Buffer,
    //pub fft_len_buf: wgpu::Buffer,
    pub fft_len: u32,
}

impl<'a> Onlyinverse<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        src: &'a wgpu::Buffer,
        src2: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let pipeline_onlyinverse = prepare_cs_model_onlyinverse(device);

        // let data_len = src.size();

        let buffer_a = src;

        let buffer_b = src2;

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_onlyinverse = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_onlyinverse.get_bind_group_layout(0),
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
            pipeline: pipeline_onlyinverse,
            bind_group: bind_group_onlyinverse,
            fft_len,
            buffer_a,
            buffer_b,
        }
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let bind_group_onlyinverse = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        cpass.set_bind_group(0, &bind_group_onlyinverse, &[]);

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
            self.buffer_b
        }
    }
}

fn prepare_cs_model_onlyinverse(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/onlyifft.wgsl"
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

pub struct Multiply<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    //bind_group: wgpu::BindGroup,
    buffer_a: &'a wgpu::Buffer,
    buffer_b: &'a wgpu::Buffer,
    result: wgpu::Buffer,
    // pub round_num: wgpu::Buffer,
    //pub fft_len_buf: wgpu::Buffer,
}

impl<'a> Multiply<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        src: &'a wgpu::Buffer,
        src2: &'a wgpu::Buffer,
    ) -> Self {
        let pipeline_multiply = prepare_cs_model_multiply(device);

        let data_len = src.size();

        let buffer_a = src;
        let buffer_b=src2;
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data_len,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_multiply = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_multiply.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline: pipeline_multiply,
            //bind_group: bind_group_multiply,
           
            buffer_a,
            buffer_b,
            result,
        }
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let bind_group_multiply = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.result.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group_multiply, &[]);
        let workgroup_len=64 ;
        let x = 1024/workgroup_len;
        let y = (self.buffer_a.size() / 8 / 1024) as u32; //一个字节是8个bit
        let z = 1;

        // dbg!(self);
        //for i in 0..(self.fft_len as f32).log2().round() as u32 {
           // cpass.set_push_constants(4, &i.to_le_bytes());
            cpass.dispatch_workgroups(x, y, z);
       // }

        &self.result
    }
}



fn prepare_cs_model_multiply(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/multiply.wgsl"
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        push_constant_ranges: &[],
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
