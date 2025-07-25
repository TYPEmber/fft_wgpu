use num_complex::Complex;
use rustfft::Fft;
use std::f64::consts::PI;
use std::result;
use wgpu::hal::auxil::db;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FftRadix {
    Radix2,
    Radix3,
    Radix4,
    Radix5,
    Radix6,
}

#[derive(Debug)]
pub enum FftDirection {
    Forward,
    Inverse,
}

#[derive(Debug)]
pub struct FftProcessor<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pub buffer_a: &'a wgpu::Buffer,
    pub buffer_b: wgpu::Buffer,
    twiddle_buffer: wgpu::Buffer,
    radix: FftRadix,
    fft_len: u32,
    data_len: u32,
    direction: FftDirection,
}

#[derive(Debug)]
pub enum FftError {
    InvalidRadix,
    InvalidLength,
    UnsupportedCombination,
}

impl FftRadix {
    /// 获取基底的数值表示
    pub fn value(&self) -> u32 {
        match self {
            FftRadix::Radix2 => 2,
            FftRadix::Radix3 => 3,
            FftRadix::Radix4 => 4,
            FftRadix::Radix5 => 5,
            FftRadix::Radix6 => 6,
        }
    }

    /// 计算FFT所需的阶段数
    pub fn num_stages(&self, len: u32) -> u32 {
        let base = self.value() as f32;
        (len as f32).log(base).round() as u32
    }
}

impl<'a> FftProcessor<'a> {
    pub fn new(
        radix: FftRadix,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        src: &'a wgpu::Buffer,
        fft_len: u32,
        direction: FftDirection,
    ) -> Result<Self, FftError> {
        // 验证长度是否匹配基底
        // if !radix.is_valid_length(fft_len) {
        //     return Err(FftError::InvalidLength);
        // }

        // 创建计算管线
        let pipeline = create_pipeline(device, radix, &direction)?;

        let data_len = src.size();
        let data_len_u32 = data_len as u32;
        let buffer_a = src;

        // 创建输出缓冲区
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Output Buffer"),
            size: data_len,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 生成旋转因子
        let n = fft_len as usize;
        let mut twiddles = Vec::with_capacity(n);

        for k in 0..n {
            let theta = -2.0 * PI * (k as f64) / (n as f64);
            twiddles.push(Complex::new(theta.cos() as f32, theta.sin() as f32));
        }

        // 创建旋转因子缓冲区
        let twiddle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Twiddle Buffer"),
            contents: bytemuck::cast_slice(&twiddles),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 创建绑定组
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFT Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
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

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            buffer_a,
            buffer_b,
            twiddle_buffer,
            radix,
            fft_len,
            data_len: data_len_u32,
            direction,
        })
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Compute Pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);

        // 计算工作组数量
        let workgroup_size = 64;
        let workgroup_count_per_fft = match self.radix {
            FftRadix::Radix2 => (self.fft_len / 2 / workgroup_size).max(1),
            FftRadix::Radix3 => (self.fft_len / 3 / workgroup_size).max(1),
            FftRadix::Radix4 => (self.fft_len / 4 / workgroup_size).max(1),
            FftRadix::Radix5 => (self.fft_len / 5 / workgroup_size).max(1),
            FftRadix::Radix6 => (self.fft_len / 6 / workgroup_size).max(1),
        };

        // 计算数据批次数量
        let batch_count = (self.data_len / (self.fft_len * 8)) as u32; // 每个复数8字节

        // 计算阶段数
        let num_stages = self.radix.num_stages(self.fft_len);
        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        // 设置FFT长度和当前阶段
        for stage in 0..num_stages {
            cpass.set_push_constants(4, &stage.to_le_bytes());
            cpass.dispatch_workgroups(workgroup_count_per_fft, batch_count, 1);
        }
        &self.buffer_b
    }

    pub fn get_output_buffer(&self) -> &wgpu::Buffer {
        &self.buffer_b
    }
}

/// 统一创建计算管线
fn create_pipeline(
    device: &wgpu::Device,
    radix: FftRadix,
    direction: &FftDirection,
) -> Result<wgpu::ComputePipeline, FftError> {
    // 根据基底和方向选择WGSL文件
    let shader_source = match (radix, direction) {
        (FftRadix::Radix2, FftDirection::Forward) => include_str!("kernel/fftct.wgsl"),
        (FftRadix::Radix2, FftDirection::Inverse) => include_str!("kernel/ifftct.wgsl"),
        (FftRadix::Radix3, FftDirection::Forward) => include_str!("kernel/fft3ct.wgsl"),
        (FftRadix::Radix3, FftDirection::Inverse) => include_str!("kernel/ifft3ct.wgsl"),
        (FftRadix::Radix4, FftDirection::Forward) => include_str!("kernel/fft4ct.wgsl"),
        (FftRadix::Radix4, FftDirection::Inverse) => include_str!("kernel/ifft4ct.wgsl"),
        (FftRadix::Radix5, FftDirection::Forward) => include_str!("kernel/fft5ct.wgsl"),
        (FftRadix::Radix5, FftDirection::Inverse) => include_str!("kernel/ifft5ct.wgsl"),
        (FftRadix::Radix6, FftDirection::Forward) => include_str!("kernel/fft6ct.wgsl"),
        (FftRadix::Radix6, FftDirection::Inverse) => include_str!("kernel/ifft6ct.wgsl"),
        _ => return Err(FftError::UnsupportedCombination),
    };

    // 创建着色器模块
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });

    // 创建绑定组布局
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFT Bind Group Layout"),
        entries: &[
            // 输入缓冲区
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
            // 输出缓冲区
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
            // 旋转因子缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    // 创建管道布局
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8, // 用于传递fft_len和stage
        }],
    });

    // 创建计算管线
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Compute Pipeline"),
            layout: Some(&ppl),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        }),
    )
}

// #[derive(Debug)]
// pub struct FftProcessor<'a> {
//     device: &'a wgpu::Device,
//     queue: &'a wgpu::Queue,
//     pipeline: wgpu::ComputePipeline,
//     bind_group: wgpu::BindGroup,
//     pub buffer_a: &'a wgpu::Buffer,
//     pub buffer_b: wgpu::Buffer,
//     twiddle_buffer: wgpu::Buffer,
//     //pub round_num: wgpu::Buffer,
//     // pub fft_len_buf: wgpu::Buffer,
//     pub fft_len: u32,
//     pub data_len: u32,
//     direction: FftDirection,
// }

// impl<'a> FftProcessor<'a> {
//     pub fn new(
//         device: &'a wgpu::Device,
//         queue: &'a wgpu::Queue,
//         src: &'a wgpu::Buffer,
//         fft_len: u32,
//         direction: FftDirection,
//     ) -> Self {
//         let pipeline = match direction {
//             FftDirection::Forward => prepare_cs_model_forward(device),
//             FftDirection::Inverse => prepare_cs_model_backward(device),
//         };
//         let data_len = src.size();
//         let data_len_u32 = data_len as u32;
//         let buffer_a = src;

//         let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
//             label: None,
//             size: data_len,
//             usage: wgpu::BufferUsages::COPY_DST
//                 | wgpu::BufferUsages::COPY_SRC
//                 | wgpu::BufferUsages::STORAGE,
//             mapped_at_creation: false,
//         });

//         let n = fft_len as usize;
//         let mut twiddles = Vec::with_capacity(n / 2);

//         for k in 0..n / 2 {
//             let theta = -2.0 * PI * (k as f64) / (n as f64);
//             twiddles.push(Complex::new(theta.cos() as f32, theta.sin() as f32));
//         }

//         let twiddle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//             label: Some("Twiddle Buffer"),
//             contents: bytemuck::cast_slice(&twiddles),
//             usage: wgpu::BufferUsages::STORAGE,
//         });

//         let bind_group_forward = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             label: None,
//             layout: &pipeline.get_bind_group_layout(0),
//             entries: &[
//                 wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: buffer_a.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 1,
//                     resource: buffer_b.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 2,
//                     resource: twiddle_buffer.as_entire_binding(),
//                 },
//             ],
//         });

//         Self {
//             device,
//             queue,
//             pipeline,
//             bind_group: bind_group_forward,

//             buffer_a,
//             buffer_b,
//             twiddle_buffer,
//             fft_len,
//             data_len: data_len_u32, //round_num,
//             direction,              // fft_len_buf,
//         }
//     }

//     pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
//         // let bind_group_forward = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
//         //     label: None,
//         //     layout: &self.pipeline.get_bind_group_layout(0),
//         //     entries: &[
//         //         wgpu::BindGroupEntry {
//         //             binding: 0,
//         //             resource: self.buffer_a.as_entire_binding(),
//         //         },
//         //         wgpu::BindGroupEntry {
//         //             binding: 1,
//         //             resource: self.buffer_b.as_entire_binding(),
//         //         },
//         //         wgpu::BindGroupEntry {
//         //             binding: 2,
//         //             resource: self.twiddle_buffer.as_entire_binding(),
//         //         },
//         //     ],
//         // });
//         let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//             label: None,
//             timestamp_writes: None,
//         });

//         cpass.set_pipeline(&self.pipeline);
//         cpass.set_bind_group(0, &self.bind_group, &[]);

//         let x = (self.fft_len / 2 / 64).max(1); //每个x对应一组fft运算
//         //let x =self.data_len/self.fft_len;
//         let y = (self.buffer_a.size() / 8 / self.fft_len as u64) as u32; //一个data中有2个u32，一个u32有4个byte
//         //let y=1;
//         let z = 1;

//         // dbg!(self);

//         cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
//         //let i: u32 = 0;
//         for i in 0..(self.fft_len as f32).log2().round() as u32 {
//             cpass.set_push_constants(4, &i.to_le_bytes());
//             cpass.dispatch_workgroups(x, y, z);
//         }
//         if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
//             self.buffer_a
//         } else {
//             &self.buffer_b
//         }
//     }
//     pub fn get_output_buffer(&self) -> &wgpu::Buffer {
//         &self.buffer_b
//     }
// }

#[derive(Debug)]
pub struct Forward<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pub buffer_a: &'a wgpu::Buffer,
    pub buffer_b: wgpu::Buffer,
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
        let pipeline_forward = prepare_cs_model_forward(device);

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
        // let bind_group_forward = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     label: None,
        //     layout: &self.pipeline.get_bind_group_layout(0),
        //     entries: &[
        //         wgpu::BindGroupEntry {
        //             binding: 0,
        //             resource: self.buffer_a.as_entire_binding(),
        //         },
        //         wgpu::BindGroupEntry {
        //             binding: 1,
        //             resource: self.buffer_b.as_entire_binding(),
        //         },
        //         wgpu::BindGroupEntry {
        //             binding: 2,
        //             resource: self.twiddle_buffer.as_entire_binding(),
        //         },
        //     ],
        // });
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);

        let x = (self.fft_len / 2 / 64).max(1); //每个x对应一组fft运算
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

fn prepare_cs_model_forward(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/fftct.wgsl"
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
    pub buffer_a: &'a wgpu::Buffer,
    pub buffer_b: wgpu::Buffer,
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

fn prepare_cs_model_backward(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/ifftct.wgsl"
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
    pub result: wgpu::Buffer,
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
        let buffer_b = src2;
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
        let workgroup_len = 64;
        let total_elements = (self.buffer_a.size() / 8) as u32;
        let x = 1024 / workgroup_len;
        let y = ((total_elements + 1024 - 1) / 1024).max(1); //一个字节是8个bit
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

pub struct IntegratedMultiply<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    //bind_group: wgpu::BindGroup,
    buffer: &'a wgpu::Buffer,
    pub result: wgpu::Buffer,
    pub fft_len: u32,
    // pub round_num: wgpu::Buffer,
    //pub fft_len_buf: wgpu::Buffer,
}

impl<'a> IntegratedMultiply<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        src: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let pipeline_integratedmultiply = prepare_cs_model_integratedmultiply(device);

        let data_len = src.size() - (fft_len as u64 * 8);

        let buffer = src;
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
            layout: &pipeline_integratedmultiply.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline: pipeline_integratedmultiply,
            //bind_group: bind_group_multiply,
            buffer,
            result,
            fft_len,
        }
    }

    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        let bind_group_integratedmultiply =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.result.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group_integratedmultiply, &[]);

        let workgroup_len = 64;
        let data_len = self.buffer.size() / 8;
        let x = 1024 / workgroup_len; //每个x对应一组fft运算
        let y = (data_len / 1024) as u32; //一个data中有2个u32，一个u32有4个byte
        let z = 1;
        compute_pass.set_push_constants(0, &self.fft_len.to_le_bytes());
        compute_pass.dispatch_workgroups(x, y, z);

        &self.result
    }
}

fn prepare_cs_model_integratedmultiply(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/multiply2.wgsl"
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

use num_complex::Complex32;
use std::borrow::Cow;
//use wgpu::util::DeviceExt;

/// 张量转置处理器
///
/// 专门用于GPU上执行张量的循环右移转置操作。
/// 每个处理器实例维护自己的维度信息和转置逻辑，
/// 持有输入缓冲区的引用并内部创建输出缓冲区。
pub struct TransposeProcessor<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // 维度信息
    dims: Vec<u32>,
    output_dims: Vec<u32>, // 转置后的维度

    total_elements: usize,

    // 缓冲区
    input_buffer: &'a wgpu::Buffer,
    pub output_buffer: wgpu::Buffer, // 内部创建的输出缓冲区
    dims_buffer: wgpu::Buffer,
    strides_buffer: wgpu::Buffer,
}

impl<'a> TransposeProcessor<'a> {
    /// 创建新的转置处理器
    ///
    /// # 参数
    /// * `device` - WGPU设备引用
    /// * `queue` - WGPU队列引用
    /// * `input_buffer` - 输入数据缓冲区引用
    /// * `dims` - 输入张量的维度
    ///
    /// # 返回
    /// 新的TransposeProcessor实例
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        input_buffer: &'a wgpu::Buffer,
        dims: &[u32],
    ) -> Self {
        let rank = dims.len();
        assert!(rank > 0, "张量维度不能为空");

        let total_elements = dims.iter().product::<u32>() as usize;
        let buffer_size = input_buffer.size();

        // 内部创建输出缓冲区
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transpose Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 计算转置信息
        let (output_dims, strides_t_r) = Self::calculate_transpose_info(dims);

        // 创建维度缓冲区
        let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transpose Dims Buffer"),
            contents: bytemuck::cast_slice(dims),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 创建步长缓冲区
        let strides_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transpose Strides Buffer"),
            contents: bytemuck::cast_slice(&strides_t_r),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 创建绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Transpose Bind Group Layout"),
            entries: &[
                // 输入数据
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 输出数据
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
                // 形状数据
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 步长数据
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // 创建计算管道
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Transpose Compute Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Transpose Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tensor Transpose Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "kernel/transpose_nd.wgsl"
                ))),
            }),
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        // 创建绑定组
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Transpose Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: strides_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group,
            dims: dims.to_vec(),
            output_dims,
            total_elements,
            input_buffer,
            output_buffer,
            dims_buffer,
            strides_buffer,
        }
    }

    /// 执行转置操作
    ///
    /// # 参数
    /// * `encoder` - 命令编码器
    ///
    /// # 返回
    /// 输出缓冲区的引用
    pub fn proc(&self, encoder: &mut wgpu::CommandEncoder) -> &wgpu::Buffer {
        // 创建计算通道
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Transpose Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        // 计算工作组数量 (每个工作组64个线程)
        let workgroup_count = (self.total_elements as u32 + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

        // 返回输出缓冲区
        &self.output_buffer
    }

    /// 更新输入缓冲区（在流水线处理中可能需要）
    ///
    /// # 参数
    /// * `new_input_buffer` - 新的输入缓冲区引用
    ///
    /// # 返回
    /// 更新了绑定组的自身引用
    pub fn update_input_buffer(&mut self, new_input_buffer: &'a wgpu::Buffer) -> &mut Self {
        // 更新输入缓冲区引用
        self.input_buffer = new_input_buffer;

        // 获取绑定组布局
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);

        // 重新创建绑定组
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Updated Transpose Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.strides_buffer.as_entire_binding(),
                },
            ],
        });

        self
    }

    /// 获取转置后的输出维度
    pub fn get_output_dims(&self) -> &[u32] {
        &self.output_dims
    }

    /// 获取总元素数量
    pub fn get_total_elements(&self) -> usize {
        self.total_elements
    }

    /// 获取输入缓冲区引用
    pub fn get_input_buffer(&self) -> &wgpu::Buffer {
        self.input_buffer
    }

    /// 获取输出缓冲区引用
    pub fn get_output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer
    }

    /// 计算转置信息（输出形状和步长）
    fn calculate_transpose_info(dims: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let rank = dims.len();

        // 计算循环右移的置换
        let mut perm = vec![0; rank];
        perm[0] = rank - 1; // 最后一个维度移到第一位
        for i in 1..rank {
            perm[i] = i - 1; // 其它维度依次后移
        }

        // 计算转置后的形状
        let output_dims: Vec<u32> = perm.iter().map(|&i| dims[i]).collect();

        // 计算转置用步长
        let mut strides_t_r = vec![0u32; rank];
        let mut current_stride = 1;
        for j in (0..rank).rev() {
            let dim = perm[j];
            strides_t_r[dim] = current_stride;
            current_stride *= dims[dim];
        }

        (output_dims, strides_t_r)
    }
}
