use crate::typed_buffer;
use num_complex::Complex;
use wgpu::{
    BindGroup, BufferUsages, CommandEncoderDescriptor, ComputePipeline, Device, Queue,
    naga::FastHashMap,
    util::{self, DeviceExt},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Recipe {
    Radix2,
    Radix3,
    Radix4,
    Radix5,
    Radix6,
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Forward,
    Inverse,
}

pub struct Config {
    pub fft_len: u32,
    pub recipe: Recipe,
    pub direction: Direction,
}

pub struct Processor<'a> {
    config: Config,
    twiddles: typed_buffer::Array<Complex<f32>>,
    device: &'a Device,
    queue: &'a Queue,
    pipeline: ComputePipeline,
    bind_group_cache: FastHashMap<[usize; 2], BindGroup>,
}

impl<'a> Processor<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue, config: Config) -> Result<Self, FftError> {
        let pipeline = create_pipeline(device, &config.recipe, &config.direction)?;
        let twiddles = typed_buffer::Array::new(
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Twiddle Buffer"),
                contents: bytemuck::cast_slice(
                    &(0..config.fft_len)
                        .map(|k| -2.0 * std::f64::consts::PI * (k as f64) / (config.fft_len as f64))
                        .map(|theta| Complex::new(theta.cos() as f32, theta.sin() as f32))
                        .collect::<Vec<_>>(),
                ),
                usage: BufferUsages::STORAGE,
            }),
        );

        Ok(Self {
            config,
            twiddles,
            device,
            queue,
            pipeline,
            bind_group_cache: Default::default(),
        })
    }

    pub fn proc(
        &mut self,
        input: &typed_buffer::Array<Complex<f32>>,
        output: &typed_buffer::Array<Complex<f32>>,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("FFT Command Encoder"),
            });

        if self.bind_group_cache.len() > 1024 {
            self.bind_group_cache.clear();
        }
        let bind_group: &BindGroup = self
            .bind_group_cache
            .entry([input as *const _ as usize, output as *const _ as usize])
            // Do not use `or_insert`
            // It will new default value everytime.
            .or_insert_with(|| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FFT Bind Group"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.twiddles.inner.as_entire_binding(),
                        },
                    ],
                })
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Compute Pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            // 计算工作组数量
            let workgroup_size = 64;
            let base = match self.config.recipe {
                Recipe::Radix2 => 2,
                Recipe::Radix3 => 3,
                Recipe::Radix4 => 4,
                Recipe::Radix5 => 5,
                Recipe::Radix6 => 6,
            };

            let workgroup_count_per_fft = (self.config.fft_len / base / workgroup_size).max(1);

            // 计算数据批次数量
            let batch_count = (input.size() / self.config.fft_len as u64) as u32;
            let num_stages = self.config.fft_len.ilog(base);

            cpass.set_push_constants(0, &self.config.fft_len.to_le_bytes());
            // 设置FFT长度和当前阶段
            for stage in 0..num_stages {
                cpass.set_push_constants(4, &stage.to_le_bytes());
                cpass.dispatch_workgroups(workgroup_count_per_fft, batch_count, 1);
            }
        }

        self.queue.submit([encoder.finish()]);
    }
}

#[derive(Debug)]
pub enum FftError {
    InvalidRadix,
    InvalidLength,
    InvalidUsage,
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidRadix => write!(f, "Invalid FFT radix specified"),
            Self::InvalidLength => write!(f, "FFT length incompatible with selected radix"),
            Self::InvalidUsage => write!(f, "Buffer missing required usage flags"),
        }
    }
}

impl std::error::Error for FftError {}

fn create_pipeline(
    device: &Device,
    recipe: &Recipe,
    direction: &Direction,
) -> Result<ComputePipeline, FftError> {
    // 根据基底和方向选择WGSL文件
    let shader_source = match (recipe, direction) {
        (Recipe::Radix2, Direction::Forward) => include_str!("kernel/fftct.wgsl"),
        (Recipe::Radix2, Direction::Inverse) => include_str!("kernel/ifftct.wgsl"),
        (Recipe::Radix3, Direction::Forward) => include_str!("kernel/fft3ct.wgsl"),
        (Recipe::Radix3, Direction::Inverse) => include_str!("kernel/ifft3ct.wgsl"),
        (Recipe::Radix4, Direction::Forward) => include_str!("kernel/fft4ct.wgsl"),
        (Recipe::Radix4, Direction::Inverse) => include_str!("kernel/ifft4ct.wgsl"),
        (Recipe::Radix5, Direction::Forward) => include_str!("kernel/fft5ct.wgsl"),
        (Recipe::Radix5, Direction::Inverse) => include_str!("kernel/ifft5ct.wgsl"),
        (Recipe::Radix6, Direction::Forward) => include_str!("kernel/fft6ct.wgsl"),
        (Recipe::Radix6, Direction::Inverse) => include_str!("kernel/ifft6ct.wgsl"),
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

fn create_multiply_pipeline(device: &Device) -> Result<ComputePipeline, FftError> {
    // 选择 WGSL 文件
    let shader_source = include_str!("kernel/multiply.wgsl");

    // 创建着色器模块
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Multiply Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });

    // 创建绑定组布局
    let bgl: wgpu::BindGroupLayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Multiply Bind Group Layout"),
        entries: &[
            // 输入缓冲区 A
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,  // 可在计算阶段被访问
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 输入缓冲区 B
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
            // 输出缓冲区
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

    // 创建管道布局
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Multiply Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // 创建计算管线
    Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Multiply Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    }))
}

pub struct MultiplyProcessor<'a> {
    device: &'a Device,
    queue: &'a Queue,
    pipeline: ComputePipeline,
    bind_group_cache: FastHashMap<[usize; 3], BindGroup>,
}

impl<'a> MultiplyProcessor<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue) -> Result<Self, FftError> {
        let pipeline = create_multiply_pipeline(device)?;

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_cache: Default::default(),
        })
    }

    pub fn proc(
        &mut self,
        input_a: &typed_buffer::Array<Complex<f32>>,
        input_b: &typed_buffer::Array<Complex<f32>>,
        output: &typed_buffer::Array<Complex<f32>>,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Multiply Encoder"),
            });

        if self.bind_group_cache.len() > 1024 {
            self.bind_group_cache.clear();
        }

        let bind_group: &BindGroup = self
            .bind_group_cache
            .entry([
                input_a as *const _ as usize,
                input_b as *const _ as usize,
                output as *const _ as usize,
            ])
            .or_insert_with(|| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Multiply Bind Group"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_a.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_b.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output.inner.as_entire_binding(),
                        },
                    ],
                })
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Multiply Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            let workgroup_size = 64;
            let num_workgroups = (input_a.size() as u32 + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.queue.submit([encoder.finish()]);
    }
}
