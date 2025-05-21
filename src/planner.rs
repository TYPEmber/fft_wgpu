use crate::processor::{Forward, Inverse, Multiply};

use num_complex::Complex;
use std::f64::consts::PI;
use wgpu::util::DeviceExt;

/// 主FFT计划管理器，负责管理GPU资源和缓冲区
pub struct FFTPlanner {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl FFTPlanner {
    /// 异步创建新的FFT计划管理器
    pub async fn new() -> Self {
        // 初始化WebGPU实例
        let instance = wgpu::Instance::default();
        
        // 请求高性能适配器
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("无法找到合适的GPU适配器");

        // 创建设备和队列
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: adapter.features(),
                    required_limits: adapter.limits(),
                    label: Some("FFT Device"),
                    ..Default::default()
                },
                None
            )
            .await
            .expect("无法创建设备");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// 从复数数据创建缓冲区
    pub fn create_buffer_from_data(&self, data: &[Complex<f32>]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Complex Data Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    /// 创建指定大小的空缓冲区
    pub fn create_empty_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empty Complex Buffer"),
            size: (size * std::mem::size_of::<Complex<f32>>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    /// 创建用于读取结果的暂存缓冲区
    pub fn create_staging_buffer(&self, size: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (size * std::mem::size_of::<Complex<f32>>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn create_forward(&self, fft_len: u32) -> ForwardFFT {
        ForwardFFT::new(&self.device, &self.queue, fft_len)
    }

    /// 创建逆向FFT计算器
    pub fn create_inverse(&self, fft_len: u32) -> InverseFFT {
        InverseFFT::new(&self.device, &self.queue, fft_len)
    }

    /// 创建复数乘法计算器
    pub fn create_multiply(&self) -> MultiplyFFT {
        MultiplyFFT::new(&self.device, &self.queue)
    }
    
    /// 创建命令编码器
    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { 
            label: Some("FFT Command Encoder") 
        })
    }
    
    /// 向缓冲区写入数据
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, data: &[Complex<f32>]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }
    
    /// 提交命令到队列
    pub fn submit_commands(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }
    
    /// 异步从暂存缓冲区读取数据
    pub async fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> Vec<Complex<f32>> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        rx.receive().await.unwrap().expect("Failed to map buffer");
        
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        buffer.unmap();
        
        result
    }
    
    /// 获取设备引用
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    
    /// 获取队列引用
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// 正向FFT计算器 - 与具体缓冲区解耦
pub struct ForwardFFT<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    twiddle_buffer: wgpu::Buffer,  // 旋转因子缓冲区保留在结构体中
    fft_len: u32,
    twiddles: Vec<Complex<f32>>,   // 保存CPU端的旋转因子，以便需要时重用
}

impl<'a> ForwardFFT<'a> {
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, fft_len: u32) -> Self {
        let pipeline = prepare_cs_model(device);
        
        // 创建旋转因子 - 这部分与具体输入数据无关，只与FFT长度有关
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

        Self {
            device,
            queue,
            pipeline,
            twiddle_buffer,
            fft_len,
            twiddles,
        }
    }

    /// 执行FFT计算，传入输入缓冲区和一个可以接收输出的缓冲区
    pub fn proc<'b>(&self, 
                encoder: &mut wgpu::CommandEncoder, 
                input_buffer: &'b wgpu::Buffer, 
                output_buffer: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
        
        // 每次执行时动态创建绑定组
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
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
                    resource: self.twiddle_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // 使用基于FFT长度的工作组配置
        let x = (self.fft_len / 2 / 512).max(1); // 每个x对应一组fft运算
        let elements_count = input_buffer.size() / 8; // 每个复数8字节
        let y = (elements_count / self.fft_len as u64) as u32;
        let z = 1;

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        
        for i in 0..(self.fft_len as f32).log2().round() as u32 {
            cpass.set_push_constants(4, &i.to_le_bytes());
            cpass.dispatch_workgroups(x, y, z);
        }

        // 根据迭代次数的奇偶性决定结果在哪个缓冲区
        if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
            input_buffer
        } else {
            output_buffer
        }
    }
    
    /// 获取FFT长度
    pub fn fft_len(&self) -> u32 {
        self.fft_len
    }
    
    /// 获取旋转因子
    pub fn twiddles(&self) -> &[Complex<f32>] {
        &self.twiddles
    }
}

/// 逆向FFT计算器 - 与具体缓冲区解耦
pub struct InverseFFT<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    fft_len: u32,
}

impl<'a> InverseFFT<'a> {
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, fft_len: u32) -> Self {
        let pipeline = prepare_cs_model_inverse(device);

        Self {
            device,
            queue,
            pipeline,
            fft_len,
        }
    }

    /// 执行IFFT计算，传入输入缓冲区和输出缓冲区
    pub fn proc<'b>(&self, 
                encoder: &mut wgpu::CommandEncoder, 
                input_buffer: &'b wgpu::Buffer, 
                output_buffer: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
        
        // 动态创建绑定组
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let x = (self.fft_len / 2 / 32).max(1);
        let elements_count = input_buffer.size() / 8;
        let y = (elements_count / self.fft_len as u64) as u32;
        let z = 1;

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        let round_num = self.fft_len.trailing_zeros();
        cpass.set_push_constants(8, &round_num.to_le_bytes());

        for i in 0..(self.fft_len as f32).log2().round() as u32 {
            cpass.set_push_constants(4, &i.to_le_bytes());
            cpass.dispatch_workgroups(x, y, z);
        }

        // 根据迭代次数的奇偶性决定结果在哪个缓冲区
        if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
            input_buffer
        } else {
            output_buffer
        }
    }
    
    /// 获取FFT长度
    pub fn fft_len(&self) -> u32 {
        self.fft_len
    }
}

/// 复数乘法计算器 - 与具体缓冲区解耦
pub struct MultiplyFFT<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> MultiplyFFT<'a> {
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Self {
        let pipeline = prepare_cs_model_multiply(device);

        Self {
            device,
            queue,
            pipeline,
        }
    }

    /// 执行复数乘法，传入两个输入缓冲区和一个输出缓冲区
    pub fn proc<'b>(&self, 
                encoder: &mut wgpu::CommandEncoder, 
                buffer_a: &wgpu::Buffer, 
                buffer_b: &wgpu::Buffer,
                result_buffer: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
        
        // 动态创建绑定组
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
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
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        let workgroup_len = 64;
        let total_elements = (buffer_a.size() / 8) as u32;
        let x = 1024 / workgroup_len;
        let y = ((total_elements + 1024 - 1) / 1024).max(1);
        let z = 1;

        cpass.dispatch_workgroups(x, y, z);

        result_buffer
    }
}

// 着色器编译函数保持不变
fn prepare_cs_model(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // 加载WGSL着色器
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/fft.wgsl"
        ))),
    });

    // 创建绑定组布局
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // 创建管线布局
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });

    // 创建计算管线
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

// 其他着色器编译函数同样保持不变
fn prepare_cs_model_inverse(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // 实现与之前相同
    // ...
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

fn prepare_cs_model_multiply(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // 实现与之前相同
    // ...
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
