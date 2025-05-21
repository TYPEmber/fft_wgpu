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

     pub fn create_forward(&self, fft_len: u32, initial_capacity: usize) -> ForwardFFT {
        ForwardFFT::new(&self.device, &self.queue, fft_len, initial_capacity)
    }

    /// 创建具有初始缓冲区容量的逆向FFT计算器
    pub fn create_inverse(&self, fft_len: u32, initial_capacity: usize) -> InverseFFT {
        InverseFFT::new(&self.device, &self.queue, fft_len, initial_capacity)
    }

    /// 创建复数乘法计算器
    pub fn create_multiply(&self, initial_capacity: usize) -> MultiplyFFT {
        MultiplyFFT::new(&self.device, &self.queue, initial_capacity)
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

/// 正向FFT计算器 - 内部管理临时缓冲区
pub struct ForwardFFT<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    twiddle_buffer: wgpu::Buffer,       // 旋转因子缓冲区
    temp_buffer: wgpu::Buffer,          // 内部临时缓冲区
    fft_len: u32,
    twiddles: Vec<Complex<f32>>,        // 保存CPU端的旋转因子
    current_capacity: usize,            // 当前临时缓冲区容量
    shrink_threshold: f32,              // 收缩阈值因子（如：4.0表示当容量>所需的4倍时收缩）
    min_capacity: usize,                // 最小容量，避免频繁调整很小的缓冲区
}

impl<'a> ForwardFFT<'a> {
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, fft_len: u32, initial_capacity: usize) -> Self {
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
        
        // 创建初始临时缓冲区
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Temp Buffer"),
            size: (initial_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            twiddle_buffer,
            temp_buffer,
            fft_len,
            twiddles,
            current_capacity: initial_capacity,
            shrink_threshold: 4.0,      // 默认阈值：当容量超过所需的4倍时收缩
            min_capacity: 1024,         // 默认最小容量：1024个元素
        }
    }

    /// 设置收缩阈值
    pub fn set_shrink_threshold(&mut self, threshold: f32) {
        if threshold >= 1.0 {
            self.shrink_threshold = threshold;
        }
    }

    /// 设置最小容量
    pub fn set_min_capacity(&mut self, capacity: usize) {
        self.min_capacity = capacity;
    }

    /// 确保临时缓冲区容量足够，带收缩阈值
    fn ensure_buffer_capacity(&mut self, required_capacity: usize) {
        // 首先确保不低于最小容量
        let required_capacity = required_capacity.max(self.min_capacity);
        
        // 需要扩容
        if self.current_capacity < required_capacity {
            // 创建新的更大缓冲区，适当增加额外容量避免频繁调整
            let new_capacity = (required_capacity as f32 * 1.2) as usize; // 增加20%的余量
            
            self.temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FFT Temp Buffer (Resized)"),
                size: (new_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.current_capacity = new_capacity;
            
            // 调试信息
            println!("FFT缓冲区扩容: {} -> {}", required_capacity, new_capacity);
        }
        // 需要收缩
        else if self.current_capacity > (required_capacity as f32 * self.shrink_threshold) as usize {
            // 避免收缩到过小的容量
            if required_capacity >= self.min_capacity {
                // 收缩时添加少量余量
                let new_capacity = (required_capacity as f32 * 1.1) as usize; // 增加10%的余量
                
                self.temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("FFT Temp Buffer (Shrunk)"),
                    size: (new_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.current_capacity = new_capacity;
                
                // 调试信息
                println!("FFT缓冲区收缩: {} -> {}", self.current_capacity, new_capacity);
            }
        }
        // 当前容量适中，无需调整
    }


    pub fn proc_inplace(&mut self, 
                        encoder: & mut wgpu::CommandEncoder, 
                        input_buffer: & wgpu::Buffer) {
        
        // 计算输入缓冲区的元素数量并确保临时缓冲区足够大
        let element_count = (input_buffer.size() / 8) as usize;
        self.ensure_buffer_capacity(element_count);
        
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
                    resource: self.temp_buffer.as_entire_binding(),
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

        // 计算工作组配置
        let x = (self.fft_len / 2 / 512).max(1); 
        let elements_count = input_buffer.size() / 8; // 每个复数8字节
        let y = (elements_count / self.fft_len as u64) as u32;
        let z = 1;

        cpass.set_push_constants(0, &self.fft_len.to_le_bytes());
        
        for i in 0..(self.fft_len as f32).log2().round() as u32 {
            cpass.set_push_constants(4, &i.to_le_bytes());
            cpass.dispatch_workgroups(x, y, z);
        }

        // if ((self.fft_len as f32).log2().round() as usize) % 2 != 0 {
        //     encoder.copy_buffer_to_buffer(&self.temp_buffer, 0, input_buffer, 0, input_buffer.size());
        // }
    }

       
      
    
    /// 执行FFT计算，只需提供输入缓冲区，使用内部临时缓冲区
    pub fn proc<'b>(&'b mut self, 
                   encoder: &mut wgpu::CommandEncoder, 
                   input_buffer: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
        
        // 计算输入缓冲区的元素数量并确保临时缓冲区足够大
        let element_count = (input_buffer.size() / 8) as usize;
        self.ensure_buffer_capacity(element_count);
        
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
                    resource: self.temp_buffer.as_entire_binding(),
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

        // 计算工作组配置
        let x = (self.fft_len / 2 / 512).max(1); 
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
            &self.temp_buffer
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
    
    /// 获取内部临时缓冲区引用
    pub fn temp_buffer(&self) -> &wgpu::Buffer {
        &self.temp_buffer
    }
    
    /// 获取当前缓冲区容量
    pub fn current_capacity(&self) -> usize {
        self.current_capacity
    }
}

/// 逆向FFT计算器 - 内部管理临时缓冲区
pub struct InverseFFT<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    temp_buffer: wgpu::Buffer,          // 内部临时缓冲区
    fft_len: u32,
    current_capacity: usize,            // 当前临时缓冲区容量
    shrink_threshold: f32,              // 收缩阈值因子
    min_capacity: usize,                // 最小容量
}

impl<'a> InverseFFT<'a> {
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, fft_len: u32, initial_capacity: usize) -> Self {
        let pipeline = prepare_cs_model_inverse(device);
        
        // 创建初始临时缓冲区
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IFFT Temp Buffer"),
            size: (initial_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            temp_buffer,
            fft_len,
            current_capacity: initial_capacity,
            shrink_threshold: 4.0,      // 默认阈值
            min_capacity: 1024,         // 默认最小容量
        }
    }
    
    /// 设置收缩阈值
    pub fn set_shrink_threshold(&mut self, threshold: f32) {
        if threshold >= 1.0 {
            self.shrink_threshold = threshold;
        }
    }

    /// 设置最小容量
    pub fn set_min_capacity(&mut self, capacity: usize) {
        self.min_capacity = capacity;
    }
    
    /// 确保临时缓冲区容量足够，带收缩阈值
    fn ensure_buffer_capacity(&mut self, required_capacity: usize) {
        // 首先确保不低于最小容量
        let required_capacity = required_capacity.max(self.min_capacity);
        
        // 需要扩容
        if self.current_capacity < required_capacity {
            // 创建新的更大缓冲区，适当增加额外容量避免频繁调整
            let new_capacity = (required_capacity as f32 * 1.2) as usize; // 增加20%的余量
            
            self.temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("IFFT Temp Buffer (Resized)"),
                size: (new_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.current_capacity = new_capacity;
            
            // 调试信息
            println!("IFFT缓冲区扩容: {} -> {}", required_capacity, new_capacity);
        }
        // 需要收缩
        else if self.current_capacity > (required_capacity as f32 * self.shrink_threshold) as usize {
            // 避免收缩到过小的容量
            if required_capacity >= self.min_capacity {
                // 收缩时添加少量余量
                let new_capacity = (required_capacity as f32 * 1.1) as usize; // 增加10%的余量
                
                self.temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("IFFT Temp Buffer (Shrunk)"),
                    size: (new_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.current_capacity = new_capacity;
                
                // 调试信息
                println!("IFFT缓冲区收缩: {} -> {}", self.current_capacity, new_capacity);
            }
        }
        // 当前容量适中，无需调整
    }

    /// 执行IFFT计算，只需提供输入缓冲区，使用内部临时缓冲区
    pub fn proc<'b>(&'b mut self, 
                   encoder: &mut wgpu::CommandEncoder, 
                   input_buffer: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
        
        // 计算输入缓冲区的元素数量并确保临时缓冲区足够大
        let element_count = (input_buffer.size() / std::mem::size_of::<Complex<f32>>() as u64) as usize;
        self.ensure_buffer_capacity(element_count);
        
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
                    resource: self.temp_buffer.as_entire_binding(),
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
            &self.temp_buffer
        }
    }
    
    /// 获取FFT长度
    pub fn fft_len(&self) -> u32 {
        self.fft_len
    }
    
    /// 获取内部临时缓冲区引用
    pub fn temp_buffer(&self) -> &wgpu::Buffer {
        &self.temp_buffer
    }
    
    /// 获取当前缓冲区容量
    pub fn current_capacity(&self) -> usize {
        self.current_capacity
    }
}

/// 复数乘法计算器 - 内部管理结果缓冲区
pub struct MultiplyFFT<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    result_buffer: wgpu::Buffer,        // 内部结果缓冲区
    current_capacity: usize,            // 当前缓冲区容量
    shrink_threshold: f32,              // 收缩阈值因子
    min_capacity: usize,                // 最小容量
}

impl<'a> MultiplyFFT<'a> {
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, initial_capacity: usize) -> Self {
        let pipeline = prepare_cs_model_multiply(device);
        
        // 创建初始结果缓冲区
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Multiply Result Buffer"),
            size: (initial_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            result_buffer,
            current_capacity: initial_capacity,
            shrink_threshold: 4.0,      // 默认阈值
            min_capacity: 1024,         // 默认最小容量
        }
    }
    
    /// 设置收缩阈值
    pub fn set_shrink_threshold(&mut self, threshold: f32) {
        if threshold >= 1.0 {
            self.shrink_threshold = threshold;
        }
    }

    /// 设置最小容量
    pub fn set_min_capacity(&mut self, capacity: usize) {
        self.min_capacity = capacity;
    }
    
    /// 确保结果缓冲区容量足够，带收缩阈值
    fn ensure_buffer_capacity(&mut self, required_capacity: usize) {
        // 首先确保不低于最小容量
        let required_capacity = required_capacity.max(self.min_capacity);
        
        // 需要扩容
        if self.current_capacity < required_capacity {
            // 创建新的更大缓冲区，适当增加额外容量避免频繁调整
            let new_capacity = (required_capacity as f32 * 1.2) as usize; // 增加20%的余量
            
            self.result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Multiply Result Buffer (Resized)"),
                size: (new_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.current_capacity = new_capacity;
            
            // 调试信息
            println!("乘法缓冲区扩容: {} -> {}", required_capacity, new_capacity);
        }
        // 需要收缩
        else if self.current_capacity > (required_capacity as f32 * self.shrink_threshold) as usize {
            // 避免收缩到过小的容量
            if required_capacity >= self.min_capacity {
                // 收缩时添加少量余量
                let new_capacity = (required_capacity as f32 * 1.1) as usize; // 增加10%的余量
                
                self.result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Multiply Result Buffer (Shrunk)"),
                    size: (new_capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.current_capacity = new_capacity;
                
                // 调试信息
                println!("乘法缓冲区收缩: {} -> {}", self.current_capacity, new_capacity);
            }
        }
        // 当前容量适中，无需调整
    }
    
    /// 执行复数乘法，只需提供输入缓冲区，使用内部结果缓冲区
    pub fn proc<'b>(&'b mut self, 
                   encoder: &mut wgpu::CommandEncoder, 
                   buffer_a: &'b wgpu::Buffer, 
                   buffer_b: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
        
        // 计算输入缓冲区的元素数量并确保结果缓冲区足够大
        let element_count = (buffer_a.size() / std::mem::size_of::<Complex<f32>>() as u64) as usize;
        self.ensure_buffer_capacity(element_count);
        
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
                    resource: self.result_buffer.as_entire_binding(),
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

        &self.result_buffer
    }
    
    /// 获取内部结果缓冲区引用
    pub fn result_buffer(&self) -> &wgpu::Buffer {
        &self.result_buffer
    }
    
    /// 获取当前缓冲区容量
    pub fn current_capacity(&self) -> usize {
        self.current_capacity
    }
}

// 保留原始的着色器编译函数
fn prepare_cs_model(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // 加载WGSL着色器
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "kernel/fft2.wgsl"
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

// 其他着色器编译函数实现同样保留
fn prepare_cs_model_inverse(device: &wgpu::Device) -> wgpu::ComputePipeline {
    // 实现与之前相同...
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
    // 实现与之前相同...
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

