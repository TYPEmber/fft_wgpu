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

    /// 创建正向FFT计算器，接收输入缓冲区引用
    pub fn create_forward<'a>(&'a self, input_buffer: &'a wgpu::Buffer, fft_len: u32) -> Forward<'a> {
        // 只为FFT内部交替计算创建一个额外缓冲区，输入缓冲区由外部提供
        let buffer_size = input_buffer.size();
        Forward::new(&self.device, &self.queue, input_buffer, fft_len)
    }

    /// 创建逆向FFT计算器，接收输入缓冲区引用
    pub fn create_inverse<'a>(&'a self, input_buffer: &'a wgpu::Buffer, fft_len: u32) -> Inverse<'a> {
        // 只为IFFT内部交替计算创建一个额外缓冲区，输入缓冲区由外部提供
        let buffer_size = input_buffer.size();
        Inverse::new(&self.device, &self.queue, input_buffer, fft_len)
    }

    /// 创建复数乘法计算器，接收输入缓冲区引用
    pub fn create_multiply<'a>(
        &'a self, 
        buffer_a: &'a wgpu::Buffer, 
        buffer_b: &'a wgpu::Buffer
    ) -> Multiply<'a> {
        // 创建结果缓冲区
        let buffer_size = buffer_a.size();
        Multiply::new(&self.device, &self.queue, buffer_a, buffer_b)
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
