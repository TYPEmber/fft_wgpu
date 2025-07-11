use futures::executor::block_on;
use num_complex::Complex32 as Complex;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    // 启用Vulkan异步计算支持（如果可用）
    //std::env::set_var("WGPU_VULKAN_ASYNC_COMPUTE", "1");
    
    // 初始化WebGPU实例
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: Default::default(),
    });
    
    // 请求适配器
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    
    //println!("适配器信息: {:?}", adapter.get_info());
  //  println!("支持的特性: {:?}", adapter.features());
    
    // 请求设备
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                label: Some("GPU Device"),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    
    // -------- 创建长时间计算所需的资源 --------
    
    // 创建一个大型缓冲区用于FFT计算
    let fft_len: u32 = 512;
    let batch_count = 500 * 5;
    let fft_data_size = (fft_len as usize) * batch_count;
    
    let fft_data = vec![Complex::new(5.0, 0.0); fft_data_size];
    
    let fft_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FFT Input Buffer"),
        contents: bytemuck::cast_slice(&fft_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    
    let fft_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FFT Output Buffer"),
        size: (fft_data_size * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // 预计算旋转因子
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
    
    // 创建FFT着色器
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../kernel/fftct.wgsl"
        ))),
    });
    
    // 创建FFT管线和绑定组
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFT Bind Group Layout"),
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
    
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFT Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FFT Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: fft_buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fft_buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: twiddle_buffer.as_entire_binding(),
            },
        ],
    });
    
    // -------- 创建I/O测试所需的资源 --------
    
    // 创建一个小型缓冲区用于I/O测试
    let io_data_size = 1024 * 1024; // 约1MB的数据
    let io_data = vec![42u32; io_data_size];
    
    let io_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("I/O Test Buffer"),
        contents: bytemuck::cast_slice(&io_data),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    
    let io_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("I/O Staging Buffer"),
        size: (io_data_size * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // -------- 创建时间戳查询资源（如果支持） --------
    
    let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("设备是否支持时间戳查询: {}", supports_timestamps);
    
    let (timestamp_query_set, timestamp_buffer) = if supports_timestamps {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: 4, // 计算开始、I/O开始、I/O结束、计算结束
        });
        
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Buffer"),
            size: 8 * 4, // 8字节 * 4个时间戳
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        
        (Some(query_set), Some(buffer))
    } else {
        (None, None)
    };
    
    // -------- 执行测试 --------
    
    println!("\n===== 并发I/O测试开始 =====");
    println!("测试方法: 提交大量FFT计算，然后立即提交I/O操作");
    println!("预期结果: 如果GPU支持并发操作，I/O操作应在计算完成前完成");
    
    // 创建一个变量来跟踪I/O操作的完成时间
    let io_completion_time = Arc::new(Mutex::new(None));
    let io_completion_time_clone = Arc::clone(&io_completion_time);
    
    // 记录CPU时间
    let cpu_start_time = Instant::now();
    
    // 创建两个独立的命令编码器
    
    // 1. 长时间计算的命令编码器
    let mut compute_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Long Computation Encoder"),
    });
    
    // 如果支持，记录计算开始时间戳
    if let Some(query_set) = &timestamp_query_set {
        compute_encoder.write_timestamp(query_set, 0);
    }
    
    // 为了增加计算时间，我们重复执行多个FFT阶段
    // 每个阶段都进行多次
    {
        let threads_per_fft = fft_len / 2;
        let workgroup_len = 64;
        let x = (threads_per_fft / workgroup_len).max(1);
        let y = batch_count as u32;
        
        let mut cpass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Long Compute Pass"),
            timestamp_writes: None,
        });
        
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        // 设置FFT长度到推送常量
        cpass.set_push_constants(0, &fft_len.to_le_bytes());
        
        // 执行多次FFT迭代以增加计算时间
        let total_stages = f32::log2(fft_len as f32) as u32;
        let repeat_count = 10000; // 增加此值以延长计算时间
        
        for _ in 0..repeat_count {
            for stage in 0..total_stages {
                cpass.set_push_constants(4, &stage.to_le_bytes());
                cpass.dispatch_workgroups(x, y, 1);
            }
        }
    }
    
    // 如果支持，记录计算结束时间戳（将在最后写入）
    if let Some(query_set) = &timestamp_query_set {
        compute_encoder.write_timestamp(query_set, 3);
    }
    
    // 2. I/O操作的命令编码器
    let mut io_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("I/O Operation Encoder"),
    });
    
    // 如果支持，记录I/O开始时间戳
    if let Some(query_set) = &timestamp_query_set {
        io_encoder.write_timestamp(query_set, 1);
    }
    
    // 执行简单的缓冲区复制操作
    io_encoder.copy_buffer_to_buffer(
        &io_buffer, 
        0, 
        &io_staging_buffer, 
        0, 
        (io_data_size * std::mem::size_of::<u32>()) as u64
    );
    
    // 如果支持，记录I/O结束时间戳
    if let Some(query_set) = &timestamp_query_set {
        io_encoder.write_timestamp(query_set, 2);
    }
    
    // 完成两个命令编码器并获取命令缓冲区
    let compute_commands = compute_encoder.finish();
    let io_commands = io_encoder.finish();
    
    // 先提交计算命令（包含长时间运行的FFT计算）
    println!("提交长时间运行的FFT计算命令...");
    queue.submit(Some(compute_commands));
    let compute_submit_time = Instant::now();
    println!("FFT计算命令提交耗时: {:?}", compute_submit_time.duration_since(cpu_start_time));
    
    // 立即提交I/O命令
    println!("提交I/O操作命令...");
    queue.submit(Some(io_commands));
    let io_submit_time = Instant::now();
    println!("I/O命令提交耗时: {:?}", io_submit_time.duration_since(compute_submit_time));
    
    // 设置异步映射回调来检测I/O操作何时完成
    io_staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        if result.is_ok() {
            // 记录I/O操作完成的时间
            let now = Instant::now();
            let mut completion_time = io_completion_time_clone.lock().unwrap();
            *completion_time = Some(now);
        }
    });
    
    // 轮询设备以推进操作，但不阻塞等待
    println!("开始非阻塞轮询设备...");
    let polling_start = Instant::now();
    
    // 持续轮询，直到I/O操作完成
    while io_completion_time.lock().unwrap().is_none() {
        device.poll(wgpu::Maintain::Poll);
        
        // 给系统一些时间来处理操作
        std::thread::sleep(Duration::from_millis(1));
        
        // 提供一些反馈以表明程序正在运行
        let elapsed = polling_start.elapsed();
        if elapsed.as_millis() % 500 == 0 {
            print!("\r正在等待I/O操作完成... 已经过 {:?}", elapsed);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    
    // I/O操作已完成
    let io_completion = *io_completion_time.lock().unwrap().as_ref().unwrap();
    println!("\nI/O操作已完成！时间: {:?} (相对于测试开始)", io_completion.duration_since(cpu_start_time));
    println!("I/O操作完成延迟: {:?} (相对于I/O命令提交)", io_completion.duration_since(io_submit_time));
    
    // 验证I/O数据
    {
        let data = io_staging_buffer.slice(..).get_mapped_range();
        let result: &[u32] = bytemuck::cast_slice(&*data);
        let is_valid = result.iter().all(|&x| x == 42);
        println!("I/O数据验证: {}", if is_valid { "成功" } else { "失败" });
        drop(data);
        io_staging_buffer.unmap();
    }
    
    // 等待所有操作完成
    println!("等待所有GPU操作完成...");
    device.poll(wgpu::Maintain::Wait);
    let all_complete_time = Instant::now();
    println!("所有操作完成时间: {:?} (相对于测试开始)", all_complete_time.duration_since(cpu_start_time));
    
    // 读取时间戳查询结果（如果支持）
    if let (Some(buffer), Some(query_set)) = (&timestamp_buffer, &timestamp_query_set) {
        let mut resolve_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Timestamp Resolve Encoder"),
        });
        
        resolve_encoder.resolve_query_set(query_set, 0..4, buffer, 0);
        queue.submit(Some(resolve_encoder.finish()));
        
        // 读取时间戳
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
       // let status = buffer.map_async(wgpu::MapMode::Read).await;
        
        let data = buffer.slice(..).get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&*data);
        
        if timestamps.len() >= 4 {
            // 转换时间戳为纳秒
            let compute_start = timestamps[0];
            let io_start = timestamps[1];
            let io_end = timestamps[2];
            let compute_end = timestamps[3];
            
            // 计算时间差
            let compute_duration = compute_end - compute_start;
            let io_duration = io_end - io_start;
            let io_start_offset = if io_start >= compute_start {
                io_start - compute_start
            } else {
                0
            };
            
            println!("\n===== GPU时间戳数据 =====");
            println!("计算开始时间戳: {}", compute_start);
            println!("I/O开始时间戳: {} (偏移 +{}ns)", io_start, io_start_offset);
            println!("I/O结束时间戳: {} (持续 {}ns)", io_end, io_duration);
            println!("计算结束时间戳: {} (持续 {}ns)", compute_end, compute_duration);
            
            // 关键分析：I/O是否在计算完成前结束
            if io_end < compute_end {
                println!("\n结论: I/O操作在计算完成前完成");
                println!("I/O比计算提前完成: {}ns", compute_end - io_end);
                println!("这表明GPU能够同时处理计算和I/O操作");
            } else {
                println!("\n结论: I/O操作在计算完成后完成");
                println!("I/O在计算完成后完成: {}ns", io_end - compute_end);
                println!("这表明GPU上的I/O操作被长时间计算阻塞");
            }
        }
        
        drop(data);
        buffer.unmap();
    } else {
        println!("\n无法提供GPU时间戳分析 - 设备不支持时间戳查询");
        
        // 基于CPU时间的简单分析
        if io_completion < all_complete_time {
            println!("基于CPU时间的分析: I/O操作在所有计算完成前完成");
            println!("这表明GPU可能支持计算和I/O的并发执行");
        } else {
            println!("基于CPU时间的分析: I/O操作直到所有计算完成时才完成");
            println!("这表明GPU可能不支持计算和I/O的并发执行");
        }
    }
    
    println!("\n===== 测试完成 =====");
}