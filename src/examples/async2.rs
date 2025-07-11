// use num_complex::Complex32 as Complex;
// use std::f64::consts::PI;
// use std::sync::{Arc, Mutex};
// use std::time::{Duration, Instant};
// use wgpu::util::DeviceExt;
// use std::thread;

// #[tokio::main]
// async fn main() {
//     println!("===== 双实例并行测试：长计算 vs 简单I/O =====");

//     // -------- 创建第一个实例用于长时间计算 --------
//     let instance1 = wgpu::Instance::new(&wgpu::InstanceDescriptor {
//         backends: wgpu::Backends::VULKAN,
//         flags: wgpu::InstanceFlags::empty(),
//         backend_options: Default::default(),
//     });
    
//     let adapter1 = instance1
//         .request_adapter(&wgpu::RequestAdapterOptions {
//             power_preference: wgpu::PowerPreference::HighPerformance,
//             ..Default::default()
//         })
//         .await
//         .unwrap();
    
//     println!("计算实例适配器信息: {:?}", adapter1.get_info());
    
//     let (device1, queue1) = adapter1
//         .request_device(
//             &wgpu::DeviceDescriptor {
//                 required_features: adapter1.features(),
//                 required_limits: adapter1.limits(),
//                 label: Some("Compute Device"),
//                 ..Default::default()
//             },
//             None,
//         )
//         .await
//         .unwrap();
    
//     // -------- 创建第二个实例用于I/O操作 --------
//     // 注意：这里我们启用异步计算环境变量，尝试获取更好的I/O性能
//     unsafe {
//         std::env::set_var("WGPU_VULKAN_ASYNC_COMPUTE", "1");
//     }
    
//     let instance2 = wgpu::Instance::new(&wgpu::InstanceDescriptor {
//         backends: wgpu::Backends::VULKAN,
//         flags: wgpu::InstanceFlags::empty(),
//         backend_options: Default::default(),
//     });
    
//     let adapter2 = instance2
//         .request_adapter(&wgpu::RequestAdapterOptions {
//             power_preference: wgpu::PowerPreference::HighPerformance,
//             ..Default::default()
//         })
//         .await
//         .unwrap();
    
//     println!("I/O实例适配器信息: {:?}", adapter2.get_info());
    
//     let (device2, queue2) = adapter2
//         .request_device(
//             &wgpu::DeviceDescriptor {
//                 required_features: adapter2.features(),
//                 required_limits: adapter2.limits(),
//                 label: Some("I/O Device"),
//                 ..Default::default()
//             },
//             None,
//         )
//         .await
//         .unwrap();
    
//     // 创建Arc包装以便在线程间共享
//     let device1 = Arc::new(device1);
//     let queue1 = Arc::new(queue1);
//     let device2 = Arc::new(device2);
//     let queue2 = Arc::new(queue2);
    
//     // -------- 创建长时间计算所需的资源 --------
//     let fft_len: u32 = 512;
//     let batch_count = 500 * 5;
//     let fft_data_size = (fft_len as usize) * batch_count;
//     let fft_data = vec![Complex::new(5.0, 0.0); fft_data_size];
    
//     let fft_buffer_a = device1.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("FFT Input Buffer"),
//         contents: bytemuck::cast_slice(&fft_data),
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });
    
//     let fft_buffer_b = device1.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("FFT Output Buffer"),
//         size: (fft_data_size * std::mem::size_of::<Complex>()) as u64,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//         mapped_at_creation: false,
//     });
    
//     // 预计算旋转因子
//     let n = fft_len as usize;
//     let mut twiddles = Vec::with_capacity(n / 2);
//     for k in 0..n / 2 {
//         let theta = -2.0 * PI * (k as f64) / (n as f64);
//         twiddles.push(Complex::new(theta.cos() as f32, theta.sin() as f32));
//     }
    
//     let twiddle_buffer = device1.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("Twiddle Buffer"),
//         contents: bytemuck::cast_slice(&twiddles),
//         usage: wgpu::BufferUsages::STORAGE,
//     });
    
//     // 创建FFT着色器
//     let cs_module = device1.create_shader_module(wgpu::ShaderModuleDescriptor {
//         label: Some("FFT Shader"),
//         source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
//             "../kernel/async.wgsl"
//         ))),
//     });
    
//     // 创建FFT管线和绑定组
//     let bgl = device1.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//         label: Some("FFT Bind Group Layout"),
//         entries: &[
//             wgpu::BindGroupLayoutEntry {
//                 binding: 0,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: false },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             wgpu::BindGroupLayoutEntry {
//                 binding: 1,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: false },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             wgpu::BindGroupLayoutEntry {
//                 binding: 2,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: true },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//         ],
//     });
    
//     let ppl = device1.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//         label: Some("FFT Pipeline Layout"),
//         bind_group_layouts: &[&bgl],
//         push_constant_ranges: &[wgpu::PushConstantRange {
//             stages: wgpu::ShaderStages::COMPUTE,
//             range: 0..8,
//         }],
//     });
    
//     let pipeline = device1.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//         label: Some("FFT Pipeline"),
//         layout: Some(&ppl),
//         module: &cs_module,
//         entry_point: Some("main"),
//         compilation_options: Default::default(),
//         cache: None,
//     });
    
//     let bind_group = device1.create_bind_group(&wgpu::BindGroupDescriptor {
//         label: Some("FFT Bind Group"),
//         layout: &pipeline.get_bind_group_layout(0),
//         entries: &[
//             wgpu::BindGroupEntry {
//                 binding: 0,
//                 resource: fft_buffer_a.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 1,
//                 resource: fft_buffer_b.as_entire_binding(),
//             },
//             wgpu::BindGroupEntry {
//                 binding: 2,
//                 resource: twiddle_buffer.as_entire_binding(),
//             },
//         ],
//     });
    
//     // -------- 创建I/O测试所需的资源 --------
//     // 创建多组不同大小的缓冲区以测试各种I/O场景
//     let io_sizes = [
//         1024 * 1024,      // 1MB
//         4 * 1024 * 1024,  // 4MB
//        // 16 * 1024 * 1024, // 16MB
//     ];
    
//     let mut io_buffers = Vec::new();
//     let mut io_staging_buffers = Vec::new();
//     let mut io_data_vecs = Vec::new();
    
//     for &size in &io_sizes {
//         let io_data = vec![42u32; size];
//         io_data_vecs.push(io_data.clone());
        
//         let io_buffer = device2.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//             label: Some(&format!("I/O Buffer ({}MB)", size / 1024 / 1024)),
//             contents: bytemuck::cast_slice(&io_data),
//             usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
//         });
        
//         let io_staging_buffer = device2.create_buffer(&wgpu::BufferDescriptor {
//             label: Some(&format!("I/O Staging Buffer ({}MB)", size / 1024 / 1024)),
//             size: (size * std::mem::size_of::<u32>()) as u64,
//             usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
//             mapped_at_creation: false,
//         });
        
//         io_buffers.push(io_buffer);
//         io_staging_buffers.push(io_staging_buffer);
//     }
    
//     // -------- 同步变量 --------
//     let compute_done = Arc::new(Mutex::new(false));
//     let io_results = Arc::new(Mutex::new(Vec::new()));
    
//     let compute_done_for_io = Arc::clone(&compute_done);
//     let io_results_for_io = Arc::clone(&io_results);
    
//     // -------- 执行测试 --------
//     println!("\n开始并行测试，请等待...");
//     let test_start = Instant::now();
    
//     // 启动计算线程
//     let compute_thread = {
//         let device = Arc::clone(&device1);
//         let queue = Arc::clone(&queue1);
//         let compute_done = Arc::clone(&compute_done);
        
//         thread::spawn(move || {
//             let compute_start = Instant::now();
//             println!("[计算] 线程启动");
            
//             // 创建计算命令
//             let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
//                 label: Some("Long Computation Encoder"),
//             });
            
//             // 为了创建长时间计算，执行多次FFT迭代
//             {
//                 let threads_per_fft = fft_len / 2;
//                 let workgroup_len = 64;
//                 let x = (threads_per_fft / workgroup_len).max(1);
//                 let y = batch_count as u32;
                
//                 let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//                     label: Some("Long Compute Pass"),
//                     timestamp_writes: None,
//                 });
                
//                 cpass.set_pipeline(&pipeline);
//                 cpass.set_bind_group(0, &bind_group, &[]);
                
//                 cpass.set_push_constants(0, &fft_len.to_le_bytes());
                
//                 let total_stages = f32::log2(fft_len as f32) as u32;
//                 let repeat_count = 10001; // 极高的重复次数以确保计算足够长
                
//                 // println!("[计算] 开始计算 {} 次 FFT，每次 {} 个阶段", repeat_count, total_stages);
//                 // for i in 0..repeat_count {
//                 //     if i > 0 && i % 10000 == 0 {
//                 //         println!("[计算] 已完成 {}/{} 次迭代", i, repeat_count);
//                 //     }
//                      let stage=total_stages-1;
//                 //     for stage in 0..total_stages {
//                         cpass.set_push_constants(4, &stage.to_le_bytes());
//                         cpass.dispatch_workgroups(x, y, 1);
//                     }
//                // }
//             //}
            
//             // 提交计算命令
           
//             queue.submit(Some(encoder.finish()));
//             println!("[计算] 提交计算命令...");
//             // 等待计算完成
//             device.poll(wgpu::Maintain::Wait);
            
//             let compute_time = compute_start.elapsed();
//             println!("[计算] 计算完成，耗时: {:?}", compute_time);
            
//             // 标记计算已完成
//             let mut done = compute_done.lock().unwrap();
//             *done = true;
            
//             compute_time
//         })
//     };
    
//     // 等待一小段时间以确保计算已开始
//     thread::sleep(Duration::from_millis(20));
    
//     // 启动I/O线程
//     let io_thread = {
//         let device = Arc::clone(&device2);
//         let queue = Arc::clone(&queue2);
        
//         thread::spawn(move || {
            
//            thread::sleep(Duration::from_millis(2));
           
//             println!("[I/O] 线程启动");
            
//             // 对每个大小的缓冲区执行I/O操作
//             for (i, (buffer, staging_buffer)) in io_buffers.iter().zip(io_staging_buffers.iter()).enumerate() {
//                 if *compute_done_for_io.lock().unwrap() {
//                     println!("[I/O] 计算已完成，停止更多I/O测试");
//                     break;
//                 }
//                 let io_size = io_sizes[i];
//                 let io_start = Instant::now();
                
//                 println!("[I/O] 对 {}MB 大小的缓冲区开始I/O操作...", io_size / 1024 / 1024);
                
//                 // 创建I/O命令
//                 let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
//                     label: Some(&format!("I/O Encoder ({}MB)", io_size / 1024 / 1024)),
//                 });
                
//                 // 执行缓冲区复制操作
//                 encoder.copy_buffer_to_buffer(
//                     &buffer,
//                     0,
//                     &staging_buffer,
//                     0,
//                     (io_size * std::mem::size_of::<u32>()) as u64
//                 );
                
//                 // 提交I/O命令
//                 queue.submit(Some(encoder.finish()));
                
//                 // 设置异步映射回调以检测I/O完成
//                 let completion = Arc::new(Mutex::new(None));
//                 let completion_clone = Arc::clone(&completion);
                
//                 staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
//                     if result.is_ok() {
//                         let now = Instant::now();
//                         let mut time = completion_clone.lock().unwrap();
//                         *time = Some(now);
//                     }
//                 });
                
//                 // 轮询直到I/O完成
//                 while completion.lock().unwrap().is_none() {
//                     device.poll(wgpu::Maintain::Poll);
//                     thread::sleep(Duration::from_millis(1));
//                 }
                
//                 let io_completion = completion.lock().unwrap().unwrap();
//                 let io_time = io_start.elapsed();
                
//                 // 验证I/O数据
//                 let data = staging_buffer.slice(..).get_mapped_range();
//                 let result: &[u32] = bytemuck::cast_slice(&*data);
//                 let is_valid = result.iter().all(|&x| x == 42);
                
//                 // 记录I/O结果
//                 let mut results = io_results_for_io.lock().unwrap();
//                 results.push((io_size / 1024 / 1024, io_time, is_valid));
                
//                 println!("[I/O] {}MB 缓冲区I/O完成，耗时: {:?}，验证 {}", 
//                          io_size / 1024 / 1024, io_time, if is_valid { "成功" } else { "失败" });
                
//                 println!("[I/O] I/O操作完成，本次总耗时: {:?}", test_start.elapsed());
//                 // 释放I/O资源
//                 drop(data);
//                 staging_buffer.unmap();
                
//                 // 检查计算是否仍在进行中
             
//             }
//         })
//     };
    
//     // 等待两个线程完成
//     let compute_time = compute_thread.join().unwrap();
//     io_thread.join().unwrap();
    
//     let total_time = test_start.elapsed();
    
//     // -------- 显示测试结果 --------
//     println!("\n===== 测试结果 =====");
//     println!("总测试时间: {:?}", total_time);
//     println!("计算耗时: {:?}", compute_time);
    
//     let io_results = io_results.lock().unwrap();
//     for &(size_mb, duration, is_valid) in io_results.iter() {
//         println!(
//             "I/O {}MB - 耗时: {:?}, 数据验证 {}, 计算进行状态: {}",
//             size_mb,
//             duration,
//             if is_valid { "成功" } else { "失败" },
//             if duration < compute_time { "仍在计算" } else { "计算已完成" }
//         );
        
//         // 如果I/O比计算快，说明成功实现了并行
//         if duration < compute_time {
//             println!(
//                 "  --> 并行成功！I/O在计算完成前 {:?} 完成",
//                 compute_time - duration
//             );
//         } else {
//             println!(
//                 "  --> I/O被计算阻塞，在计算完成后 {:?} 完成",
//                 duration - compute_time
//             );
//         }
//     }
    
//     println!("\n===== 结论 =====");
//     if io_results.iter().any(|&(_, duration, _)| duration < compute_time) {
//         println!(
//             "测试证明：使用两个WGPU实例可以实现I/O和计算操作的真正并行执行。"
//         );
//         println!(
//             "这表明两个WGPU实例能够有效地访问不同的硬件资源（如不同的队列家族）。"
//         );
//     } else {
//         println!(
//             "即使使用两个WGPU实例，I/O操作仍然被计算所阻塞。"
//         );
//         println!(
//             "这可能是因为驱动程序或硬件限制，或者两个实例共享同一个低级队列。"
//         );
//     }
// }

use num_complex::Complex32 as Complex;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use std::thread;

#[tokio::main]
async fn main() {
    // unsafe {
    //     std::env::set_var("WGPU_VULKAN_ASYNC_COMPUTE", "1");
    // }
    println!("===== 双实例并行测试：长计算 vs 完整I/O流程 =====");

    // -------- 创建第一个实例用于长时间计算 --------
    let instance1 = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: Default::default(),
    });
    
    let adapter1 = instance1
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    
    println!("计算实例适配器信息: {:?}", adapter1.get_info());
    
    let (device1, queue1) = adapter1
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter1.features(),
                required_limits: adapter1.limits(),
                label: Some("Compute Device"),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    
    // -------- 创建第二个实例用于I/O操作 --------
    // 注意：这里我们启用异步计算环境变量，尝试获取更好的I/O性能
   
    
    let instance2 = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: Default::default(),
    });
    
    let adapter2 = instance2
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    
    println!("I/O实例适配器信息: {:?}", adapter2.get_info());
    
    let (device2, queue2) = adapter2
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter2.features(),
                required_limits: adapter2.limits(),
                label: Some("I/O Device"),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    
    // 创建Arc包装以便在线程间共享
    let device1 = Arc::new(device1);
    let queue1 = Arc::new(queue1);
    let device2 = Arc::new(device2);
    let queue2 = Arc::new(queue2);
    
    // -------- 创建长时间计算所需的资源 --------
    let fft_len: u32 = 512;
    let batch_count = 500 * 5;
    let fft_data_size = (fft_len as usize) * batch_count;
    let fft_data = vec![Complex::new(5.0, 0.0); fft_data_size];
    
    let fft_buffer_a = device1.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("FFT Input Buffer"),
        contents: bytemuck::cast_slice(&fft_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    
    let fft_buffer_b = device1.create_buffer(&wgpu::BufferDescriptor {
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
    
    let twiddle_buffer = device1.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddle Buffer"),
        contents: bytemuck::cast_slice(&twiddles),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    // 创建FFT着色器
    let cs_module = device1.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../kernel/async.wgsl"
        ))),
    });
    
    // 创建FFT管线和绑定组
    let bgl = device1.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    
    let ppl = device1.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });
    
    let pipeline = device1.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFT Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let bind_group = device1.create_bind_group(&wgpu::BindGroupDescriptor {
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
    // 创建多组不同大小的缓冲区以测试各种I/O场景，但不预填充数据
    let io_sizes = [
        1024 * 1024,      // 1MB
        4 * 1024 * 1024,  // 4MB
        16 * 1024 * 1024, // 16MB - 添加更大的测试尺寸
    ];
    
    let mut io_buffers = Vec::new();
    let mut io_staging_buffers = Vec::new();
    
    for &size in &io_sizes {
        // 创建源缓冲区，现在是空的
        let io_buffer = device2.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("I/O Buffer ({}MB)", size / 1024 / 1024)),
            size: (size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 创建目标缓冲区
        let io_staging_buffer = device2.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("I/O Staging Buffer ({}MB)", size / 1024 / 1024)),
            size: (size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        io_buffers.push(io_buffer);
        io_staging_buffers.push(io_staging_buffer);
    }
    
    // -------- 同步变量 --------
    let compute_done = Arc::new(Mutex::new(false));
    let io_results = Arc::new(Mutex::new(Vec::new()));
    
    let compute_done_for_io = Arc::clone(&compute_done);
    let io_results_for_io = Arc::clone(&io_results);
    
    // -------- 执行测试 --------
    println!("\n开始并行测试，请等待...");
    let test_start = Instant::now();
    
    // 启动计算线程
    let compute_thread = {
        let device = Arc::clone(&device1);
        let queue = Arc::clone(&queue1);
        let compute_done = Arc::clone(&compute_done);
        
        thread::spawn(move || {
            let compute_start = Instant::now();
            println!("[计算] 线程启动");
            
            // 创建计算命令
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Long Computation Encoder"),
            });
            
            // 为了创建长时间计算，执行多次FFT迭代
            {
                let threads_per_fft = fft_len / 2;
                let workgroup_len = 64;
                let x = (threads_per_fft / workgroup_len).max(1);
                let y = batch_count as u32;
                
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Long Compute Pass"),
                    timestamp_writes: None,
                });
                
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                
                cpass.set_push_constants(0, &fft_len.to_le_bytes());
                
                let total_stages = f32::log2(fft_len as f32) as u32;
                let stage = total_stages - 1;
                
                // 增加计算强度，确保计算时间足够长
                let dispatch_count = 200;  // 较大的数字以确保长时间计算
                
                println!("[计算] 提交 {} 次dispatch以模拟长时间计算", dispatch_count);
                //for i in 0..dispatch_count {
                    cpass.set_push_constants(4, &stage.to_le_bytes());
                    cpass.dispatch_workgroups(x, y, 1);
                    
                //     if i > 0 && i % 50 == 0 {
                //         println!("[计算] 已提交 {}/{} 次dispatch", i, dispatch_count);
                //     }
                // }
            }
            
            // 提交计算命令
            println!("[计算] 提交计算命令...");
            queue.submit(Some(encoder.finish()));
            
            // 等待计算完成
            println!("[计算] 等待计算完成...");
            device.poll(wgpu::Maintain::Wait);
            
            let compute_time = compute_start.elapsed();
            println!("[计算] 计算完成，耗时: {:?}", compute_time);
            
            // 标记计算已完成
            let mut done = compute_done.lock().unwrap();
            *done = true;
            
            compute_time
        })
    };
    
    // 等待一小段时间以确保计算已开始
    thread::sleep(Duration::from_millis(1000));
    
    // 启动I/O线程
    let io_thread = {
        let device = Arc::clone(&device2);
        let queue = Arc::clone(&queue2);
        
        thread::spawn(move || {
            println!("[I/O] 线程启动");
            
            // 对每个大小的缓冲区执行I/O操作
            for (i, (buffer, staging_buffer)) in io_buffers.iter().zip(io_staging_buffers.iter()).enumerate() {
                // 检查计算是否已完成
                if *compute_done_for_io.lock().unwrap() {
                    println!("[I/O] 计算已完成，停止更多I/O测试");
                    break;
                }
                
                let io_size = io_sizes[i];
                let io_start = Instant::now();
                
                println!("[I/O] 对 {}MB 大小的缓冲区开始完整I/O操作...", io_size / 1024 / 1024);
                
                // 创建测试数据 - 现在在运行时创建
                let io_data = vec![42u32; io_size];
                
                // 1. 记录写入时间开始
                let write_start = Instant::now();
                
                // 写入数据到缓冲区 - 这是新增的步骤，之前是提前填充的
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&io_data));
                
                // 创建I/O命令
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("I/O Encoder ({}MB)", io_size / 1024 / 1024)),
                });
                
                // 2. 执行缓冲区复制操作
                encoder.copy_buffer_to_buffer(
                    &buffer,
                    0,
                    &staging_buffer,
                    0,
                    (io_size * std::mem::size_of::<u32>()) as u64
                );
                
                // 提交I/O命令
                queue.submit(Some(encoder.finish()));
                
                // 3. 设置异步映射回调以检测I/O完成
                let completion = Arc::new(Mutex::new(None));
                let completion_clone = Arc::clone(&completion);
                
                staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        let now = Instant::now();
                        let mut time = completion_clone.lock().unwrap();
                        *time = Some(now);
                    }
                });
                
                // 轮询直到I/O完成
                let mut poll_start = Instant::now();
                while completion.lock().unwrap().is_none() {
                    device.poll(wgpu::Maintain::Poll);
                    
                    // 输出轮询进度
                    if poll_start.elapsed().as_millis() > 500 {
                        println!("[I/O] 持续轮询 {}MB 数据操作，已经过 {:?}...", 
                                io_size / 1024 / 1024, io_start.elapsed());
                        poll_start = Instant::now();
                    }
                    
                    thread::sleep(Duration::from_millis(1));
                }
                if *compute_done_for_io.lock().unwrap() {
                    println!("[I/O] 计算已完成，停止更多I/O测试");
                    break;
                }
                let io_completion = completion.lock().unwrap().unwrap();
                let io_time = io_start.elapsed();
                
                // 验证I/O数据
                let data = staging_buffer.slice(..).get_mapped_range();
                let result: &[u32] = bytemuck::cast_slice(&*data);
                
                // 部分验证数据，避免验证过大的缓冲区
                let verify_count = 10000.min(result.len());
                let is_valid = result[..verify_count].iter().all(|&x| x == 42);
                
                // 记录I/O结果
                let mut results = io_results_for_io.lock().unwrap();
                results.push((io_size / 1024 / 1024, io_time, is_valid));
                
                println!("[I/O] {}MB 缓冲区I/O完成，总耗时: {:?}，验证 {}", 
                         io_size / 1024 / 1024, io_time, if is_valid { "成功" } else { "失败" });
                
                // 释放I/O资源
                drop(data);
                staging_buffer.unmap();
            }
        })
    };
    
    // 等待两个线程完成
    let compute_time = compute_thread.join().unwrap();
    io_thread.join().unwrap();
    
    let total_time = test_start.elapsed();
    
    // -------- 显示测试结果 --------
    println!("\n===== 测试结果 =====");
    println!("总测试时间: {:?}", total_time);
    println!("计算耗时: {:?}", compute_time);
    
    let io_results = io_results.lock().unwrap();
    for &(size_mb, duration, is_valid) in io_results.iter() {
        println!(
            "I/O {}MB - 耗时: {:?}, 数据验证 {}, 计算进行状态: {}",
            size_mb,
            duration,
            if is_valid { "成功" } else { "失败" },
            if duration < compute_time { "仍在计算" } else { "计算已完成" }
        );
        
        // 如果I/O比计算快，说明成功实现了并行
        if duration < compute_time {
            println!(
                "  --> 并行成功！完整I/O操作在计算完成前 {:?} 完成",
                compute_time - duration
            );
        } else {
            println!(
                "  --> I/O被计算阻塞，在计算完成后 {:?} 完成",
                duration - compute_time
            );
        }
    }
    
    println!("\n===== 结论 =====");
    if io_results.iter().any(|&(_, duration, _)| duration < compute_time) {
        println!(
            "测试证明：使用两个WGPU实例可以实现完整I/O流程（包括写入）和计算操作的真正并行执行。"
        );
        println!(
            "这表明两个WGPU实例能够有效地访问不同的硬件资源（如不同的队列家族），实现全流程并行。"
        );
    } else {
        println!(
            "即使使用两个WGPU实例，完整I/O操作（包括写入）仍然被计算所阻塞。"
        );
        println!(
            "这可能是因为驱动程序或硬件限制，或者两个实例共享同一个低级队列。"
        );
    }
}