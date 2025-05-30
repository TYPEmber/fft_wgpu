use num_complex::Complex32 as Complex;
use std::f64::consts::PI;
use wgpu::util::DeviceExt;
#[tokio::main]
async fn main() {
    // 初始化 WebGPU 实例
    //let instance = wgpu::Instance::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    // 请求适配器
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    // 请求设备
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    // 创建测试数据
    let data = vec![Complex::new(1.0, 0.0); 512 * 500 * 5];
    let len = data.len();
    let mut ans = vec![Complex::ZERO; len];

    // 创建暂存缓冲区用于读取结果
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 创建数据缓冲区
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer A"),
        contents: bytemuck::cast_slice(data.as_slice()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, // | wgpu::BufferUsages::COPY_SRC,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer B"),
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            //| wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 加载着色器
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Cooley-Tukey Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../kernel/fftct.wgsl"
        ))),
    });

    // 创建绑定组布局
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

    // 创建管线布局
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });

    // 创建计算管线
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFT Cooley-Tukey Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // FFT 参数
    let fft_len: u32 = 512;
    // 总阶段数：1个位反转+第一阶段蝶形运算，然后是剩余的 log2(fft_len)-1 个蝶形运算阶段
    let total_stages = f32::log2(fft_len as f32) as u32;

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

    let upload_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Upload Staging"),
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let total_size = (len * std::mem::size_of::<Complex>()) as u64;
    let first_part_size = total_size * 3 / 4; // 前3/4部分
    let second_part_size = total_size - first_part_size; // 剩余1/4部分
    let upload_first_part_size = total_size / 4;
    let upload_second_part_size = total_size - upload_first_part_size;
    let upload_first_part_elements =
        (upload_first_part_size / std::mem::size_of::<Complex>() as u64) as usize;
    let upload_second_part_elements = len - upload_first_part_elements;

    // 创建两个分段的staging buffers
    let staging_buffer_first = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer First 3/4"),
        size: first_part_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer_second = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer Last 1/4"),
        size: second_part_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let upload_staging_buffer_first = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("First Part Staging Buffer"),
        size: upload_first_part_size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let upload_staging_buffer_second = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Second Part Staging Buffer"),
        size: upload_second_part_size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let timer = std::time::Instant::now();
    let buffer_slice = staging_buffer.slice(..);
    let first_part_slice = staging_buffer_first.slice(..);
    let second_part_slice = staging_buffer_second.slice(..);
    for _ in 0..1000 {
        // 将输入数据写入 buffer_a
        // queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(data.as_slice()));
        upload_staging_buffer_first
            .slice(..)
            .map_async(wgpu::MapMode::Write, |result| result.unwrap());
        device.poll(wgpu::Maintain::Wait);

        let mut upload_view_first = upload_staging_buffer_first.slice(..).get_mapped_range_mut();
        bytemuck::cast_slice_mut(&mut upload_view_first)
            .copy_from_slice(&data[0..upload_first_part_elements]);
        drop(upload_view_first);
        upload_staging_buffer_first.unmap();
        // 第2步: 开始将第一部分从staging buffer复制到目标buffer
        // 第3步: 同时映射第二部分(3/4)并写入数据
        // 使用oneshot channel以便后续确认操作完成
        let (upload_tx, upload_rx) = futures::channel::oneshot::channel();
        upload_staging_buffer_second
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |result| {
                let _ = upload_tx.send(result);
            });

        // 不等待，继续执行以允许与第一部分复制重叠
        // 注意：此处不调用device.poll(wgpu::Maintain::Wait)以允许重叠

        // 第4步: 等待第二部分映射完成
        // 实际应用中，可能需要执行device.poll(wgpu::Maintain::Poll)以确保进度
        // device.poll(wgpu::Maintain::Poll); // 非阻塞轮询
        let mut upload_encoder_first =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("First Part Copy Encoder"),
            });
        upload_encoder_first.copy_buffer_to_buffer(
            &upload_staging_buffer_first,
            0,
            &buffer_a, // 假设这是目标buffer
            0,
            upload_first_part_size,
        );
        queue.submit(Some(upload_encoder_first.finish()));
        let mut upload_encoder_second =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Second Part Copy Encoder"),
            });
        // 在异步环境中等待映射完成
        if let Ok(Ok(_)) = upload_rx.await {
            let mut upload_view_second = upload_staging_buffer_second
                .slice(..)
                .get_mapped_range_mut();
            bytemuck::cast_slice_mut(&mut upload_view_second)
                .copy_from_slice(&data[upload_first_part_elements..]);
            drop(upload_view_second);
            upload_staging_buffer_second.unmap();
            // 第5步: 将第二部分从staging buffer复制到目标buffer
            upload_encoder_second.copy_buffer_to_buffer(
                &upload_staging_buffer_second,
                0,
                &buffer_a,
                upload_first_part_size,
                upload_second_part_size,
            );
            queue.submit(Some(upload_encoder_second.finish()));
        }

      
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Command Encoder"),
        });
        // // 计算工作组维度
        let batches = len as u32 / fft_len;

        // 运行所有 FFT 阶段 - 都需要 fft_len/2 个线程
        {
            let threads_per_fft = fft_len / 2; // 每个 FFT 需要 fft_len/2 个线程
            //let total_threads = batches * threads_per_fft;
            let workgroup_len = 64;
            // let workgroups_x = (total_threads + workgroup_len - 1) / workgroup_len; // 向上取整
            let x = (threads_per_fft / workgroup_len).max(1);
            let y = (buffer_a.size() / 8 / fft_len as u64) as u32; //一个data中有2个u32，一个u32有4个byte
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // 设置推送常量
            cpass.set_push_constants(0, &fft_len.to_le_bytes());
            // push_data.extend_from_slice(&stage.to_le_bytes());
            for i in 0..total_stages {
                cpass.set_push_constants(4, &i.to_le_bytes());

                cpass.dispatch_workgroups(x, y, 1);
            }
        }

        encoder.copy_buffer_to_buffer(&buffer_b, 0, &staging_buffer_first, 0, first_part_size);
        queue.submit(Some(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        //let (sender, receiver) = futures::channel::oneshot::channel();
        first_part_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        let mut encoder_second = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Second Part Encoder"),
        });
        // 提交第二个复制命令
        encoder_second.copy_buffer_to_buffer(
            &buffer_b,
            first_part_size,
            &staging_buffer_second,
            0,
            second_part_size,
        );
        queue.submit(Some(encoder_second.finish()));
        // 等待第一部分映射完成
        if let Ok(result) = receiver.recv() {
            // if let Ok(Ok(_)) = receiver.await {
            // 处理第一部分数据

            let first_data_mapped = first_part_slice.get_mapped_range();
            let first_elements = first_part_size as usize / std::mem::size_of::<Complex>();

            ans[..first_elements]
                .copy_from_slice(&bytemuck::cast_slice(&first_data_mapped)[..first_elements]);

            // 释放第一部分映射
            drop(first_data_mapped);
            staging_buffer_first.unmap();

            // 现在映射第二部分

            second_part_slice.map_async(wgpu::MapMode::Read, |result| {
                result.unwrap();
            });
            device.poll(wgpu::Maintain::Wait);

            // // 处理第二部分数据
            let second_data_mapped = second_part_slice.get_mapped_range();
            let second_elements = second_part_size as usize / std::mem::size_of::<Complex>();
            ans[first_elements..]
                .copy_from_slice(&bytemuck::cast_slice(&second_data_mapped)[..second_elements]);

            // // 释放第二部分映射
            drop(second_data_mapped);
            staging_buffer_second.unmap();
        }
    }
    // for _ in 0..1000 {
    //     // 将输入数据写入 buffer_a
    //     queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(data.as_slice()));
        

    //     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //         label: Some("FFT Command Encoder"),
    //     });
    //     // // encoder.copy_buffer_to_buffer(
    //     // //     &upload_staging,
    //     // //     0,
    //     // //     &buffer_a,
    //     // //     0,
    //     // //     (len * std::mem::size_of::<Complex>()) as u64,
    //     // // );
    //     // // 计算工作组维度
    //     let batches = len as u32 / fft_len;

    //     // 运行所有 FFT 阶段 - 都需要 fft_len/2 个线程
    //     {
    //         let threads_per_fft = fft_len / 2; // 每个 FFT 需要 fft_len/2 个线程
    //         //let total_threads = batches * threads_per_fft;
    //         let workgroup_len = 64;
    //         // let workgroups_x = (total_threads + workgroup_len - 1) / workgroup_len; // 向上取整
    //         let x = (threads_per_fft / workgroup_len).max(1);
    //         let y = (buffer_a.size() / 8 / fft_len as u64) as u32; //一个data中有2个u32，一个u32有4个byte
    //         let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //             label: None,
    //             timestamp_writes: None,
    //         });

    //         cpass.set_pipeline(&pipeline);
    //         cpass.set_bind_group(0, &bind_group, &[]);

    //         // 设置推送常量
    //         cpass.set_push_constants(0, &fft_len.to_le_bytes());
    //         // push_data.extend_from_slice(&stage.to_le_bytes());
    //         for i in 0..total_stages {
    //             cpass.set_push_constants(4, &i.to_le_bytes());

    //             cpass.dispatch_workgroups(x, y, 1);
    //         }
    //     }
    //     //     // 结果总是在 buffer_b 中
    //     encoder.copy_buffer_to_buffer(
    //         &buffer_b,
    //         0,
    //         &staging_buffer,
    //         0,
    //         (len * std::mem::size_of::<Complex>()) as u64,
    //     );
      
    //     queue.submit(Some(encoder.finish()));
    //     //读回结果
    //     buffer_slice.map_async(wgpu::MapMode::Read, |result| {
    //         result.unwrap();
    //     });
    //     device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    //      let data_mapped = buffer_slice.get_mapped_range();
    //      ans.copy_from_slice(bytemuck::cast_slice(&data_mapped));
    //      drop(data_mapped);
    //      staging_buffer.unmap();

    // }
    println!("执行时间: {:?}", timer.elapsed());
    println!("前几个结果: {:?}", &ans[..4]);
}
