use fft_wgpu::typed_buffer;
use num_complex::Complex32 as Complex;

/// 主函数 - 演示使用WebGPU进行FFT计算的基本流程
///
/// 这个示例展示了如何：
/// 1. 初始化WebGPU实例和设备
/// 2. 创建GPU缓冲区用于数据传输
/// 3. 使用自定义的FFT处理器进行正向和逆向FFT计算
/// 4. 测量GPU计算性能和PCIe带宽
#[tokio::main]
async fn main() {
    // ========== WebGPU初始化阶段 ==========
    // 创建WebGPU实例 - 这是与GPU通信的入口点
    // 负责发现系统上的 GPU 硬件并初始化底层的图形后端（如 Vulkan, DirectX 12, Metal 等）
    // 使用默认配置，会自动选择可用的后端
    let instance = wgpu::Instance::default();

    // 请求适配器 - adapter 代表物理 GPU 设备
    // PowerPreference::HighPerformance 优先选择高性能GPU（如独立显卡）
    // 这是一个 async 方法，因为查询硬件需要时间，所以需要 await
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    // ========== 设备和队列创建 ==========
    // 创建GPU设备和命令队列
    // device: 是与 GPU 交互的主要接口，用于创建GPU资源（缓冲区、纹理、着色器等）
    // queue: 用于提交命令到GPU执行
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            // 请求适配器支持的所有功能特性
            required_features: adapter.features(),
            // 使用适配器的默认限制
            required_limits: adapter.limits(),
            label: Some("GPU Device"),
            experimental_features: unsafe {
                wgpu::ExperimentalFeatures::enabled()
            },
            ..Default::default()
        })
        .await
        .unwrap();

    // 检查GPU是否支持时间戳查询功能（用于性能分析）
    // 时间戳查询：这是一种高级功能，允许你在 GPU 命令执行的精确时刻记录时间戳。
    // 这对于精确测量 GPU 上的计算耗时（Profiling）至关重要，因为 CPU 端的计时器无法准确反映 GPU 内部的执行情况
    let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("设备是否支持时间戳查询: {}", supports_timestamps);

    // ========== 数据准备 ==========
    // 创建输入数据：512 * 500 * 5 = 1,280,000 个复数元素
    // 所有元素初始化为 (5.0 + 0.0i)
    let data = vec![Complex::new(5.0, 0.0); 512 * 500 * 5];
    let len = data.len();
    //  let a:u8=1.0;
    // let buffer=device.create_buffer_from_hal{

    // let mut data_cpu = data
    //     .iter()
    //     .map(|c| rustfft::num_complex::Complex::new(c.re, 0.0))
    //     .collect::<Vec<_>>();
    // let fft = rustfft::FftPlanner::new().plan_fft_forward(16);
    // fft.process(&mut data_cpu);
    // fft.process(&mut data_cpu);
    // println!("{:?}", &data_cpu[..]);

    // 创建CPU端结果缓冲区，用于存储从GPU读取的计算结果
    let mut ans = vec![Complex::ZERO; len];

    // ========== GPU缓冲区创建 ==========
    // 创建暂存缓冲区（Staging Buffer）- 用于从GPU读取数据到CPU
    // MAP_READ: 允许CPU映射读取此缓冲区
    // COPY_DST: 允许作为复制操作的目标
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 创建上传缓冲区（Upload Buffer）- 用于从CPU上传数据到GPU
    // MAP_WRITE: 允许CPU映射写入此缓冲区
    // COPY_SRC: 允许作为复制操作的源
    let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // 创建GPU存储缓冲区 - 用于FFT计算的输入数据
    // STORAGE: 允许在着色器中作为存储缓冲区使用
    // COPY_DST/SRC: 允许进行缓冲区间的数据复制
    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // 使用typed_buffer包装器，提供类型安全的数组操作
    let input_arr: typed_buffer::Array<Complex> = typed_buffer::Array::new(src);

    // 创建GPU存储缓冲区 - 用于FFT计算的输出结果
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_arr: typed_buffer::Array<Complex> = typed_buffer::Array::new(output);

    // ========== FFT处理器初始化 ==========
    // 创建正向FFT处理器
    // fft_len: 512 - 单个FFT变换的长度
    // 处理器会自动将输入数据划分为多个批次（Batch）并行处理
    // 本例中：1,280,000 / 512 = 2,500 个FFT变换
    let mut fft_forward = fft_wgpu::draft_fft::Processor::new(
        &device,
        &queue,
        fft_wgpu::draft_fft::Config {
            fft_len: 512,
            recipe: fft_wgpu::draft_fft::Recipe::Radix2,
            direction: fft_wgpu::draft_fft::Direction::Forward,
        },
    )
    .unwrap();

    // 创建逆向FFT处理器（当前未使用，可以注释掉以避免警告）
    // let mut fft_inverse = fft_wgpu::draft_fft::Processor::new(
    //     &device,
    //     &queue,
    //     fft_wgpu::draft_fft::Config {
    //         fft_len: 512,
    //         recipe: fft_wgpu::draft_fft::Recipe::Radix2,
    //         direction: fft_wgpu::draft_fft::Direction::Inverse,
    //     },
    // )
    // .unwrap();

    // 创建缓冲区切片，用于异步映射操作
    let buffer_slice = staging_buffer.slice(..);
    let upload_buffer_slice = upload_buffer.slice(..);

    // ========== 性能分析设置 ==========
    // 创建时间戳查询集 - 用于测量GPU执行时间
    // count: 4 - 需要4个时间戳点来测量不同阶段的耗时
    let ts_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Timestamp Query Set"),
        ty: wgpu::QueryType::Timestamp,
        count: 4,
    });

    // 创建时间戳结果缓冲区 - 存储查询结果
    let ts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Buffer"),
        size: 8 * 4, // 8字节 * 4个时间戳（每个时间戳8字节）
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::MAP_READ
            | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    // CPU端计时器 - 测量整个1000次迭代的总时间
    let timer = std::time::Instant::now();

    // 控制是否启用GPU时间戳查询（默认关闭）
    let query_flag = false;

    for i in 0..1000 {
        // queue.write_buffer(&input_arr.inner, 0, bytemuck::cast_slice(data.as_slice()));
        // let timer = std::time::Instant::now();
        // queue.submit([]);
        // ========== 数据上传阶段 ==========
        {
            // 只在第一次迭代时上传初始数据到GPU
            if i == 0 {
                // 异步映射上传缓冲区到CPU可写状态
                upload_buffer_slice.map_async(wgpu::MapMode::Write, |_| {});
                // 等待映射操作完成
                while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {
                    std::thread::sleep(std::time::Duration::from_micros(1));
                }
                // _let timer = std::time::Instant::now(); // 调试用计时器（已注释）

                // 将 CPU 端的数据写入到已经映射（Mapped）的 GPU 缓冲区中
                // get_mapped_range_mut 返回一个可变视图，代表了 GPU 缓冲区在 CPU 内存空间的映射区域，通过该视图，CPU 能直接修改这块内存的内容，前提是必须调用 map_async 并等待映射完成
                // copy_from_slice 非常底层的、高效内存复制操作
                // bytemuck 库处理纯数据类型转换，cast_slice 安全地将 Complex 类型的切片转换为字节切片 &[u8]，不需要重新分配内存，只是改变了看待这块内存的方式
                upload_buffer_slice
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(data.as_slice()));
                // 解除缓冲区映射，使其可用于GPU操作
                upload_buffer.unmap();
            }

            // 创建命令编码器，用于记录GPU命令
            // 现代图形 API 的工作模式是记录 + 提交，encoder 负责记录
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // 记录时间戳：开始PCIe上传
            encoder.write_timestamp(&ts_query_set, 0);

            // 将数据从上传缓冲区复制到计算输入缓冲区
            encoder.copy_buffer_to_buffer(
                &upload_buffer,
                0,
                &input_arr.inner,
                0,
                input_arr.inner.size(),
            );

            // 记录时间戳：结束PCIe上传
            encoder.write_timestamp(&ts_query_set, 1);

            // 提交命令到GPU队列执行
            queue.submit(Some(encoder.finish()));
        }

        // ========== FFT计算阶段 ==========
        // 执行正向FFT计算
        fft_forward.proc(&input_arr, &output_arr);

        // 创建新的命令编码器用于后续操作
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 记录时间戳：FFT计算完成
        encoder.write_timestamp(&ts_query_set, 2);

        // ========== 数据下载阶段 ==========
        // 将FFT计算结果从输出缓冲区复制到暂存缓冲区（准备读取到CPU）
        encoder.copy_buffer_to_buffer(
            &output_arr.inner,
            0,
            &staging_buffer,
            0,
            output_arr.inner.size(),
        );

        // 记录时间戳：PCIe下载完成
        encoder.write_timestamp(&ts_query_set, 3);

        // 提交命令到GPU队列
        queue.submit(Some(encoder.finish()));

        // ========== 准备下一次迭代的数据 ==========
        // 使用通道进行异步通知，避免阻塞整个队列
        {
            let (tx, rx) = std::sync::mpsc::channel();
            upload_buffer_slice.map_async(wgpu::MapMode::Write, move |_| {
                let _ = tx.send(());
            });

            // 注意：不能使用wait()，因为wait会等待队列中所有指令完成
            // 这里只需要等待map_async操作完成即可
            device.poll(wgpu::PollType::Poll);
            while rx.try_recv().is_err() {
                // std::thread::sleep(std::time::Duration::from_micros(1));
                device.poll(wgpu::PollType::Poll);
            }

            // _let timer = std::time::Instant::now(); // 调试用计时器（已注释）

            upload_buffer_slice
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(data.as_slice()));
            // 解除上传缓冲区映射
            upload_buffer.unmap();
        }

        // ========== 读取计算结果 ==========
        // 异步映射暂存缓冲区到CPU可读状态
        buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});
        // 等待所有GPU操作完成，确保数据已准备好
        while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {
            // std::thread::sleep(std::time::Duration::from_micros(1));
        }

        // _let timer = std::time::Instant::now(); // 调试用计时器（已注释）
        // 从GPU缓冲区读取计算结果到CPU内存
        ans.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()));
        // 解除暂存缓冲区映射
        staging_buffer.unmap();

        // ========== 性能分析（如果启用） ==========
        if query_flag {
            // 解析时间戳查询结果到时间戳缓冲区
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.resolve_query_set(&ts_query_set, 0..4, &ts_buffer, 0);
            queue.submit(Some(encoder.finish()));

            // 读取时间戳结果
            ts_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {}

            let ts = ts_buffer.slice(..).get_mapped_range();
            let tss: &[u64] = bytemuck::cast_slice(&ts);

            // 计算并打印性能指标
            println!(
                "PCIe upload bandwith: {} GB/s, dur: {} us",
                (input_arr.inner.size() as f64 / (1024.0 * 1024.0 * 1024.0))
                    / ((tss[1] - tss[0]) as f64 / 1e9),
                (tss[1] - tss[0]) as f64 / 1e3
            );
            println!("Calculate: {} us", (tss[2] - tss[1]) as f64 / 1e3);
            println!(
                "PCIe download bandwith: {} GB/s, dur: {} us",
                (input_arr.inner.size() as f64 / (1024.0 * 1024.0 * 1024.0))
                    / ((tss[3] - tss[2]) as f64 / 1e9),
                (tss[3] - tss[2]) as f64 / 1e3
            );
            drop(ts);
            ts_buffer.unmap();
        }
    }

    // 打印1000次迭代的总耗时
    dbg!(timer.elapsed());

    // 可以取消注释以下行来查看前10个计算结果
    // dbg!(&ans[0..10]);
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32 as Complex;
    #[tokio::test]
    // 在main函数末尾添加以下测试代码
    async fn test_fft() {
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        dbg!(adapter.limits());

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
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

        let data = vec![Complex::new(1.0, 0.0); 16];
        let len = data.len();

        // let mut data_cpu = data
        //     .iter()
        //     .map(|c| rustfft::num_complex::Complex::new(c.re, 0.0))
        //     .collect::<Vec<_>>();
        // let fft = rustfft::FftPlanner::new().plan_fft_forward(16);
        // fft.process(&mut data_cpu);
        // fft.process(&mut data_cpu);
        // println!("{:?}", &data_cpu[..]);

        let mut ans = vec![Complex::ZERO; len];

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * std::mem::size_of::<Complex>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let src = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * std::mem::size_of::<Complex>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let fft_forward = fft_wgpu::Forward::new(&device, &queue, &src, 512);
        // let fft_forward_2 = fft_wgpu::Forward::new(&device, &queue, &src, 16);

        let timer = std::time::Instant::now();

        for _ in 0..1000 {
            queue.write_buffer(&src, 0, bytemuck::cast_slice(data.as_slice()));
            // A command encoder executes one or many pipelines.
            // It is to WebGPU what a command buffer is to Vulkan.
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // let _output = fft_forward.proc(&mut encoder);
            // let  output= fft_forward.proc(&mut encoder);
            // // let output = fft_forward.proc(&mut encoder);
            // //let output = fft_forward_2.proc(&mut encoder);

            // encoder.copy_buffer_to_buffer(
            //     output,
            //     0,
            //     &staging_buffer,
            //     0,
            //     (len * std::mem::size_of::<Complex>()) as u64,
            // );

            queue.submit(Some(encoder.finish()));

            // let rn = fft_forward.round_num.slice(..);

            // rn.map_async(wgpu::MapMode::Read, move |_| {});

            // device.poll(wgpu::Maintain::wait()).panic_on_timeout();
            // let a: Vec<u8> = rn.get_mapped_range().iter().copied().collect();
            // dbg!(a);
            // fft_forward.round_num.unmap();

            // Note that we're not calling `.await` here.
            // let buffer_slice = staging_buffer.slice(..);

            // buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});

            device.poll(wgpu::Maintain::wait()).panic_on_timeout();

            // Gets contents of buffer
            //     let data = buffer_slice.get_mapped_range();

            //     // // Since contents are got in bytes, this converts these bytes back to u32
            //    // bytemuck::cast_slice(&data).clone_into(&mut ans);
            //     ans.copy_from_slice(bytemuck::cast_slice(&data));
            //     println!("{:?}", &ans[..16]);

            //     // With the current interface, we have to make sure all mapped views are
            //     // dropped before we unmap the buffer.
            //     drop(data);
            //     staging_buffer.unmap(); // Unmaps buffer from memory
            // If you are familiar with C++ these 2 lines can be thought of similarly to:
            //   delete myPointer;
            //   myPointer = NULL;
            // It effectively frees the memory
        }
        //device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        dbg!(timer.elapsed());
    }
}
