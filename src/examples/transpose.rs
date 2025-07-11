use num_complex::Complex32;
use std::mem::size_of;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    // 初始化WGPU
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features:  wgpu::Features::PUSH_CONSTANTS,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .unwrap();

    // 创建测试数据
    let dims = [512, 8, 512];
    let total_elements = dims.iter().product::<u32>() as usize;
    let mut input_data = vec![Complex32::new(0.0, 0.0); total_elements];
    for i in 0..total_elements {
        input_data[i] = Complex32::new(i as f32, (i * 2) as f32);
    }

    // 创建缓冲区
    let buffer_size = (total_elements * size_of::<Complex32>()) as u64;

    // 创建输入缓冲区
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // 创建结果读回缓冲区
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 创建FFT处理器
    let fft1 = fft_wgpu::Forward::new(&device, &queue, &input_buffer, 512);
    let transpose1 = fft_wgpu::TransposeProcessor::new(&device, &queue, &fft1.buffer_b, &dims);
    // 创建第一个转置处理器

    let fft2 = fft_wgpu::Forward::new(&device, &queue, &transpose1.output_buffer, 512);
    // 创建第二个FFT处理器（使用转置后的维度）

    // 创建第二个转置处理器（使用第一次转置后的维度）
    let transpose2 = fft_wgpu::TransposeProcessor::new(
        &device,
        &queue,
        &input_buffer,
        transpose1.get_output_dims(),
    );

    // 开始计时
    let timer = std::time::Instant::now();

    // 创建命令编码器
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    // 第一次FFT
    let fft1_output = fft1.proc(&mut encoder);

    // 第一次转置 (如果fft1_output与transpose1的input_buffer不同，需要更新)

    //let _transpose1_output = transpose1.proc(&mut encoder);

    // 第二次FFT
    //let _fft2_output = fft2.proc(&mut encoder);

    // 第二次转置 (如果fft2_output与transpose2的input_buffer不同，需要更新)

    let final_output = transpose2.proc(&mut encoder);

    // 复制最终结果到可读回的缓冲区
    encoder.copy_buffer_to_buffer(fft1_output, 0, &result_buffer, 0, buffer_size);

    // 提交命令
    queue.submit(Some(encoder.finish()));

    // 读取结果
    let buffer_slice = result_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<Complex32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        result_buffer.unmap();

        println!("处理完成，用时: {:?}", timer.elapsed());
        println!("原始维度: {:?}", dims);
        println!("第一次转置后维度: {:?}", transpose1.get_output_dims());
        println!("第二次转置后维度: {:?}", transpose2.get_output_dims());

        // 打印部分结果
        println!("结果前10个元素:");
        for i in 0..10.min(result.len()) {
            println!("[{}]: ({}, {})", i, result[i].re, result[i].im);
        }
    }
}
