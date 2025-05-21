use num_complex::Complex32 as Complex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建FFT规划器
    let planner = fft_wgpu::FFTPlanner::new().await;
    
    // 基本参数
    let fft_len = 512;
    let fft_len_u32 = fft_len as u32;
    let batch_size = 500 * 5;
    let data_size = fft_len * batch_size;
    
    // 创建信号和卷积核数据
    let data = vec![Complex::new(3.0, 0.0); data_size];
    let kernel = vec![Complex::new(5.0, 0.0); data_size];
    
    
    
    // 创建FFT处理器
    let forward_fft = planner.create_forward(fft_len_u32);
    let inverse_fft = planner.create_inverse(fft_len_u32);
    let multiply = planner.create_multiply();
    

    // 创建缓冲区
    let data_buffer = planner.create_buffer_from_data(&data);
    let kernel_buffer = planner.create_buffer_from_data(&kernel);
    let temp_buffer1 = planner.create_empty_buffer(data_size);
    let temp_buffer2 = planner.create_empty_buffer(data_size);
    let result_buffer = planner.create_empty_buffer(data_size);
    let staging_buffer = planner.create_staging_buffer(data_size);
    // 第一次FFT处理
    let mut encoder = planner.create_encoder();
    
    // 对信号执行FFT
    let data_freq = forward_fft.proc(&mut encoder, &data_buffer, &temp_buffer1);
    // 对卷积核执行FFT
    let kernel_freq = forward_fft.proc(&mut encoder, &kernel_buffer, &temp_buffer2);
    
    // 执行频域乘法
    multiply.proc(&mut encoder, data_freq, kernel_freq, &result_buffer);
    
    // 执行逆FFT
    let ifft_result = inverse_fft.proc(&mut encoder, &result_buffer, &temp_buffer1);
    
    // 复制结果到暂存缓冲区
    encoder.copy_buffer_to_buffer(
        ifft_result,
        0,
        &staging_buffer,
        0,
        (data_size * std::mem::size_of::<Complex>()) as u64
    );
    
    planner.submit_commands(encoder);
    
    let result2 = planner.read_buffer(&staging_buffer, data_size).await;
    println!("结果前10个元素: {:?}", &result2[..10]);
    println!("结果512-520: {:?}", &result2[512..520]);
    
    // 演示如何重用同一个FFT处理器处理不同数据
    println!("\n重用FFT处理器处理不同数据：");
    
    // 准备新数据
    let data2 = vec![Complex::new(2.0, 1.0); data_size];
    planner.write_buffer(&data_buffer, &data2);
    
    // 重用相同的Forward实例，但提供不同的缓冲区
    let mut encoder = planner.create_encoder();
    let signal2_freq = forward_fft.proc(&mut encoder, &data_buffer, &temp_buffer1);
    
    // 继续处理
    multiply.proc(&mut encoder, signal2_freq, kernel_freq, &result_buffer);
    let ifft_result2 = inverse_fft.proc(&mut encoder, &result_buffer, &temp_buffer1);
    
    encoder.copy_buffer_to_buffer(
        ifft_result2,
        0,
        &staging_buffer,
        0,
        (data_size * std::mem::size_of::<Complex>()) as u64
    );
    
    planner.submit_commands(encoder);
    
    let result3 = planner.read_buffer(&staging_buffer, data_size).await;
    println!("新数据结果前10个元素: {:?}", &result3[..10]);
    
    Ok(())

}