use num_complex::Complex32 as Complex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建FFT规划器
    let planner = fft_wgpu::FFTPlanner::new().await;
    
    // 基本参数
    let fft_len = 512 ;
    let  fft_len_u32 = fft_len as u32;
    let batch_size = 500 * 5;
    let data_size = fft_len * batch_size;
    
    // 创建信号和卷积核数据
    let data = vec![Complex::new(3.0, 0.0); data_size];
    let kernel = vec![Complex::new(5.0, 0.0); data_size];
    let data_buffer = planner.create_buffer_from_data(&data);
    let kernel_buffer = planner.create_buffer_from_data(&kernel);
    let staging_buffer = planner.create_staging_buffer(data_size); // 用于读回结果
    
    // 创建操作计算器
    let data_fft = planner.create_forward(&data_buffer, fft_len_u32);
    let kernel_fft = planner.create_forward(&kernel_buffer, fft_len_u32);
    
    // 执行命令
    let mut encoder = planner.create_encoder();
    
    // 执行信号和卷积核的FFT
    let data_output = data_fft.proc(&mut encoder);
    let kernel_output = kernel_fft.proc(&mut encoder);
    
    // 频域乘法
    let multiply = planner.create_multiply(data_output, kernel_output);
    let product_output = multiply.proc(&mut encoder);
    
    // 逆向FFT
    let ifft = planner.create_inverse(product_output, fft_len_u32);
    let result = ifft.proc(&mut encoder);
    
    // 复制结果到暂存缓冲区
    encoder.copy_buffer_to_buffer(
        result,
        0,
        &staging_buffer,
        0,
        (data_size * std::mem::size_of::<Complex>()) as u64
    );
    
    // 提交命令
    planner.submit_commands(encoder);
    
    // 读取结果
    let result2 = planner.read_buffer(&staging_buffer, data_size).await;
    println!("前10个元素: {:?}", &result2[..10]);
    println!(" {:?}", &result2[512..520]);
    
    
     Ok(())
}