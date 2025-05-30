use num_complex::Complex32 as Complex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {


        // 创建FFT规划器
        let planner = fft_wgpu::FFTPlanner::new().await;
        
        // 基本参数
        let fft_len = 1024;
        let fft_len_u32 = fft_len as u32;
        let batch_size = 500 * 5;
        let data_size = fft_len * batch_size;
        
        // 创建信号和卷积核数据
        let data = vec![Complex::new(5.0, 0.0); data_size];
        let kernel = vec![Complex::new(3.0, 0.0); data_size];
        
        // 高层次API：一步执行卷积
        
        // 创建处理器，配置收缩阈值和最小容量
        let mut forward_fft = planner.create_forward(fft_len_u32, data_size);
      //  forward_fft.set_shrink_threshold(3.0);  // 当容量 > 所需的3倍时收缩
       // forward_fft.set_min_capacity(2048);     // 最小保持2048个元素的容量
        
        let mut inverse_fft = planner.create_inverse(fft_len_u32, data_size);
       // inverse_fft.set_shrink_threshold(3.0);
       // inverse_fft.set_min_capacity(2048);
        
        let mut multiply = planner.create_multiply(data_size);
       // multiply.set_shrink_threshold(3.0);
       // multiply.set_min_capacity(2048);
        
        // 创建数据缓冲区
        let data_buffer = planner.create_buffer_from_data(&data);
        let kernel_buffer = planner.create_buffer_from_data(&kernel);
        let staging_buffer = planner.create_staging_buffer(data_size);
        
        // 执行命令
        let mut encoder = planner.create_encoder();
        
        // 处理大数据
       // println!("\n处理大数据 ({}个元素)：", data_size);
       // println!("初始容量: {}", forward_fft.current_capacity());
     //  let kernel_freq = forward_fft.proc(&mut encoder, &kernel_buffer);
        forward_fft.proc_inplace(&mut encoder, & data_buffer);
       
        forward_fft.proc_inplace(&mut encoder, & kernel_buffer);
        
        let product_freq = multiply.proc(&mut encoder, &data_buffer, &kernel_buffer);
        let result = inverse_fft.proc(&mut encoder, product_freq);
        encoder.copy_buffer_to_buffer(
           result,
            0,
            &staging_buffer,
            0,
            (data_size * std::mem::size_of::<Complex>()) as u64
        );
       
        
        planner.submit_commands(encoder);
        let result = planner.read_buffer(&staging_buffer, data_size).await;
        println!(" {:?}", &result[..5]);
      
    Ok(())

}