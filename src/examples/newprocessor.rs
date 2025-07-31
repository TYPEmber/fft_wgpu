use fft_wgpu::SegmentedTransfer;
use fft_wgpu::{FftDirection, FftProcessor, FftRadix, Multiply};
use num_complex::Complex32 as Complex;
use std::sync::Arc;
#[tokio::main]
async fn main() {
    //     // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN, //默认后端（着色器等）
        //     // //默认dx12编译器
        flags: wgpu::InstanceFlags::empty(), //没有额外的标志，行为

        backend_options: Default::default(),
    });
    //     let instance = wgpu::Instance::default();
    //     // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    //     //dbg!(adapter.limits());

    //     // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //     //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                //  required_features: wgpu::Features::empty(),
                // required_features: adapter.features(),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: adapter.limits(),
                label: Some("GPU Device"),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    let data = vec![Complex::new(5.0, 0.0); 512 * 500 * 5];
    let len = data.len();
    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);
    // let len_u32 = len as u32;
    let mut ans = vec![Complex::ZERO; len];
    let src = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let src1 = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let fft_forward = FftProcessor::new(
        
        device_arc.clone(),
        queue_arc.clone(),
        &src,
        512,
        FftDirection::Forward,
        FftRadix::Radix2,
    )
    .unwrap();

    let fft_forward1 = FftProcessor::new(
        device_arc.clone(),
        queue_arc.clone(),
        &src1,
        512,
        FftDirection::Forward,
        FftRadix::Radix2,
    )
    .unwrap();

    let multiply = Multiply::new(
        device_arc.clone(),
        queue_arc.clone(),
        fft_forward.get_output_buffer(),
        fft_forward1.get_output_buffer(),
    )
    .unwrap();

    let fft_inverse = FftProcessor::new(
        device_arc.clone(),
        queue_arc.clone(),
        multiply.get_output_buffer(),
        512,
        FftDirection::Inverse,
        FftRadix::Radix2,
    )
    .unwrap();
    let custom_transfer = SegmentedTransfer::new::<Complex, f32>(
        Arc::clone(&device_arc),
        Arc::clone(&queue_arc),
        data.len(),
        &[1.0, 1.0], // 任何数字都可以！
        None,
    );
    custom_transfer.upload_data(&data, &src).await;
    custom_transfer.upload_data(&data, &src1).await;
    let mut encoder =
        device_arc.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    fft_forward.proc(&mut encoder);
    fft_forward1.proc(&mut encoder);
    multiply.proc(&mut encoder);
    fft_inverse.proc(&mut encoder);
    queue_arc.submit(Some(encoder.finish()));

    let _result: Vec<Complex> = custom_transfer
        .download_data(fft_inverse.get_output_buffer(), Some(&mut ans))
        .await;
    println!("前几个结果: {:?}", &ans[..4]);
}
