// filepath: /Users/wenhui/Desktop/eagle_sense/code/fft_wgpu/src/examples/draft_basic_f16.rs
use fft_wgpu::typed_buffer;
use num_complex::Complex;
use half::f16;

#[tokio::main]
async fn main() {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();
    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let mut required_features = adapter.features();
    if !required_features.contains(wgpu::Features::SHADER_F16) {
        panic!("当前 GPU 适配器不支持 SHADER_F16 特性，无法运行 f16 版本的 FFT。");
    }

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                label: Some("GPU Device"),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("设备是否支持时间戳查询: {}", supports_timestamps);

    let data = vec![Complex::new(f16::from_f32(5.0), f16::from_f32(0.0)); 512 * 500 * 5];
    let len = data.len();

    let mut ans = vec![Complex::new(f16::ZERO, f16::ZERO); len];

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex<f16>>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let upload_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex<f16>>()) as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex<f16>>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let input_arr: typed_buffer::Array<Complex<f16>> = typed_buffer::Array::new(src);
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex<f16>>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_arr: typed_buffer::Array<Complex<f16>> = typed_buffer::Array::new(output);

    let mut fft_forward = fft_wgpu::draft_fft_f16::Processor::new(
        &device,
        &queue,
        fft_wgpu::draft_fft_f16::Config {
            fft_len: 512,
            recipe: fft_wgpu::draft_fft_f16::Recipe::Radix2,
            direction: fft_wgpu::draft_fft_f16::Direction::Forward,
        },
    )
    .unwrap();
    
    // Inverse might not be fully implemented or tested for f16 in the same way, 
    // but keeping the structure if needed. For now commenting out as in draft_basic.rs it was used but here we focus on forward.
    // let mut fft_inverse = fft_wgpu::draft_fft_f16::Processor::new(
    //     &device,
    //     &queue,
    //     fft_wgpu::draft_fft_f16::Config {
    //         fft_len: 512,
    //         recipe: fft_wgpu::draft_fft_f16::Recipe::Radix2,
    //         direction: fft_wgpu::draft_fft_f16::Direction::Inverse,
    //     },
    // )
    // .unwrap();

    let buffer_slice = staging_buffer.slice(..);
    let upload_buffer_slice = upload_buffer.slice(..);

    let ts_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Timestamp Query Set"),
        ty: wgpu::QueryType::Timestamp,
        count: 4,
    });

    let ts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Buffer"),
        size: 8 * 4, // 8字节 * 4个时间戳
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::MAP_READ
            | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let timer = std::time::Instant::now();

    let query_flag = false;

    for i in 0..1000 {
        {
            if i == 0 {
                upload_buffer_slice.map_async(wgpu::MapMode::Write, |_| {});
                while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {
                    // std::thread::sleep(std::time::Duration::from_micros(1));
                }
                
                upload_buffer_slice
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(data.as_slice()));
                upload_buffer.unmap();
            }

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.write_timestamp(&ts_query_set, 0);
            encoder.copy_buffer_to_buffer(
                &upload_buffer,
                0,
                &input_arr.inner,
                0,
                input_arr.inner.size(),
            );
            encoder.write_timestamp(&ts_query_set, 1);

            queue.submit(Some(encoder.finish()));
        }

        fft_forward.proc(&input_arr, &output_arr);
        
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.write_timestamp(&ts_query_set, 2);

        encoder.copy_buffer_to_buffer(
            &output_arr.inner,
            0,
            &staging_buffer,
            0,
            output_arr.inner.size(),
        );

        encoder.write_timestamp(&ts_query_set, 3);

        queue.submit(Some(encoder.finish()));

        // 下一次的数据
        {
            let (tx, rx) = std::sync::mpsc::channel();
            upload_buffer_slice.map_async(wgpu::MapMode::Write, move |_| {
                let _ = tx.send(());
            });

            device.poll(wgpu::PollType::Poll);
            while rx.try_recv().is_err() {
                device.poll(wgpu::PollType::Poll);
            }

            upload_buffer_slice
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(data.as_slice()));
            upload_buffer.unmap();
        }

        buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});
        while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {}


        ans.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()));
        staging_buffer.unmap();

        if query_flag {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            encoder.resolve_query_set(&ts_query_set, 0..4, &ts_buffer, 0);
            queue.submit(Some(encoder.finish()));

            ts_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {}

            let ts = ts_buffer.slice(..).get_mapped_range();
            let tss: &[u64] = bytemuck::cast_slice(&ts);
            
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
    dbg!(timer.elapsed());

    // dbg!(&ans[0..10]);

}
