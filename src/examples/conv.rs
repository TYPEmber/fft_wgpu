use num_complex::Complex32 as Complex;
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

    //dbg!(adapter.limits());

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                label: Some("GPU Device"),
                ..Default::default()
            },
            None
        )
        .await
        .unwrap();

    let data = vec![Complex::new(3.0, 0.0); 512*500*5];
    let len = data.len();

    let kernel=vec![Complex::new(1.0, 0.0); 512*500*5];
    let mut ans = vec![Complex::ZERO; len];
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let data_src = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let knl_src = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let data_forward = fft_wgpu::Forward::new(&device, &queue, &data_src, 512);
    let knl_forward = fft_wgpu::Forward::new(&device, &queue, &knl_src, 512);
    let buffer_slice = staging_buffer.slice(..);

    queue.write_buffer(&data_src, 0, bytemuck::cast_slice(data.as_slice()));
    queue.write_buffer(&knl_src, 0, bytemuck::cast_slice(kernel.as_slice()));
   
    let mut encoder =
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    
    let data_output = data_forward.proc(&mut encoder);
    let knl_output = knl_forward.proc(&mut encoder);
    let multiply=fft_wgpu::Multiply::new(&device, &queue, data_output, knl_output);
    
    let result = multiply.proc(&mut encoder);
    let fft_inverse = fft_wgpu::Inverse::new(&device, &queue, result, 512);
    let output = fft_inverse.proc(&mut encoder);
    encoder.copy_buffer_to_buffer(output, 0, &staging_buffer, 0, (len * std::mem::size_of::<Complex>()) as u64);
    
    queue.submit(Some(encoder.finish()));
    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    let data1 = buffer_slice.get_mapped_range();
   // Gets contents of buffer
   // // Since contents are got in bytes, this converts these bytes back to u32
    bytemuck::cast_slice(&data1).clone_into(&mut ans);
    println!("{:?}", &ans[..10]);
    println!("{:?}", &ans[512..520]);
   // With the current interface, we have to make sure all mapped views are
   // dropped before we unmap the buffer.
    drop(data1);
    staging_buffer.unmap(); // Unmaps buffer from memory

}