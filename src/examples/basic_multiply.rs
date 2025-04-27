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
    let data1 = vec![Complex::new(2.0, 0.0); 512*500*5 ];
    let data2 = vec![Complex::new(3.0, 0.0); 512*500*5 ];
    let len = data1.len();
    let mut ans = vec![Complex::ZERO; len];

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let src2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let multiply=fft_wgpu::Multiply::new(&device, &queue, &src1, &src2);
    let buffer_slice = staging_buffer.slice(..);
    queue.write_buffer(&src1, 0, bytemuck::cast_slice(data1.as_slice()));
    queue.write_buffer(&src2, 0, bytemuck::cast_slice(data2.as_slice()));
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
          device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let output = multiply.proc(&mut encoder);
        //let output=fft_forward.buffer_a;
         //let output = fft_forward.proc(&mut encoder);
        //let output = fft_forward_2.proc(&mut encoder);

        encoder.copy_buffer_to_buffer(
            output,
            0,
            &staging_buffer,
            0,
            (len * std::mem::size_of::<Complex>()) as u64,
        );
        // encoder.copy_buffer_to_buffer(
        //     output,
        //     0,
        //     &staging_buffer,
        //     0,
        //     (len * std::mem::size_of::<Complex>()) as u64,
        // );

        queue.submit(Some(encoder.finish()));
       // queue.submit(None);
        // let rn = fft_forward.round_num.slice(..);

        // rn.map_async(wgpu::MapMode::Read, move |_| {});

         //device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        // let a: Vec<u8> = rn.get_mapped_range().iter().copied().collect();
        // dbg!(a);
        // fft_forward.round_num.unmap();

        // Note that we're not calling `.await` here.
        
         buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});
         device.poll(wgpu::Maintain::wait()).panic_on_timeout();
         let data = buffer_slice.get_mapped_range();
       
        
        // Gets contents of buffer
       
       
        // // Since contents are got in bytes, this converts these bytes back to u32
         bytemuck::cast_slice(&data).clone_into(&mut ans);
        
         println!("{:?}", &ans[..10]);
         println!("{:?}", &ans[512..520]);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
         drop(data);
         staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory
}