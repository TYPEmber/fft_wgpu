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
            None,
        )
        .await
        .unwrap();
    let mut data = vec![Complex::new(3.0, 0.0); 512 * 500 * 5];
    let data_len = data.len();
    let mut kernel = vec![Complex::new(1.0, 0.0); 512];
    kernel.append(&mut data);
    let total_len = kernel.len();
    let mut ans = vec![Complex::ZERO; data_len];
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data_len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (total_len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let fft_len = 512;
    let fft_forward = fft_wgpu::Forward::new(&device, &queue, &src, fft_len);
    let buffer_slice = staging_buffer.slice(..);
    queue.write_buffer(&src, 0, bytemuck::cast_slice(kernel.as_slice()));
    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let output = fft_forward.proc(&mut encoder);
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data_len* std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mul_cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../kernel/multiply2.wgsl"
        ))),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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
        ],
    });

    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..4,
        }],
    });

    let pipeline_multiply = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&ppl),
        module: &mul_cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    });

    let bind_group_forward = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline_multiply.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result.as_entire_binding(),
            },
        ],
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline_multiply);
        compute_pass.set_bind_group(0, &bind_group_forward, &[]);

        let workgroup_len = 64;
        let x = 1024 / workgroup_len; //每个x对应一组fft运算
        let y = (data_len / 1024) as u32; //一个data中有2个u32，一个u32有4个byte
        let z = 1;
        compute_pass.set_push_constants(0,&fft_len.to_le_bytes());
        compute_pass.dispatch_workgroups(x, y, z);
    }
    let fft_inverse = fft_wgpu::Inverse::new(&device, &queue, &result, 512);
    let output = fft_inverse.proc(&mut encoder);
    encoder.copy_buffer_to_buffer(
        output,
        0,
        &staging_buffer,
        0,
        (data_len * std::mem::size_of::<Complex>()) as u64,
    );

    queue.submit(Some(encoder.finish()));
    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    let data1 = buffer_slice.get_mapped_range();

    bytemuck::cast_slice(&data1).clone_into(&mut ans);
    println!("{:?}", &ans[..10]);
    println!("{:?}", &ans[512..520]);
    drop(data1);
    staging_buffer.unmap(); 
}
