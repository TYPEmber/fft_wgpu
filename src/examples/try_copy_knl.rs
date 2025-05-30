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

    let data = vec![Complex::new(5.0, 0.0); 512 * 500 * 5];
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
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
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

    queue.write_buffer(&src, 0, bytemuck::cast_slice(data.as_slice()));
    // A command encoder executes one or many pipelines.
    let shader_source = include_str!("../kernel/copy.wgsl"); //读取着色器代码文本
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE, //只在计算阶段可见
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, //可读写
                    has_dynamic_offset: false,                                 //没有动态偏移
                    //wgpu: COPY_BUFFER_ALIGNMENT 缓冲区对齐要求，偏移量以及大小必须是它的倍数
                    min_binding_size: None, //没有最小绑定大小
                },
                count: None,
            }, //绑定资源数量无限制（ty数量无限制）
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE, //只在计算阶段可见
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, //可读写
                    has_dynamic_offset: false,                                 //没有动态偏移
                    //wgpu: COPY_BUFFER_ALIGNMENT 缓冲区对齐要求，偏移量以及大小必须是它的倍数
                    min_binding_size: None, //没有最小绑定大小
                                            //buffer的最小值，但一定要大于wgsl中的绑定长度
                },
                count: None, //绑定资源数量无限制（ty数量无限制）
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.as_entire_binding(), //将之前的存储缓冲区转换成绑定组，适当的封装与转换
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: staging_buffer.as_entire_binding(),
            },
        ],
    });

    //可以通过pipeline来创建bind_group_layout，但是不同的pipeline之间派生的bind_group_layout是不兼容的，即使客观相同
    // 创建计算管道
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout], //定义关联的绑定组布局
        push_constant_ranges: &[],                 //没有设置推送常量的范围
    });

    // let pipeline_layout = device.create_pipeline_layout(&[
    //     &bind_group_layout_0, // 对应 @group(0)
    //     &bind_group_layout_1, // 对应 @group(1)
    // ]);
    //可以像上面这样依次对应不同的group
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &shader_module,    //指定着色器模板
        entry_point: Some("main"), //告诉计算管线从这个 main 函数处开始执行着色器里的计算逻辑              //没有其他存储设置
        compilation_options: wgpu::PipelineCompilationOptions::default(), //默认编译选项
        cache: None,
    });

    let timer = std::time::Instant::now();

    for _ in 0..1000 {
        // queue.write_buffer(&src, 0, bytemuck::cast_slice(data.as_slice()));

        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                //创建计算通道
                label: Some("Compute Pass"),
                timestamp_writes: None, //未配置时间戳
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            //index为0，表示第一个bind_group,如果有多个bgl与bg，那么需要和index一一对应
            let workgroup_len = 64;
            let total_elements = (src.size() / 8) as u32;
            let x = 1024 / workgroup_len;
            let y = ((total_elements + 1024 - 1) / 1024).max(1); //一个字节是8个bit
            let z = 1;
            compute_pass.dispatch_workgroups(x, y, z); //分发计算任务
        }

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

        // Gets contents of buffer

        // // Since contents are got in bytes, this converts these bytes back to u32
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory
    }

    dbg!(timer.elapsed());
}
