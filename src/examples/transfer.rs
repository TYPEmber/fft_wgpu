use num_complex::Complex32 as Complex;
use std::f64::consts::PI;
use std::sync::Arc;
use fft_wgpu::SegmentedTransfer;
use wgpu::util::DeviceExt;
#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    // 请求适配器
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    // 请求设备
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);
    let data = vec![Complex::new(1.0, 0.0); 512 * 500 * 5];
    let len = data.len();
    let mut ans = vec![Complex::ZERO; len];
    let buffer_a = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let buffer_b = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer B"),
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            //| wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // 1. 使用预定义的三段分割

    // 2. 使用自定义比例
    let custom_transfer = SegmentedTransfer::new::<Complex, f32>(
        Arc::clone(&device_arc),
        Arc::clone(&queue_arc),
        data.len(),
        &[1.0,1.0], // 任何数字都可以！
        None,
    );

    // 3. 使用整数比例
    // let integer_transfer = SegmentedTransfer::new::<Complex, i32>(
    //     Arc::clone(&device_arc),
    //     Arc::clone(&queue_arc),
    //     data.len(),
    //     &[1, 3, 6], // 简单的整数比例
    //     None,
    // );
    let cs_module = device_arc.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Cooley-Tukey Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../kernel/fftct.wgsl"
        ))),
    });

    // 创建绑定组布局
    let bgl = device_arc.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFT Bind Group Layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // 创建管线布局
    let ppl = device_arc.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });

    // 创建计算管线
    let pipeline = device_arc.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFT Cooley-Tukey Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // FFT 参数
    let fft_len: u32 = 512;
    // 总阶段数：1个位反转+第一阶段蝶形运算，然后是剩余的 log2(fft_len)-1 个蝶形运算阶段
    let total_stages = f32::log2(fft_len as f32) as u32;

    // 预计算旋转因子
    let n = fft_len as usize;
    let mut twiddles = Vec::with_capacity(n / 2);

    for k in 0..n / 2 {
        let theta = -2.0 * PI * (k as f64) / (n as f64);
        twiddles.push(Complex::new(theta.cos() as f32, theta.sin() as f32));
    }

    let twiddle_buffer = device_arc.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddle Buffer"),
        contents: bytemuck::cast_slice(&twiddles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 创建绑定组
    let bind_group = device_arc.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FFT Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: twiddle_buffer.as_entire_binding(),
            },
        ],
    });

    // 使用示例
    let timer = std::time::Instant::now();

    for _ in 0..1000
     {
        custom_transfer.upload_data(&data, &buffer_a).await;
        // let mut compute_encoder =
        //     device_arc.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //         label: Some("FFT Compute Encoder"),
        //     });

        // // 计算工作组维度
        // let threads_per_fft = fft_len / 2;
        // let workgroup_len = 64;
        // let x = (threads_per_fft / workgroup_len).max(1);
        // let y = (buffer_a.size() / 8 / fft_len as u64) as u32;

        // let mut cpass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //     label: None,
        //     timestamp_writes: None,
        // });

        // cpass.set_pipeline(&pipeline);
        // cpass.set_bind_group(0, &bind_group, &[]);

        // // 运行所有FFT阶段
        // cpass.set_push_constants(0, &fft_len.to_le_bytes());
        // for i in 0..total_stages {
        //     cpass.set_push_constants(4, &i.to_le_bytes());
        //     cpass.dispatch_workgroups(x, y, 1);
        // }

        // drop(cpass); // 结束计算通道
        // queue_arc.submit(Some(compute_encoder.finish()));

        let _result: Vec<Complex> = custom_transfer
            .download_data(&buffer_b, Some(&mut ans))
            .await;
    }
    println!("执行时间: {:?}", timer.elapsed());
    println!("前几个结果: {:?}", &ans[..4]);
}
