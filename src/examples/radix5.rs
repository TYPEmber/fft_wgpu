use num_complex::Complex32 as Complex;
use std::f64::consts::PI;
use std::sync::Arc;
use fft_wgpu::SegmentedTransfer;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

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
    
    // Test data - for radix-5 FFT, we need a length that's a power of 5
    // Using 5^3 = 125 as an example
    let fft_len = 125;
    let batch_count = 500;
    let data = vec![Complex::new(1.0, 3.0); fft_len * batch_count];
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
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // Use custom transfer
    let custom_transfer = SegmentedTransfer::new::<Complex, f32>(
        Arc::clone(&device_arc),
        Arc::clone(&queue_arc),
        data.len(),
        &[1.0, 1.0],
        None,
    );

    // Create radix-5 FFT shader module
    let cs_module = device_arc.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Radix-5 Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../kernel/fft5ct.wgsl"
        ))),
    });

    // Create bind group layout
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

    // Create pipeline layout
    let ppl = device_arc.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });

    // Create compute pipeline
    let pipeline = device_arc.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FFT Radix-5 Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // FFT parameters
    let fft_len: u32 = 125; // 5^3 for radix-5 FFT
    
    // For radix-5, total stages is log5(fft_len)
    let total_stages = (f32::log2(fft_len as f32) / f32::log2(5.0)).round() as u32;

    // Precompute twiddle factors
    let n = fft_len as usize;
    let mut twiddles = Vec::with_capacity(n);

    for k in 0..n {
        let theta = -2.0 * PI * (k as f64) / (n as f64);
        twiddles.push(Complex::new(theta.cos() as f32, theta.sin() as f32));
    }

    let twiddle_buffer = device_arc.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddle Buffer"),
        contents: bytemuck::cast_slice(&twiddles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Create bind group
    let bind_group = device_arc.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FFT Bind Group"),
        layout: &bgl,
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

    // Execute timing test
    let timer = std::time::Instant::now();

    for _ in 0..10 {
        custom_transfer.upload_data(&data, &buffer_a).await;
        
        let mut compute_encoder = device_arc.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Radix-5 Compute Encoder"),
        });

        // Compute workgroup dimensions - each work item processes 5 points
        let threads_per_fft = fft_len / 5; // Radix-5 each thread processes 5 points
        let workgroup_len = 64;
        let x = (threads_per_fft as f32 / workgroup_len as f32).ceil() as u32;
        let y = (buffer_a.size() / 8 / fft_len as u64) as u32; // Number of batches

        let mut cpass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // Run all FFT stages
        cpass.set_push_constants(0, &fft_len.to_le_bytes());
        
        for i in 0..total_stages {
            cpass.set_push_constants(4, &i.to_le_bytes());
            // for _ in 0..100{
            //     cpass.dispatch_workgroups(x, y, 1);
            //     }
            cpass.dispatch_workgroups(x, y, 1);
        }

        drop(cpass); // End compute pass
        queue_arc.submit(Some(compute_encoder.finish()));

        let _result: Vec<Complex> = custom_transfer
            .download_data(&buffer_b, Some(&mut ans))
            .await;
    }
    
    println!("Radix-5 FFT execution time: {:?}", timer.elapsed());
    println!("First few results: {:?}", &ans[..126]);
}