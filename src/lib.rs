use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::*;

pub mod processor;
pub use processor::*;
pub mod wgpu_helper;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Complex {
    pub real: f32,
    pub imag: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { real: re, imag: im }
    }
    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }
}

async fn prepare_gpu() -> Option<(Device, Queue)> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    dbg!(instance.enumerate_adapters(Backends::VULKAN));
    dbg!(instance.enumerate_adapters(Backends::GL));
    dbg!(instance.enumerate_adapters(Backends::METAL));

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;

    dbg!(adapter.limits());

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
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
        .ok()?;

    Some((device, queue))
}

fn prepare_cs_model(device: &Device) -> ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernel/fft_stage.wgsl"))),
    });
    // Instantiates the pipeline.
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    })
}

pub async fn basic() {
    let (device, queue) = prepare_gpu().await.unwrap();
    let pipeline = prepare_cs_model(&device);

    let data = vec![Complex::new(1.0, 0.0); 512 * 500 * 5];
    let fft_len = 512;
    let len = data.len();

    // let mut data_cpu = data
    //     .iter()
    //     .map(|c| rustfft::num_complex::Complex::new(c.real, 0.0))
    //     .collect::<Vec<_>>();
    // let fft = rustfft::FftPlanner::new().plan_fft_forward(fft_len);
    // fft.process(&mut data_cpu);
    // println!("{:?}", &data_cpu[..10]);

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let round_num = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let fft_len_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
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
                resource: round_num.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: fft_len_buf.as_entire_binding(),
            },
        ],
    });

    let mut ans = vec![Complex::zero(); len];

    let timer = std::time::Instant::now();

    for _ in 0..1000 {
        queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(data.as_slice()));
        queue.write_buffer(&round_num, 0, &0u32.to_le_bytes());
        queue.write_buffer(&fft_len_buf, 0, &(fft_len as u32).to_le_bytes());

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let x = (fft_len as u32 / 2 / 32).max(1);
            let y = (data.len() / fft_len) as u32;
            let z = 1;

            for _ in 0..(fft_len as f32).log2().round() as usize {
                cpass.dispatch_workgroups(x, y, z);
            }
        }

        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        if ((fft_len as f32).log2().round() as usize) % 2 == 0 {
            encoder.copy_buffer_to_buffer(
                &buffer_a,
                0,
                &staging_buffer,
                0,
                (len * std::mem::size_of::<Complex>()) as u64,
            );
        } else {
            encoder.copy_buffer_to_buffer(
                &buffer_b,
                0,
                &staging_buffer,
                0,
                (len * std::mem::size_of::<Complex>()) as u64,
            );
        }

        queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});

        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();

        // // Since contents are got in bytes, this converts these bytes back to u32
        bytemuck::cast_slice(&data).clone_into(&mut ans);

        // println!("{:?}", &ans[512 * 100 - 1.. 512 * 100 + 10]);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory
    }

    dbg!(timer.elapsed());
}
