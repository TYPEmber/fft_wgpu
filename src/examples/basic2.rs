use ash::vk;
use num_complex::Complex32 as Complex;
use wgpu::hal::vulkan::Api;
use wgpu::{Buffer, Device, Queue};
use std::thread;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() {
    // Instantiates instance of WebGPU

    // unsafe {
    //     std::env::set_var("WGPU_VULKAN_ASYNC_COMPUTE", "1");
    // }
    //let instance = wgpu::Instance::default();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        // There is no ASYNC_COMPUTE flag in InstanceFlags
        flags: wgpu::InstanceFlags::empty(),
        ..Default::default()
    });
    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    dbg!(adapter.limits());
    dbg!(adapter.features());
    // dbg!(wgpu::Limits::default());
    let info = adapter.get_info();
    dbg!(info);
    // std::process::exit(0);

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device_0, queue_0) = adapter
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

    // Instantiates instance of WebGPU
    unsafe {
        std::env::set_var("WGPU_VULKAN_ASYNC_COMPUTE", "1");
    }
    let instance1 = wgpu::Instance::default();
    // `request_adapter` instantiates the general connection to the GPU
    let adapter1 = instance1
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    dbg!(adapter1.limits());
    //dbg!(adapter.features());
    //dbg!(wgpu::Limits::default());
    let info = adapter1.get_info();
    dbg!(info);

    // std::process::exit(0);

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device_1, queue_1) = adapter1
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter1.features(),
                required_limits: adapter1.limits(),
                label: Some("GPU Device"),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    let data = vec![Complex::new(5.0, 0.0); 512 * 500 * 5 ];
    let len = data.len() / 2;

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
    let staging_buffer_0 = device_0.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src_0 = device_0.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer_1 = device_1.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src_1 = device_1.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Instead of trying to store a reference, we can store information we need from the buffer
    // let mut vk_buffer_raw_handle: Option<u64> = None;
    //    let mut vk_buffer: Option<&wgpu::hal::vulkan::Buffer> = None;
    //     unsafe {
    //         staging_buffer_0.as_hal::<Api, _, _>(|hal_buffer| {
    //             if let Some(vulkan_buffer) = hal_buffer {
    //                 // vulkan_buffer 是 &wgpu_hal::vulkan::Buffer
    //                 // If you need the raw Vulkan handle, add the ash crate as a dependency
    //               //  Some(vulkan_buffer)

    //                 dbg!(vulkan_buffer);

    //                 //let vk_buffer=vulkan_buffer;

    //                 // Store a value extracted from the buffer rather than a reference to the buffer itself
    //                 // Note: For illustration only - replace with the actual field you need from vulkan_buffer
    //              //   vk_buffer_raw_handle = Some(std::ptr::addr_of!(*vulkan_buffer) as u64);
    //               //  dbg!(vk_buffer_raw_handle);
    //                 // let raw_handle = vulkan_buffer.raw;
    //                 // 现在 raw_handle 就是 Vulkan 的 VkBuffer，可以用于 Vulkan FFI 互操作
    //             }
    //             else{
    //                 dbg!("Failed to get Vulkan buffer from staging_buffer_0");

    //                // None
    //             }
    //         });
    //     }
    // unsafe {
    //     staging_buffer_0.as_hal::<Api, _, _>(|hal_buffer| {
    //         if let Some(vulkan_buffer) = hal_buffer {
    //             // vulkan_buffer 是 &wgpu_hal::vulkan::Buffer
    //             // If you need the raw Vulkan handle, add the ash crate as a dependency
    //             dbg!(vulkan_buffer);
    //             vk_buffer = Some(vulkan_buffer);
    //             // let raw_handle = vulkan_buffer.raw;
    //             // 现在 raw_handle 就是 Vulkan 的 VkBuffer，可以用于 Vulkan FFI 互操作
    //         }
    //     });
    // }
    let (data_0, data_1) = data.split_at(len);

    // let data_0 = &data.clone();
    // let data_1 = &data.clone();

    let ans_0 = &mut ans.clone();
    let ans_1 = &mut ans.clone();

    let timer = std::time::Instant::now();

    std::thread::scope(|s| {
        let h0 = s.spawn(|| {
            compute_test(device_0, queue_0, src_0, staging_buffer_0, data_0, ans_0);
        });
        thread::sleep(Duration::from_millis(200));
        let h1 = s.spawn(|| {
            compute_test(device_1, queue_1, src_1, staging_buffer_1, data_1, ans_1);
        });

        h0.join();
        
        h1.join();
    });

    // println!("{:?}", &ans[data.len() - 512..]);

    dbg!(timer.elapsed());
}

fn compute_test(
    device: Device,
    queue: Queue,
    src: Buffer,
    staging_buffer: Buffer,
    data: &[Complex],
    ans: &mut [Complex],
) {
    let len = data.len();
    let fft_forward = fft_wgpu::Forward::new(&device, &queue, &src, 512);
    let buffer_slice = staging_buffer.slice(..);

    for _ in 0..1000 {
        queue.write_buffer(&src, 0, bytemuck::cast_slice(data));
        // let mut view = queue
        //     .write_buffer_with(
        //         &src,
        //         0,
        //         std::num::NonZero::new((len * std::mem::size_of::<Complex>()) as u64).unwrap(),
        //     )
        //     .unwrap();
        // view.copy_from_slice(bytemuck::cast_slice(data));

        // drop(view);

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // let _output = fft_forward.proc(&mut encoder);
        let output = fft_forward.proc(&mut encoder);
        // let output = fft_forward.proc(&mut encoder);
        //let output = fft_forward_2.proc(&mut encoder);

        encoder.copy_buffer_to_buffer(
            output,
            0,
            &staging_buffer,
            0,
            (len * std::mem::size_of::<Complex>()) as u64,
        );

        queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.

        buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});

        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // while !device.poll(wgpu::MaintainBase::Poll).is_queue_empty() {}

        // Gets contents of buffer
        let ans_view = buffer_slice.get_mapped_range();

        // // Since contents are got in bytes, this converts these bytes back to u32
        ans.copy_from_slice(bytemuck::cast_slice(&ans_view));

        // println!("{:?}", &ans[..16]);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(ans_view);
        staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory
    }
}

#[tokio::main]
async fn view() {
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

    dbg!(adapter.limits());
    dbg!(adapter.features());
    dbg!(wgpu::Limits::default());

    // std::process::exit(0);

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device_0, queue_0) = adapter
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

    dbg!(adapter.limits());
    dbg!(adapter.features());
    dbg!(wgpu::Limits::default());

    // std::process::exit(0);

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device_1, queue_1) = adapter
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
    let len = data.len() / 2;

    // let mut data_cpu = data
    //     .iter()
    //     .map(|c| rustfft::num_complex::Complex::new(c.re, 0.0))
    //     .collect::<Vec<_>>();
    // let fft = rustfft::FftPlanner::new().plan_fft_forward(16);
    // fft.process(&mut data_cpu);
    // fft.process(&mut data_cpu);
    // println!("{:?}", &data_cpu[..]);

    let mut ans = vec![Complex::ZERO; data.len()];

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer_0 = device_0.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src_0 = device_0.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer_1 = device_1.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src_1 = device_1.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let fft_forward_0 = fft_wgpu::Forward::new(&device_0, &queue_0, &src_0, 512);
    let fft_forward_1 = fft_wgpu::Forward::new(&device_1, &queue_1, &src_1, 512);
    // let fft_forward_2 = fft_wgpu::Forward::new(&device, &queue, &src, 16);
    let buffer_slice_0 = staging_buffer_0.slice(..);
    let buffer_slice_1 = staging_buffer_1.slice(..);

    let timer = std::time::Instant::now();

    let (ans_0, ans_1) = ans.split_at_mut(len);

    for _ in 0..10000 {
        queue_0.write_buffer(&src_0, 0, bytemuck::cast_slice(&data[..len]));
        queue_1.write_buffer(&src_1, 0, bytemuck::cast_slice(&data[len..]));
        // let mut view = queue
        //     .write_buffer_with(
        //         &src,
        //         0,
        //         std::num::NonZero::new((len * std::mem::size_of::<Complex>()) as u64).unwrap(),
        //     )
        //     .unwrap();
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));
        // view.copy_from_slice(bytemuck::cast_slice(data.as_slice()));

        // drop(view);

        // queue.submit(None);
        // device.poll(wgpu::Maintain::Poll).panic_on_timeout();
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder_0 =
            device_0.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut encoder_1 =
            device_1.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let output_0 = fft_forward_0.proc(&mut encoder_0);
        let output_1 = fft_forward_1.proc(&mut encoder_1);
        //let output=fft_forward.buffer_a;
        // let output = fft_forward.proc(&mut encoder);
        //let output = fft_forward_2.proc(&mut encoder);

        encoder_0.copy_buffer_to_buffer(
            output_0,
            0,
            &staging_buffer_0,
            0,
            (len * std::mem::size_of::<Complex>()) as u64,
        );
        encoder_1.copy_buffer_to_buffer(
            output_1,
            0,
            &staging_buffer_1,
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

        queue_0.submit(Some(encoder_0.finish()));
        queue_1.submit(Some(encoder_1.finish()));
        // queue.submit(None);
        // let rn = fft_forward.round_num.slice(..);

        // rn.map_async(wgpu::MapMode::Read, move |_| {});

        //device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        // let a: Vec<u8> = rn.get_mapped_range().iter().copied().collect();
        // dbg!(a);
        // fft_forward.round_num.unmap();

        // Note that we're not calling `.await` here.

        buffer_slice_0.map_async(wgpu::MapMode::Read, move |_| {});
        device_0.poll(wgpu::Maintain::Poll).panic_on_timeout();

        buffer_slice_1.map_async(wgpu::MapMode::Read, move |_| {});
        device_1.poll(wgpu::Maintain::Poll).panic_on_timeout();

        device_0.poll(wgpu::Maintain::Wait).panic_on_timeout();
        device_1.poll(wgpu::Maintain::Wait).panic_on_timeout();

        let data_0 = buffer_slice_0.get_mapped_range();
        let data_1 = buffer_slice_1.get_mapped_range();

        // Gets contents of buffer

        // // Since contents are got in bytes, this converts these bytes back to u32
        // bytemuck::cast_slice(&data_0).clone_into(&mut ans);
        //  bytemuck::cast_slice(&data_1).clone_into(&mut ans[len..]);
        ans[..len].copy_from_slice(bytemuck::cast_slice(&data_0));
        // ans[len..].copy_from_slice(bytemuck::cast_slice(&data_1));

        //  println!("{:?}", &ans[..10]);
        // println!("{:?}", &ans[data.len() - 512..]);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data_0);
        staging_buffer_0.unmap(); // Unmaps buffer from memory
        drop(data_1);
        staging_buffer_1.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory
    }

    dbg!(timer.elapsed());
}

#[tokio::main]
async fn raw() {
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
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    let data = vec![Complex::new(1.0, 0.0); 512 * 500 * 5];
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
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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

    let fft_forward = fft_wgpu::Forward::new(&device, &queue, &src, 512);
    // let fft_forward_2 = fft_wgpu::Forward::new(&device, &queue, &src, 16);
    let buffer_slice = staging_buffer.slice(..);

    let timer = std::time::Instant::now();

    for _ in 0..1000 {
        queue.write_buffer(&src, 0, bytemuck::cast_slice(data.as_slice()));
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let output = fft_forward.proc(&mut encoder);
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
        let data1 = buffer_slice.get_mapped_range();

        // Gets contents of buffer

        // // Since contents are got in bytes, this converts these bytes back to u32
        //  bytemuck::cast_slice(&data1).clone_into(&mut ans);
        ans.copy_from_slice(bytemuck::cast_slice(&data1));

        // println!("{:?}", &ans[..10]);
        //println!("{:?}", &ans[512..520]);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data1);
        staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory
    }

    dbg!(timer.elapsed());
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32 as Complex;
    #[tokio::test]
    // 在main函数末尾添加以下测试代码
    async fn test_fft() {
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

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
            .unwrap();

        let data = vec![Complex::new(1.0, 0.0); 16];
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
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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

        let fft_forward = fft_wgpu::Forward::new(&device, &queue, &src, 16);
        // let fft_forward_2 = fft_wgpu::Forward::new(&device, &queue, &src, 16);

        let timer = std::time::Instant::now();

        for _ in 0..10 {
            queue.write_buffer(&src, 0, bytemuck::cast_slice(data.as_slice()));
            // A command encoder executes one or many pipelines.
            // It is to WebGPU what a command buffer is to Vulkan.
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // let _output = fft_forward.proc(&mut encoder);
            let output = fft_forward.proc(&mut encoder);
            // let output = fft_forward.proc(&mut encoder);
            //let output = fft_forward_2.proc(&mut encoder);

            encoder.copy_buffer_to_buffer(
                output,
                0,
                &staging_buffer,
                0,
                (len * std::mem::size_of::<Complex>()) as u64,
            );

            queue.submit(Some(encoder.finish()));

            // let rn = fft_forward.round_num.slice(..);

            // rn.map_async(wgpu::MapMode::Read, move |_| {});

            // device.poll(wgpu::Maintain::wait()).panic_on_timeout();
            // let a: Vec<u8> = rn.get_mapped_range().iter().copied().collect();
            // dbg!(a);
            // fft_forward.round_num.unmap();

            // Note that we're not calling `.await` here.
            let buffer_slice = staging_buffer.slice(..);

            buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});

            device.poll(wgpu::Maintain::wait()).panic_on_timeout();

            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();

            // // Since contents are got in bytes, this converts these bytes back to u32
            bytemuck::cast_slice(&data).clone_into(&mut ans);

            println!("{:?}", &ans[..16]);

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
}
