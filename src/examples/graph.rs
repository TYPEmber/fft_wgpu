//use crate::compute_graph;
use fft_wgpu::GraphNode;
use num_complex::Complex32 as Complex;
use std::sync::Arc;
#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::default();
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
                required_features: adapter.features(), // Or specific features
                required_limits: adapter.limits(),     // Or wgpu::Limits::default() or custom
                label: Some("GPU Device"),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    // Wrap device and queue in Arc for the graph
    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);

    let data_len = 512 * 500 * 5; // Number of Complex numbers
    let buffer_size_bytes = (data_len * std::mem::size_of::<Complex>()) as u64;
    let cpu_data = vec![Complex::new(3.0, 0.0); data_len];
    let cpu_kernel = vec![Complex::new(5.0, 0.0); data_len];
    let fft_size = 512; // Example FFT size

    // Create initial GPU buffers
    let data_src_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Data Source Buffer"),
        size: (data_len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
  

    let knl_src_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Kernel Source Buffer"),
        size: (data_len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    

    // Create staging buffer for reading results
    let staging_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (data_len * std::mem::size_of::<Complex>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Build the compute graph
    // The lifetime 'a for nodes will be tied to the lifetime of data_src_buffer, knl_src_buffer, etc.
    // These buffers live for the 'main' scope.

    // Create intermediate buffers first so they live long enough
    // let data_fft_output_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("Data FFT Output Buffer"),
    //     size: (data_len * std::mem::size_of::<Complex<f32>>()) as u64,
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    //     mapped_at_creation: false,
    // });

    // let kernel_fft_output_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("Kernel FFT Output Buffer"),
    //     size: (data_len * std::mem::size_of::<Complex<f32>>()) as u64,
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    //     mapped_at_creation: false,
    // });

    // let multiply_output_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("Multiply Output Buffer"),
    //     size: (data_len * std::mem::size_of::<Complex<f32>>()) as u64,
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    //     mapped_at_creation: false,
    // });

    //let mut graph = fft_wgpu::ComputeGraph::new(device_arc.clone(), queue_arc.clone());

    // Use existing new method and copy outputs after execution
    let data_fft_node =
        fft_wgpu::FftNode::new(&device_arc, &queue_arc, &data_src_buffer, fft_size as u32);
    let data_fft_output = data_fft_node.get_output_buffer().clone();

    let knl_fft_node =
        fft_wgpu::FftNode::new(&device_arc, &queue_arc, &knl_src_buffer, fft_size as u32);
    let knl_fft_output = knl_fft_node.get_output_buffer().clone();
    // Node 2: Multiply
    let multiply_node =
        fft_wgpu::MultiplyNode::new(&device_arc, &queue_arc, &data_fft_output, &knl_fft_output);
    let multiply_output = multiply_node.get_output_buffer().clone();
    // // Node 3: IFFT
    let ifft_node =
        fft_wgpu::IfftNode::new(&device_arc, &queue_arc, &multiply_output, fft_size as u32);

    let mut graph_nodes_vec: Vec<Box<dyn fft_wgpu::GraphNode<'_> + '_>> = Vec::new();
    graph_nodes_vec.push(Box::new(data_fft_node));
    graph_nodes_vec.push(Box::new(knl_fft_node));
    graph_nodes_vec.push(Box::new(multiply_node));
    graph_nodes_vec.push(Box::new(ifft_node));
    // Create the graph, moving the fully constructed Vec of nodes into it.
    let graph = fft_wgpu::ComputeGraph::new(device_arc.clone(), queue_arc.clone(), graph_nodes_vec);
    queue_arc.write_buffer(&data_src_buffer, 0, bytemuck::cast_slice(&cpu_data));
    queue_arc.write_buffer(&knl_src_buffer, 0, bytemuck::cast_slice(&cpu_kernel));
    if let Some(final_result_buffer) = graph.execute_and_get_final_output() {
        let mut encoder = device_arc.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy to Staging Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &final_result_buffer,
            0,
            &staging_buffer,
            0,
            buffer_size_bytes,
        );
        queue_arc.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device_arc.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data_mapped = buffer_slice.get_mapped_range();
            let mut ans_from_gpu = vec![Complex::new(0.0, 0.0); data_len];
            bytemuck::cast_slice(&data_mapped).clone_into(&mut ans_from_gpu);

            println!("Result (first 10): {:?}", &ans_from_gpu[..10.min(data_len)]);
            if data_len > 512 {
                println!(
                    "Result (512 to 520): {:?}",
                    &ans_from_gpu[512..(512 + 8).min(data_len)]
                );
            }
            drop(data_mapped);
        } else {
            eprintln!("Failed to map staging buffer");
        }
        staging_buffer.unmap();
    } else {
        eprintln!("Graph execution did not yield an output buffer.");
    }
}
