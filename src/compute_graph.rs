use crate::processor;
use std::sync::Arc;
use wgpu::util::DeviceExt; // Ensure this is in scope if used by your fft_wgpu lib
//Trait for all compute graph nodes
pub trait GraphNode<'a> {
    fn run(&self, encoder: &mut wgpu::CommandEncoder);
    // The returned buffer reference is now tied to the lifetime of `&self` in `get_output_buffer`
    fn get_output_buffer(&self) -> &wgpu::Buffer;
}

// --- FFT Node ---
pub struct FftNode<'a> {
    // device: &'a wgpu::Device, // Only needed if proc re-creates bind groups AND device is not in forward_op
    // queue: &'a wgpu::Queue, // Only needed if proc uses it AND it's not in forward_op
    forward_op: processor::Forward<'a>,
    fft_len: u32, // Store fft_len to make the decision in get_output_buffer
}

impl<'a> FftNode<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        input_buffer: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let forward_op = processor::Forward::new(device, queue, input_buffer, fft_len);
        // No longer store output_buffer_ref here
        Self {
            forward_op,
            fft_len,
        }
    }
}

impl<'a> GraphNode<'a> for FftNode<'a> {
    fn run(&self, encoder: &mut wgpu::CommandEncoder) {
        // Assuming self.forward_op.proc is callable like this.
        // If proc needs &mut self.forward_op, then FftNode.run needs &mut self.
        let _ = self.forward_op.proc(encoder);
    }

    fn get_output_buffer(&self) -> &wgpu::Buffer {
        // Logic to determine output buffer is now here
        if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
            self.forward_op.buffer_a // Output is in the original input buffer
        } else {
            &self.forward_op.buffer_b // Output is in the internal buffer_b
        }
    }
}

// --- IFFT Node ---
pub struct IfftNode<'a> {
    // device: &'a wgpu::Device,
    inverse_op: processor::Inverse<'a>,
    fft_len: u32, // Store fft_len
}

impl<'a> IfftNode<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        input_buffer: &'a wgpu::Buffer,
        fft_len: u32,
    ) -> Self {
        let inverse_op = processor::Inverse::new(device, queue, input_buffer, fft_len);
        Self {
            inverse_op,
            fft_len,
        }
    }
}

impl<'a> GraphNode<'a> for IfftNode<'a> {
    fn run(&self, encoder: &mut wgpu::CommandEncoder) {
        let _ = self.inverse_op.proc(encoder);
    }

    fn get_output_buffer(&self) -> &wgpu::Buffer {
        if ((self.fft_len as f32).log2().round() as usize) % 2 == 0 {
            self.inverse_op.buffer_a
        } else {
            &self.inverse_op.buffer_b
        }
    }
}

// --- Multiply Node ---
// MultiplyNode was likely okay because `multiply_op.result` is an owned buffer within multiply_op,
// and `get_output_buffer` returns `&self.multiply_op.result`, correctly tying the lifetime.
pub struct MultiplyNode<'a> {
    // device: &'a wgpu::Device,
    multiply_op: processor::Multiply<'a>,
}

impl<'a> MultiplyNode<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        buffer_a: &'a wgpu::Buffer,
        buffer_b: &'a wgpu::Buffer,
    ) -> Self {
        let multiply_op = processor::Multiply::new(device, queue, buffer_a, buffer_b);
        Self { multiply_op }
    }
}

impl<'a> GraphNode<'a> for MultiplyNode<'a> {
    fn run(&self, encoder: &mut wgpu::CommandEncoder) {
        let _ = self.multiply_op.proc(encoder);
    }

    fn get_output_buffer(&self) -> &wgpu::Buffer {
        &self.multiply_op.result
    }
}

// --- Compute Graph ---
pub struct ComputeGraph<'a> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub nodes: Vec<Box<dyn GraphNode<'a> + 'a>>,
}

impl<'a> ComputeGraph<'a> {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,nodes: Vec<Box<dyn GraphNode<'a> + 'a>>) -> Self {
        Self {
            device,
            queue,
            nodes,
        }
    }

    pub fn add_node(&mut self, node: Box<dyn GraphNode<'a> + 'a>) {
        self.nodes.push(node);
    }

    pub fn execute_and_get_final_output(&self) -> Option<&wgpu::Buffer> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Graph Encoder"),
            });

        for node in &self.nodes {
            node.run(&mut encoder);
        }

        self.queue.submit(Some(encoder.finish()));

        self.nodes.last().map(|node| node.get_output_buffer())
       
    }
}




