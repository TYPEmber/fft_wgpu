use fft_wgpu::*;
mod compute_graph;
#[tokio::main]
async fn main() {
    crate::basic().await;
}
