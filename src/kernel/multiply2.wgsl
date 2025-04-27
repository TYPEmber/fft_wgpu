@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
const workgroup_len: u32 = 64u;

var<push_constant> fft_len: u32;
@compute @workgroup_size(workgroup_len)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) { return; }
    
    let a = input[idx+fft_len];
    let b = input[idx % fft_len];
    output[idx] = vec2<f32>(
        a.x * b.x - a.y * b.y, // 实部
        a.x * b.y + a.y * b.x  // 虚部
    );

}