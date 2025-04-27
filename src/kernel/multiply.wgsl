@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
const workgroup_len: u32 = 64u;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_a)) { return; }
    
    let a = input_a[idx];
    let b = input_b[idx];
    output[idx] = vec2<f32>(
        a.x * b.x - a.y * b.y, // 实部
        a.x * b.y + a.y * b.x  // 虚部
    );
}
