@group(0) @binding(0) var<storage, read_write> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
const workgroup_len: u32 = 64u;

var<push_constant> fft_len: u32;
@compute @workgroup_size(workgroup_len)
fn main(@builtin(@builtin(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
     let idx = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x) * workgroup_len + local_invocation_index;
   
    if (idx >= (arrayLength(&input)-fft_len)) { return; }
    
    let a = input[idx+fft_len];
    let b = input[idx % fft_len];
    output[idx] = vec2<f32>(
        a.x * b.x - a.y * b.y, // 实部
        a.x * b.y + a.y * b.x  // 虚部
    );

}