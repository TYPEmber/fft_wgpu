enable f16;

@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f16>>;
const workgroup_len: u32 = 64u;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
     let idx = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x) * workgroup_len + local_invocation_index;
    if (idx >= arrayLength(&input_a)) { return; }
    
    let a = input_a[idx];
    let b = input_b[idx];
    output[idx] = vec2<f16>(
        a.x * b.x - a.y * b.y, // 实部
        a.x * b.y + a.y * b.x  // 虚部
    );
}
