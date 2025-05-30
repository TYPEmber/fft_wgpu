@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<f32>>;
const workgroup_len: u32 = 64u;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
     let index = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x) * workgroup_len + local_invocation_index;
    input_b[index] = input_a[index];
    
}
