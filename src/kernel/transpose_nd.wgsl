@group(0) @binding(0)
var<storage, read> input: array<vec2<f32>>; // 原始复数数据

@group(0) @binding(1)
var<storage, read_write> output: array<vec2<f32>>; // 转置后的复数数据

@group(0) @binding(2)
var<storage, read> input_shape: array<u32>; // 原始形状

@group(0) @binding(3)
var<storage, read> strides_t_r: array<u32>; // 转置步长

const workgroup_len: u32 = 64u;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, 
        @builtin(num_workgroups) num_workgroups: vec3<u32>, 
        @builtin(local_invocation_index) local_invocation_index: u32) {

    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let idx = group_idx * workgroup_len + local_invocation_index;
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    // 确认维度数
   // var rank = 0u;
    //for (var i = 0u; i < arrayLength(&input_shape); i = i + 1u) {
       // if (input_shape[i] > 0u) {
            //rank = rank + 1u;
       // } else {
           // break;
        //}
   // }
    let  rank=arrayLength(&input_shape); // 直接使用形状数组的长度作为rank
    // 计算转置后的索引
    var output_idx = 0u;
    var temp = idx;
    for(var j=0u;j<1000u;j=j+1u){
    // 从最低维到最高维处理
    for (var i = 0u; i < rank; i = i+1u) {
        let k = rank - 1u - i;   // 当前维度索引（从最低维开始）
        let dim_size = input_shape[k];
        let coord = temp % dim_size;
        temp = temp / dim_size;
        output_idx += coord * strides_t_r[k];
    }
    
    // 写入输出数组
    output[output_idx] = input[idx];
    }
}