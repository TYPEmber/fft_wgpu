@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)  // 新增Twiddle因子缓冲区
var<storage, read> twiddles: array<vec2<f32>>;

const PI: f32 = 3.14159265358979323846;
const workgroup_len: u32 = 64u;

struct PushConstants { fft_len: u32, stage: u32 }
var<push_constant> consts: PushConstants;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    let fft_len = consts.fft_len;
    let index = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x) * workgroup_len + local_invocation_index;
    let offset = index / (fft_len / 2u) * fft_len;

    if index >= arrayLength(&buffer_a) / 2u {
        return;
    }
   fft(index % (fft_len / 2u), fft_len, offset, consts.stage);
}

fn fft(idx: u32, n: u32, offset: u32, stage: u32) {
    let J = 1u << stage;
    // 每个工作项处理一个蝶形运算
    let block_size = 2u * J;
   // let shared_value = subgroupBroadcast(local_value, 0); 
    //let total_blocks = n / block_size;
    let total_stages= u32(log2(f32(n)));
    let block_idx = idx / J;
    let j = idx % J;

    let s = block_idx;
    let twiddle = twiddles[block_idx*J];  
    //let twiddle=vec2<f32>(1.0,0.0);
    //let theta = - 2.0 * PI * f32(s * J) / f32(n);
    //let twiddle = vec2<f32>(cos(theta), sin(theta));

    // 输入位置
    let idx1 = block_idx * J + j + offset;
    let idx2 = idx1 + n / 2u;

    // 输出位置
    let out_idx1 = block_idx * block_size + j + offset;
    let out_idx2 = out_idx1 + J;

    // 根据阶段奇偶性确定读写缓冲区
    if stage % 2u == 0u {
        let a = buffer_a[idx1];
        let b = buffer_a[idx2];
        buffer_b[out_idx1] = a + b;
        buffer_b[out_idx2] = complex_mul(a - b, twiddle);
        
    }
    else {
        let a = buffer_b[idx1];
        let b = buffer_b[idx2];
        buffer_a[out_idx1] = a + b;
        buffer_a[out_idx2] = complex_mul(a - b, twiddle);
    }
}
//fn fft2(idx: u32, n: u32, offset: u32, stage: u32) {
    //let tid= idx % n;
   // let bits = u32(log2(f32(n)));
   // let target_idx = bit_reverse(tid, bits);

//}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn bit_reverse(n: u32, bits: u32) -> u32 {
    var reversed = 0u;
    var n1 = n;
    for (var i = 0u; i < bits; i++) {
        reversed = (reversed << 1u) | (n1 & 1u);  //将待处理的数字左移一位，最后一位由n1的最后一位决定
        n1 >>= 1u;   //n1右移一位
    }
    return reversed;
}