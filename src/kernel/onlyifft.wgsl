@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;

const PI: f32 = 3.14159265358979323846;
const workgroup_len: u32 = 32u;

struct PushConstants { fft_len: u32, stage: u32,round_num:u32}
var<push_constant> consts: PushConstants;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    let fft_len = consts.fft_len;

    let index = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x) * workgroup_len + local_invocation_index;
     if index >= arrayLength(&buffer_a) / 2u {
        return;
    }
    let offset = index / (fft_len / 2u) * fft_len;
   ifft(index % (fft_len / 2u), fft_len, offset, consts.stage,consts.round_num);

}

fn ifft(idx: u32, n: u32, offset: u32, stage: u32,round_num:u32) {
    let J = 1u << stage;
    // 每个工作项处理一个蝶形运算
    let block_size = 2u * J;
    let total_blocks = n / block_size;

    //if idx >= total_blocks * J {
        // buffer_a[idx] = vec2<f32>(f32(offset), f32(stage));
        // buffer_b[idx] = vec2<f32>(f32(offset), f32(block_size));
        //return;
   // }

    let block_idx = idx / J;
    let j = idx % J;

    let s = block_idx;
    let theta =  2.0 * PI * f32(s * J) / f32(n);
    let twiddle = vec2<f32>(cos(theta), sin(theta));

    // 输入位置
    let idx1 = s * J + j + offset;
    let idx2 = idx1 + n / 2u;

    // 输出位置
    let out_idx1 = s * block_size + j + offset;
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

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}




















































