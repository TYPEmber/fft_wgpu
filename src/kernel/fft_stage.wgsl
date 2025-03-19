@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> stage: u32;
@group(0) @binding(3)
var<storage, read_write> fft_len: u32;

const PI: f32 = 3.14159265358979323846;
const workgroup_len: u32 = 32u;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    let index = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.x * num_workgroups.y) * workgroup_len + local_invocation_index;
    let offset = index / (fft_len / 2u) * fft_len;
    fft(index % (fft_len / 2u), fft_len, offset);

    //  buffer_a[stage] = vec2<f32>(f32(p), f32(stage));

    // if index == arrayLength(&buffer_a) / 2u - 1u {
    //     stage += 1u;
    // }

    // if stage == 4 {
    //     stage = 0u;
    // }
}

fn fft(idx: u32, n: u32, offset: u32) {
    let J = 1u << stage;
    // 每个工作项处理一个蝶形运算
    // let idx = global_id.x;
    let block_size = 2u * J;
    let total_blocks = n / block_size;

    if idx >= total_blocks * J {
        // buffer_a[idx] = vec2<f32>(f32(offset), f32(stage));
        // buffer_b[idx] = vec2<f32>(f32(offset), f32(block_size));
        return;
    }

    let block_idx = idx / J;
    let j = idx % J;

    let s = block_idx;
    let theta = - 2.0 * PI * f32(s * J) / f32(n);
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