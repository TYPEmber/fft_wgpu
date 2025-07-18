
@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

const PI_D: f64 = 3.141592653589793238462643383279502884;
const workgroup_len: u32 = 64u;

struct PushConstants { fft_len: u32, stage: u32 }
var<push_constant> consts: PushConstants;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, 
        @builtin(num_workgroups) num_workgroups: vec3<u32>, 
        @builtin(local_invocation_index) local_invocation_index: u32) {
    
    let fft_len = consts.fft_len;
    let stage = consts.stage;
    
    // 计算全局索引
    let group_size = num_workgroups.x * num_workgroups.y;
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * workgroup_len + local_invocation_index;
    if global_idx >= arrayLength(&buffer_a) / 4u {
        return;
    }
    
    // 确定此线程处理哪个批次的哪个蝶形
    let threads_per_fft = fft_len / 4u;
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;
    
    if (stage == 0u) {
        if (local_idx < threads_per_fft) {
            radix4_bit_reversal_and_butterfly_f64(local_idx, fft_len, batch_offset);
        }
    } else {
        if (local_idx < threads_per_fft) {
            radix4_butterfly_f64(local_idx, fft_len, batch_offset, stage);
        }
    }
}

// 四进制位反转函数 - 整数版本保持不变
fn quaternary_bit_reverse(n: u32, bits: u32) -> u32 {
    let quat_digits = bits / 2u;
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < quat_digits; i++) {
        reversed = (reversed << 2u) | (n_copy & 3u);
        n_copy >>= 2u;
    }
    
    if (bits % 2u == 1u) {
        reversed = (reversed << 1u) | (n_copy & 1u);
    }
    
    return reversed;
}

// 高精度版本的基4蝶形+位反转
fn radix4_bit_reversal_and_butterfly_f64(idx: u32, n: u32, offset: u32) {
    // 使用f64计算log2(n)
    let bits = u32(log2(f64(n)) + 0.4);
    
    let step = 4u;
    let block_idx = idx;
    let k = 0u;
    
    // 计算四个输入点的位反转索引
    let a_idx_br = quaternary_bit_reverse(block_idx * 4u + 0u, bits) + offset;
    let b_idx_br = quaternary_bit_reverse(block_idx * 4u + 1u, bits) + offset;
    let c_idx_br = quaternary_bit_reverse(block_idx * 4u + 2u, bits) + offset;
    let d_idx_br = quaternary_bit_reverse(block_idx * 4u + 3u, bits) + offset;
    
    // 读取数据并转换为f64
    let a = vec2<f64>(buffer_a[a_idx_br]);
    let b = vec2<f64>(buffer_a[b_idx_br]);
    let c = vec2<f64>(buffer_a[c_idx_br]);
    let d = vec2<f64>(buffer_a[d_idx_br]);
    
    // 使用f64进行蝶形计算
    let ac = a + c;
    let bd = b + d;
    let a_c = a - c;
    let b_d = b - d;
    
    // 计算输出位置
    let out_idx_a = block_idx * 4u + 0u + offset;
    let out_idx_b = block_idx * 4u + 1u + offset;
    let out_idx_c = block_idx * 4u + 2u + offset;
    let out_idx_d = block_idx * 4u + 3u + offset;
    
    // 基-4 FFT的最终计算
    let x0 = ac + bd;
    
    // 乘以-j的高精度版本
    let b_d_rot = vec2<f64>(b_d.y, -b_d.x);
    let x1 = a_c + b_d_rot;
    
    // 乘以-1的高精度版本
    let x2 = ac - bd;
    
    // 乘以j的高精度版本
    let b_d_rot3 = vec2<f64>(-b_d.y, b_d.x);
    let x3 = a_c + b_d_rot3;
    
    // 转换回f32并写入
    buffer_b[out_idx_a] = vec2<f32>(x0);
    buffer_b[out_idx_b] = vec2<f32>(x1);
    buffer_b[out_idx_c] = vec2<f32>(x2);
    buffer_b[out_idx_d] = vec2<f32>(x3);
}

// 高精度版本的基4蝶形运算
fn radix4_butterfly_f64(idx: u32, n: u32, offset: u32, stage: u32) {
    // 计算当前阶段蝶形运算参数
    let m = 1u << (stage * 2u);
    let step = 4u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // 计算四个点的索引
    let a_idx = block_idx * step + k + offset;
    let b_idx = a_idx + m;
    let c_idx = a_idx + 2u * m;
    let d_idx = a_idx + 3u * m;
    
    // 读取数据并转换为f64
    let a = vec2<f64>(buffer_b[a_idx]);
    let b = vec2<f64>(buffer_b[b_idx]);
    let c = vec2<f64>(buffer_b[c_idx]);
    let d = vec2<f64>(buffer_b[d_idx]);
    
    // 计算旋转因子索引
    let twiddle_base = k * (n / (4u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    let w3_idx = 3u * twiddle_base;
    
    // 获取旋转因子并转换为f64
    let w1 = vec2<f64>(twiddles[w1_idx]);
    let w2 = vec2<f64>(twiddles[w2_idx]);
    let w3 = vec2<f64>(twiddles[w3_idx]);
    
    // 使用f64进行复数乘法
    let b_rot = complex_mul_f64(b, w1);
    let c_rot = complex_mul_f64(c, w2);
    let d_rot = complex_mul_f64(d, w3);
    
    // 使用f64进行蝶形计算
    let ac = a + c_rot;
    let bd = b_rot + d_rot;
    let a_c = a - c_rot;
    let b_d = b_rot - d_rot;
    
    // 高精度计算
    let x0 = ac + bd;
    
    // 乘以-j的高精度版本
    let j_b_d = vec2<f64>(b_d.y, -b_d.x);
    let x1 = a_c + j_b_d;
    
    let x2 = ac - bd;
    
    // 乘以j的高精度版本
    let neg_j_b_d = vec2<f64>(-b_d.y, b_d.x);
    let x3 = a_c + neg_j_b_d;
    
    // 转换回f32并写回
    buffer_b[a_idx] = vec2<f32>(x0);
    buffer_b[b_idx] = vec2<f32>(x1);
    buffer_b[c_idx] = vec2<f32>(x2);
    buffer_b[d_idx] = vec2<f32>(x3);
}

// f64版本的复数乘法
fn complex_mul_f64(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// f64版本的优化复数乘法