@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

const PI: f32 = 3.14159265358979323846;
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
        // IFFT阶段0：位反转和第一阶段蝶形
        if (local_idx < threads_per_fft) {
            ifft_radix4_bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
        }
    } else {
        // IFFT标准基-4蝶形
        if (local_idx < threads_per_fft) {
            ifft_radix4_butterfly(local_idx, fft_len, batch_offset, stage);
        }
    }
}

// IFFT四进制位反转函数
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

// IFFT组合位反转和第一阶段蝶形
fn ifft_radix4_bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    let bits = u32(log2(f32(n))+0.4);
    let step = 4u;
    let block_idx = idx;
    let k = 0u;
    
    // 计算四个输入点的位反转索引
    let a_idx_br = quaternary_bit_reverse(block_idx * 4u + 0u, bits) + offset;
    let b_idx_br = quaternary_bit_reverse(block_idx * 4u + 1u, bits) + offset;
    let c_idx_br = quaternary_bit_reverse(block_idx * 4u + 2u, bits) + offset;
    let d_idx_br = quaternary_bit_reverse(block_idx * 4u + 3u, bits) + offset;
    
    // 从buffer_a读取位反转后的值
    let a = buffer_a[a_idx_br];
    let b = buffer_a[b_idx_br];
    let c = buffer_a[c_idx_br];
    let d = buffer_a[d_idx_br];
    
    // IFFT第一阶段蝶形运算（旋转因子取共轭）
    let ac = a + c;
    let bd = b + d;
    let a_c = a - c;
    let b_d = b - d;
    
    // IFFT旋转因子调整（取共轭）
    // 原FFT：W^0=1, W^1=-j, W^2=-1, W^3=j
    // IFFT：W^0=1, W^1=j, W^2=-1, W^3=-j
    
    // 输出位置
    let out_idx_a = block_idx * 4u + 0u + offset;
    let out_idx_b = block_idx * 4u + 1u + offset;
    let out_idx_c = block_idx * 4u + 2u + offset;
    let out_idx_d = block_idx * 4u + 3u + offset;
    
    // IFFT基-4蝶形计算
    buffer_b[out_idx_a] = ac + bd;  // W^0 = 1
    
    // IFFT: W^1 = j (原FFT中是-j)
    let b_d_rot = vec2<f32>(-b_d.y, b_d.x); // 乘以j: (x,y) -> (-y,x)
    buffer_b[out_idx_b] = a_c + b_d_rot;
    
    buffer_b[out_idx_c] = ac - bd;  // W^2 = -1
    
    // IFFT: W^3 = -j (原FFT中是j)
    let b_d_rot3 = vec2<f32>(b_d.y, -b_d.x); // 乘以-j: (x,y) -> (y,-x)
    buffer_b[out_idx_d] = a_c + b_d_rot3;
}

// IFFT标准基-4蝶形运算
fn ifft_radix4_butterfly(idx: u32, n: u32, offset: u32, stage: u32) {
    let m = 1u << (stage * 2u);
    let step = 4u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // 计算四个点的索引
    let a_idx = block_idx * step + k + offset;
    let b_idx = a_idx + m;
    let c_idx = a_idx + 2u * m;
    let d_idx = a_idx + 3u * m;
    
    // 从buffer_b读取值
    let a = buffer_b[a_idx];
    let b = buffer_b[b_idx];
    let c = buffer_b[c_idx];
    let d = buffer_b[d_idx];
    
    // 计算旋转因子索引（与FFT相同）
    let twiddle_base = k * (n / (4u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    let w3_idx = 3u * twiddle_base;
    
    // IFFT关键修改：使用旋转因子的共轭
    // 注意：旋转因子缓冲区内容与FFT相同
    // IFFT需要旋转因子的共轭：complex_conjugate(twiddle)
    let w1 = complex_conjugate(twiddles[w1_idx]);
    let w2 = complex_conjugate(twiddles[w2_idx]);
    let w3 = complex_conjugate(twiddles[w3_idx]);
    
    let b_rot = complex_mul(b, w1);
    let c_rot = complex_mul(c, w2);
    let d_rot = complex_mul(d, w3);
    
    // 执行IFFT基-4蝶形运算
    let ac = a + c_rot;
    let bd = b_rot + d_rot;
    let a_c = a - c_rot;
    let b_d = b_rot - d_rot;
    
    // IFFT蝶形输出（注意旋转方向与FFT相反）
    buffer_b[a_idx] = ac + bd;
    
    // IFFT: 乘以j (原FFT中是-j)
    let j_b_d = vec2<f32>(-b_d.y, b_d.x); // (x,y) -> (-y,x)
    buffer_b[b_idx] = a_c + j_b_d;
    
    buffer_b[c_idx] = ac - bd;
    
    // IFFT: 乘以-j (原FFT中是j)
    let neg_j_b_d = vec2<f32>(b_d.y, -b_d.x); // (x,y) -> (y,-x)
    buffer_b[d_idx] = a_c + neg_j_b_d;
    
    // 如果是最后阶段，进行归一化（除以N）
    let total_stages = u32(log2(f32(n))+0.2) / 2u;
    if (stage == total_stages - 1u) {
        let norm_factor = 1.0 / f32(n);
        buffer_b[a_idx] = buffer_b[a_idx] * norm_factor;
        buffer_b[b_idx] = buffer_b[b_idx] * norm_factor;
        buffer_b[c_idx] = buffer_b[c_idx] * norm_factor;
        buffer_b[d_idx] = buffer_b[d_idx] * norm_factor;
    }
}

// 复数乘法
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// 复数共轭（IFFT关键函数）
fn complex_conjugate(c: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(c.x, -c.y);
}