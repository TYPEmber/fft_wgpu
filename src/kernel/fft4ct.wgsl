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
    
    // 计算全局索引 - 使用3D工作组更高效地处理多批次FFT
    let group_size = num_workgroups.x * num_workgroups.y; // 将z留下做额外的批次处理
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * workgroup_len + local_invocation_index;
     if global_idx >= arrayLength(&buffer_a) / 4u {
        return;
    }
    // 确定此线程处理哪个批次的哪个蝶形
    // 基-4需要fft_len/4个线程（每个线程处理4个点）
    let threads_per_fft = fft_len / 4u;
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;
    
    if (stage == 0u) {
        // 阶段0：结合四进制位反转和第一阶段蝶形运算
        if (local_idx < threads_per_fft) {
            radix4_bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
        }
    } else {
        // 阶段1及以上：标准基-4蝶形运算
        // 注意：stage+1是因为阶段0已经包含了第一次蝶形运算
        if (local_idx < threads_per_fft) {
            radix4_butterfly(local_idx, fft_len, batch_offset, stage);
        }
    }
}

// 四进制位反转函数（每两位二进制表示一个四进制位）
fn quaternary_bit_reverse(n: u32, bits: u32) -> u32 {
    // 确保位数是2的倍数（四进制位反转要求）
    let quat_digits = bits / 2u;
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < quat_digits; i++) {
        reversed = (reversed << 2u) | (n_copy & 3u); // 取最低2位（一个四进制位）
        n_copy >>= 2u;
    }
    
    // 如果位数是奇数，处理剩余的1位，因为基4的fft是4的幂，未验证该逻辑
    if (bits % 2u == 1u) {
        reversed = (reversed << 1u) | (n_copy & 1u);
    }
    
    return reversed;
}

// 组合位反转和第一阶段基-4蝶形运算
fn radix4_bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    // 四进制位反转所需的位数（log4(N) = log2(N)/2，向上取整）
    let bits = u32(log2(f32(n))+0.4);
    
    // 每个工作项处理一个4点蝶形
   // let m = 1u;  // 第一阶段子问题大小
   // let step = 4u * m; // 子问题步长，类似于每个块的长度（或者说大小）
    let step=4u;
    //let block_idx = idx / m; // 子问题索引
    let block_idx=idx;
   // let k = idx % m;   // 子问题内位置（第一阶段始终为0）
    let k=0u;
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
    
    // 第一阶段旋转因子是简单的
    // W^0 = 1+0i, W^(N/4) = 0-1i, W^(N/2) = -1+0i, W^(3N/4) = 0+1i
    
    // 执行基-4蝶形运算
    // 第一阶段中间结果
    let ac = a + c;
    let bd = b + d;
    let a_c = a - c;
    let b_d = b - d;
    
    // 计算输出位置（顺序写入）
    let out_idx_a = block_idx * 4u + 0u + offset;
    let out_idx_b = block_idx * 4u + 1u + offset;
    let out_idx_c = block_idx * 4u + 2u + offset;
    let out_idx_d = block_idx * 4u + 3u + offset;
    
    // 基-4 FFT的最终计算和写入
    buffer_b[out_idx_a] = ac + bd;
    
    // 应用W^1旋转因子到b_d (乘以-j)
    let b_d_rot = vec2<f32>(b_d.y, -b_d.x); // 特殊情况：乘以-j等于(b.y, -b.x)
    buffer_b[out_idx_b] = a_c + b_d_rot;
    
    // 应用W^2旋转因子到bd (乘以-1)
    buffer_b[out_idx_c] = ac - bd;
    
    // 应用W^3旋转因子到b_d (乘以j) 
    let b_d_rot3 = vec2<f32>(-b_d.y, b_d.x); // 特殊情况：乘以j等于(-b.y, b.x)
    buffer_b[out_idx_d] = a_c + b_d_rot3; // 修正：应该是加号而非减号
}

// 标准基-4蝶形运算
fn radix4_butterfly(idx: u32, n: u32, offset: u32, stage: u32) {
    // 计算当前阶段蝶形运算参数
    let m = 1u << (stage  * 2u); // 子问题大小(4^(stage-1))
    let step = 4u * m;                 // 子问题步长，也就是块的大小
    let k = idx % m;                   // 子问题内位置
    let block_idx = idx / m;           // 子问题索引
    
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
    
    // 计算旋转因子索引 - 修正取模操作为整个N
    let twiddle_base = k * (n / (4u * m));  //n是每次fft的长度，
   // let w1_idx = twiddle_base % (n);              // 修正：对n取模而非n/2
   // let w2_idx = (2u * twiddle_base) % (n);       // 修正：对n取模而非n/2
   // let w3_idx = (3u * twiddle_base) % (n);       // 修正：对n取模而非n/2
    
    let w1_idx = twiddle_base ;              // 修正：对n取模而非n/2
    let w2_idx = 2u * twiddle_base;       // 修正：对n取模而非n/2
    let w3_idx = 3u * twiddle_base;       // 修正：对n取模而非n/2
    // 获取旋转因子并应用
    let w1 = twiddles[w1_idx];
    let w2 = twiddles[w2_idx];
    let w3 = twiddles[w3_idx];
    
   // let b_rot = optimized_complex_mul(b, w1);
   // let c_rot = optimized_complex_mul(c, w2);
   // let d_rot = optimized_complex_mul(d, w3);

    let b_rot = complex_mul(b, w1);
    let c_rot = complex_mul(c, w2);
    let d_rot = complex_mul(d, w3);
    
    // 执行基-4蝶形运算
    let ac = a + c_rot;
    let bd = b_rot + d_rot;
    let a_c = a - c_rot;
    let b_d = b_rot - d_rot;
    
    // 写回结果
    buffer_b[a_idx] = ac + bd;
    
    // 计算j*b_d (乘以-j)
    let j_b_d = vec2<f32>(b_d.y, -b_d.x);
    buffer_b[b_idx] = a_c + j_b_d;
    
    buffer_b[c_idx] = ac - bd;
    
    // 计算j*b_d (乘以j)
    let neg_j_b_d = vec2<f32>(-b_d.y, b_d.x);
    buffer_b[d_idx] = a_c + neg_j_b_d; // 修正：使用加号而非减号
}

// 复数乘法辅助函数
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

//fn optimized_complex_mul(a: vec2<f32>, w: vec2<f32>) -> vec2<f32> {
   // let k1 = a.x * (w.x + w.y);
   // let k2 = a.y * (w.x - w.y);
   // let k3 = (a.x + a.y) * w.y;
   // return vec2<f32>(k1 - k3, k2 + k3);
//}