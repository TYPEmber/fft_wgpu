
@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

const PI_D: f64 = 3.141592653589793238462643383279502884;
const SQRT3_DIV2_D: f64 = 0.8660254037844386467637231707529361834714; // sin(π/3) = √3/2 高精度版本
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
    
    // 基-6需要fft_len/6线程
    let threads_per_fft = fft_len / 6u;
    if global_idx >= arrayLength(&buffer_a) / fft_len * threads_per_fft {
        return;
    }
    
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;

    if (stage == 0u) {
        if (local_idx < threads_per_fft) {
            radix6_bit_reversal_and_butterfly_f64(local_idx, fft_len, batch_offset);
        }
    } else {
        if (local_idx < threads_per_fft) {
            radix6_butterfly_f64(local_idx, fft_len, batch_offset, stage);
        }
    }
}

// 基6位反转函数 - 整数版本保持不变
fn base6_bit_reverse(n: u32, log6_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < log6_fft_len; i++) {
        reversed = reversed * 6u + n_copy % 6u;
        n_copy = n_copy / 6u;
    }
    
    return reversed;
}

// 高精度版本的基6蝶形+位反转
fn radix6_bit_reversal_and_butterfly_f64(idx: u32, n: u32, offset: u32) {
    // 使用f64计算log6(n)
    let log6_n = u32(log2(f64(n)) / log2(6.0) + 0.4);
    
    let block_idx = idx;
    
    // 计算六个输入点的位反转索引
    let z0_idx_br = base6_bit_reverse(block_idx * 6u + 0u, log6_n) + offset;
    let z1_idx_br = base6_bit_reverse(block_idx * 6u + 1u, log6_n) + offset;
    let z2_idx_br = base6_bit_reverse(block_idx * 6u + 2u, log6_n) + offset;
    let z3_idx_br = base6_bit_reverse(block_idx * 6u + 3u, log6_n) + offset;
    let z4_idx_br = base6_bit_reverse(block_idx * 6u + 4u, log6_n) + offset;
    let z5_idx_br = base6_bit_reverse(block_idx * 6u + 5u, log6_n) + offset;
    
    // 读取数据并转换为f64
    let z0 = vec2<f64>(buffer_a[z0_idx_br]);
    let z1 = vec2<f64>(buffer_a[z1_idx_br]);
    let z2 = vec2<f64>(buffer_a[z2_idx_br]);
    let z3 = vec2<f64>(buffer_a[z3_idx_br]);
    let z4 = vec2<f64>(buffer_a[z4_idx_br]);
    let z5 = vec2<f64>(buffer_a[z5_idx_br]);
    
    // 使用f64进行PFA计算
    // t1 = z2 + z4
    let t1 = z2 + z4;
    
    // t2 = z0 - t1/2
    let t1_half = vec2<f64>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    
    // t3 = sin(π/3)(z2 - z4)
    let z2_minus_z4 = z2 - z4;
    let t3 = vec2<f64>(
        SQRT3_DIV2_D * z2_minus_z4.x,
        SQRT3_DIV2_D * z2_minus_z4.y
    );
    
    // t4 = z5 + z1
    let t4 = z5 + z1;
    
    // t5 = z3 - t4/2
    let t4_half = vec2<f64>(t4.x * 0.5, t4.y * 0.5);
    let t5 = z3 - t4_half;
    
    // t6 = sin(π/3)(z5 - z1)
    let z5_minus_z1 = z5 - z1;
    let t6 = vec2<f64>(
        SQRT3_DIV2_D * z5_minus_z1.x,
        SQRT3_DIV2_D * z5_minus_z1.y
    );
    
    // t7 = z0 + t1
    let t7 = z0 + t1;
    
    // t8 = t2 + i*t3
    let t8 = vec2<f64>(t2.x - t3.y, t2.y + t3.x); // 复数乘以i
    
    // t9 = t2 - i*t3
    let t9 = vec2<f64>(t2.x + t3.y, t2.y - t3.x); // 复数乘以-i
    
    // t10 = z3 + t4
    let t10 = z3 + t4;
    
    // t11 = t5 + i*t6
    let t11 = vec2<f64>(t5.x - t6.y, t5.y + t6.x); // 复数乘以i
    
    // t12 = t5 - i*t6
    let t12 = vec2<f64>(t5.x + t6.y, t5.y - t6.x); // 复数乘以-i
    
    // 最终输出
    let x0 = t7 + t10;
    let x4 = t8 + t11;
    let x2 = t9 + t12;
    let x3 = t7 - t10;
    let x1 = t8 - t11;
    let x5 = t9 - t12;
    
    // 转换回f32并写入
    let out_idx = block_idx * 6u + offset;
    buffer_b[out_idx + 0u] = vec2<f32>(x0);
    buffer_b[out_idx + 1u] = vec2<f32>(x1);
    buffer_b[out_idx + 2u] = vec2<f32>(x2);
    buffer_b[out_idx + 3u] = vec2<f32>(x3);
    buffer_b[out_idx + 4u] = vec2<f32>(x4);
    buffer_b[out_idx + 5u] = vec2<f32>(x5);
}

// 高精度版本的基6蝶形运算
fn radix6_butterfly_f64(idx: u32, n: u32, offset: u32, stage: u32) {
    // 使用整数计算子问题大小和步长
    let m = pow6(stage);
    let step = 6u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // 计算六个点的索引
    let z0_idx = block_idx * step + k + offset;
    let z1_idx = z0_idx + m;
    let z2_idx = z0_idx + 2u * m;
    let z3_idx = z0_idx + 3u * m;
    let z4_idx = z0_idx + 4u * m;
    let z5_idx = z0_idx + 5u * m;
    
    // 读取数据并转换为f64
    let z0_raw = vec2<f64>(buffer_b[z0_idx]);
    let z1_raw = vec2<f64>(buffer_b[z1_idx]);
    let z2_raw = vec2<f64>(buffer_b[z2_idx]);
    let z3_raw = vec2<f64>(buffer_b[z3_idx]);
    let z4_raw = vec2<f64>(buffer_b[z4_idx]);
    let z5_raw = vec2<f64>(buffer_b[z5_idx]);
    
    // 计算旋转因子索引
    let twiddle_base = k * (n / (6u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    let w3_idx = 3u * twiddle_base;
    let w4_idx = 4u * twiddle_base;
    let w5_idx = 5u * twiddle_base;
    
    // 获取旋转因子并转换为f64
    let w1 = vec2<f64>(twiddles[w1_idx % arrayLength(&twiddles)]);
    let w2 = vec2<f64>(twiddles[w2_idx % arrayLength(&twiddles)]);
    let w3 = vec2<f64>(twiddles[w3_idx % arrayLength(&twiddles)]);
    let w4 = vec2<f64>(twiddles[w4_idx % arrayLength(&twiddles)]);
    let w5 = vec2<f64>(twiddles[w5_idx % arrayLength(&twiddles)]);
    
    // 使用f64进行复数乘法
    let z0 = z0_raw;
    let z1 = complex_mul_f64(z1_raw, w1);
    let z2 = complex_mul_f64(z2_raw, w2);
    let z3 = complex_mul_f64(z3_raw, w3);
    let z4 = complex_mul_f64(z4_raw, w4);
    let z5 = complex_mul_f64(z5_raw, w5);
    
    // 使用f64进行PFA计算
    // t1 = z2 + z4
    let t1 = z2 + z4;
    
    // t2 = z0 - t1/2
    let t1_half = vec2<f64>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    
    // t3 = sin(π/3)(z2 - z4)
    let z2_minus_z4 = z2 - z4;
    let t3 = vec2<f64>(
        SQRT3_DIV2_D * z2_minus_z4.x,
        SQRT3_DIV2_D * z2_minus_z4.y
    );
    
    // t4 = z5 + z1
    let t4 = z5 + z1;
    
    // t5 = z3 - t4/2
    let t4_half = vec2<f64>(t4.x * 0.5, t4.y * 0.5);
    let t5 = z3 - t4_half;
    
    // t6 = sin(π/3)(z5 - z1)
    let z5_minus_z1 = z5 - z1;
    let t6 = vec2<f64>(
        SQRT3_DIV2_D * z5_minus_z1.x,
        SQRT3_DIV2_D * z5_minus_z1.y
    );
    
    // t7 = z0 + t1
    let t7 = z0 + t1;
    
    // t8 = t2 + i*t3
    let t8 = vec2<f64>(t2.x - t3.y, t2.y + t3.x);
    
    // t9 = t2 - i*t3
    let t9 = vec2<f64>(t2.x + t3.y, t2.y - t3.x);
    
    // t10 = z3 + t4
    let t10 = z3 + t4;
    
    // t11 = t5 + i*t6
    let t11 = vec2<f64>(t5.x - t6.y, t5.y + t6.x);
    
    // t12 = t5 - i*t6
    let t12 = vec2<f64>(t5.x + t6.y, t5.y - t6.x);
    
    // 最终输出
    let x0 = t7 + t10;
    let x4 = t8 + t11;
    let x2 = t9 + t12;
    let x3 = t7 - t10;
    let x1 = t8 - t11;
    let x5 = t9 - t12;
    
    // 转换回f32并写回
    buffer_b[z0_idx] = vec2<f32>(x0);
    buffer_b[z1_idx] = vec2<f32>(x1);
    buffer_b[z2_idx] = vec2<f32>(x2);
    buffer_b[z3_idx] = vec2<f32>(x3);
    buffer_b[z4_idx] = vec2<f32>(x4);
    buffer_b[z5_idx] = vec2<f32>(x5);
}

// 计算6^x的辅助函数 - 整数版本保持不变
fn pow6(x: u32) -> u32 {
    var result = 1u;
    for (var i = 0u; i < x; i++) {
        result *= 6u;
    }
    return result;
}

// f64版本的复数乘法
fn complex_mul_f64(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
