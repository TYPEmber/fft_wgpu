
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
    
    // Calculate global index
    let group_size = num_workgroups.x * num_workgroups.y;
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * workgroup_len + local_invocation_index;
    if global_idx >= arrayLength(&buffer_a) / 3u {
        return;
    }
    
    // Determine which batch and butterfly this thread processes
    let threads_per_fft = fft_len / 3u;
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;

    if (stage == 0u) {
        if (local_idx < threads_per_fft) {
            radix3_bit_reversal_and_butterfly_f64(local_idx, fft_len, batch_offset);
        }
    } else {
        if (local_idx < threads_per_fft) {
            radix3_butterfly_f64(local_idx, fft_len, batch_offset, stage);
        }
    }
}

// 三进制位反转函数 - 整数版本保持不变
fn ternary_bit_reverse(n: u32, log3_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < log3_fft_len; i++) {
        reversed = reversed * 3u + n_copy % 3u;
        n_copy = n_copy / 3u;
    }
    
    return reversed;
}

// 高精度版本的基3蝶形+位反转
fn radix3_bit_reversal_and_butterfly_f64(idx: u32, n: u32, offset: u32) {
    // 计算log3(n)，使用f64提高精度
    let log3_n = u32(log2(f64(n)) / log2(3.0) + 0.4);
    
    let block_idx = idx;
    
    // 计算位反转索引
    let a_idx_br = ternary_bit_reverse(block_idx * 3u + 0u, log3_n) + offset;
    let b_idx_br = ternary_bit_reverse(block_idx * 3u + 1u, log3_n) + offset;
    let c_idx_br = ternary_bit_reverse(block_idx * 3u + 2u, log3_n) + offset;
    
    // 读取数据并转换为f64
    let z0 = vec2<f64>(buffer_a[a_idx_br]);
    let z1 = vec2<f64>(buffer_a[b_idx_br]);
    let z2 = vec2<f64>(buffer_a[c_idx_br]);
    
    // 使用f64进行计算
    let t1 = z1 + z2;
    let t1_half = vec2<f64>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    let z1_minus_z2 = z1 - z2;
    
    // 使用高精度SQRT3_DIV2计算
    let t3 = vec2<f64>(
        SQRT3_DIV2_D * z1_minus_z2.y, 
        -SQRT3_DIV2_D * z1_minus_z2.x
    );
    
    // 计算输出
    let x0 = z0 + t1;
    let x1 = vec2<f64>(t2.x - t3.y, t2.y + t3.x);
    let x2 = vec2<f64>(t2.x + t3.y, t2.y - t3.x);
    
    // 转换回f32并写入
    let out_idx_a = block_idx * 3u + 0u + offset;
    let out_idx_b = block_idx * 3u + 1u + offset;
    let out_idx_c = block_idx * 3u + 2u + offset;
    
    buffer_b[out_idx_a] = vec2<f32>(x0);
    buffer_b[out_idx_b] = vec2<f32>(x1);
    buffer_b[out_idx_c] = vec2<f32>(x2);
}

// 高精度版本的基3蝶形运算
fn radix3_butterfly_f64(idx: u32, n: u32, offset: u32, stage: u32) {
    // 使用整数计算子问题大小和步长
    let m = pow3(stage);
    let step = 3u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // 计算三个点的索引
    let a_idx = block_idx * step + k + offset;
    let b_idx = a_idx + m;
    let c_idx = a_idx + 2u * m;
    
    // 读取数据并转换为f64
    let z0 = vec2<f64>(buffer_b[a_idx]);
    let z1_raw = vec2<f64>(buffer_b[b_idx]);
    let z2_raw = vec2<f64>(buffer_b[c_idx]);
    
    // 计算旋转因子索引
    let twiddle_base = k * (n / (3u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    
    // 获取旋转因子并转换为f64
    let w1 = vec2<f64>(twiddles[w1_idx]);
    let w2 = vec2<f64>(twiddles[w2_idx]);
    
    // 使用f64进行复数乘法
    let z1 = complex_mul_f64(z1_raw, w1);
    let z2 = complex_mul_f64(z2_raw, w2);
    
    // 使用f64进行蝶形计算
    let t1 = z1 + z2;
    let t1_half = vec2<f64>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    let z1_minus_z2 = z1 - z2;
    
    // 使用高精度SQRT3_DIV2计算
    let t3 = vec2<f64>(
        SQRT3_DIV2_D * z1_minus_z2.y, 
        -SQRT3_DIV2_D * z1_minus_z2.x
    );
    
    // 计算输出
    let x0 = z0 + t1;
    let x1 = vec2<f64>(t2.x - t3.y, t2.y + t3.x);
    let x2 = vec2<f64>(t2.x + t3.y, t2.y - t3.x);
    
    // 转换回f32并写回
    buffer_b[a_idx] = vec2<f32>(x0);
    buffer_b[b_idx] = vec2<f32>(x1);
    buffer_b[c_idx] = vec2<f32>(x2);
}

// 计算3^x的辅助函数 - 整数版本保持不变
fn pow3(x: u32) -> u32 {
    var result = 1u;
    for (var i = 0u; i < x; i++) {
        result *= 3u;
    }
    return result;
}

// f64版本的复数乘法
fn complex_mul_f64(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// f64版本的优化复数乘法
//fn optimized_complex_mul_f64(a: vec2<f64>, w: vec2<f64>) -> vec2<f64> {
    //let k1 = a.x * (w.x + w.y);
   // let k2 = a.y * (w.x - w.y);
   // let k3 = (a.x + a.y) * w.y;
   // return vec2<f64>(k1 - k3, k2 + k3);
//}