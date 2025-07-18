//enable f64; // <--- 在文件顶部启用f64扩展

@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

// 使用f64定义高精度常量
//const PI_D: f64 = 3.141592653589793;
const SQRT5_DIV4_D: f64 = 0.5590169943749475; // sqrt(5.0)/4.0
const SIN_2PI_DIV5_D: f64 = 0.9510565162951535;  // sin(2*PI/5)
const SIN_2PI_DIV10_D: f64 = 0.5877852522924731; // sin(PI/5)

struct PushConstants { fft_len: u32, stage: u32 }
var<push_constant> consts: PushConstants;

fn pow5(x: u32) -> u32 {
    var result = 1u;
    for (var i = 0u; i < x; i++) {
        result *= 5u;
    }
    return result;
}

// Base-5 digit reversal function
fn quinary_bit_reverse1(n: u32, log5_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < log5_fft_len; i++) {
        reversed = reversed * 5u + n_copy % 5u;
        n_copy = n_copy / 5u;
    }
    
    return reversed;
}


// f64版本的复数乘法
fn complex_mul_f64(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}


// Stage 0: 使用f64进行计算
fn radix5_bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    let log5_n = u32(log2(f32(n)) / log2(5.0)+0.4);
    
    let block_idx = idx;
    
    // 读取 f32 数据并立即转换为 f64
    let z0 = vec2<f64>(buffer_a[quinary_bit_reverse(block_idx * 5u + 0u, log5_n) + offset]);
    let z1 = vec2<f64>(buffer_a[quinary_bit_reverse(block_idx * 5u + 1u, log5_n) + offset]);
    let z2 = vec2<f64>(buffer_a[quinary_bit_reverse(block_idx * 5u + 2u, log5_n) + offset]);
    let z3 = vec2<f64>(buffer_a[quinary_bit_reverse(block_idx * 5u + 3u, log5_n) + offset]);
    let z4 = vec2<f64>(buffer_a[quinary_bit_reverse(block_idx * 5u + 4u, log5_n) + offset]);
    
    // --- 所有中间计算都使用 f64 ---
    let t1 = z1 + z4;
    let t2 = z2 + z3;
    let t3 = z1 - z4;
    let t4 = z2 - z3;
    
    let t5 = t1 + t2;
    let t6 = SQRT5_DIV4_D * (t1 - t2);
    let t5_div4 = vec2<f64>(t5.x * 0.25, t5.y * 0.25);
    let t7 = z0 - t5_div4;
    
    let t8 = t7 + t6;
    let t9 = t7 - t6;
    
    let t10 = vec2<f64>(
        SIN_2PI_DIV5_D * t3.x + SIN_2PI_DIV10_D * t4.x,
        SIN_2PI_DIV5_D * t3.y + SIN_2PI_DIV10_D * t4.y
    );
    
    let t11 = vec2<f64>(
        SIN_2PI_DIV10_D * t3.x - SIN_2PI_DIV5_D * t4.x,
        SIN_2PI_DIV10_D * t3.y - SIN_2PI_DIV5_D * t4.y
    );
    
    let x0 = z0 + t5;
    let x1 = vec2<f64>(t8.x - t10.y, t8.y + t10.x);
    let x2 = vec2<f64>(t9.x - t11.y, t9.y + t11.x);
    let x3 = vec2<f64>(t9.x + t11.y, t9.y - t11.x);
    let x4 = vec2<f64>(t8.x + t10.y, t8.y - t10.x);
    
    // 写回时从 f64 转换回 f32
    buffer_b[block_idx * 5u + 0u + offset] = vec2<f32>(x0);
    buffer_b[block_idx * 5u + 1u + offset] = vec2<f32>(x1);
    buffer_b[block_idx * 5u + 2u + offset] = vec2<f32>(x2);
    buffer_b[block_idx * 5u + 3u + offset] = vec2<f32>(x3);
    buffer_b[block_idx * 5u + 4u + offset] = vec2<f32>(x4);
}

// Stage > 0: 使用f64进行计算
fn radix5_butterfly(idx: u32, n: u32, offset: u32, stage: u32) {
    let m = pow5(stage);
    let step = 5u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    let idx0 = block_idx * step + k + offset;
    
    // 读取 f32 数据并转换为 f64
    let z0 = vec2<f64>(buffer_b[idx0]);
    let z1_raw = vec2<f64>(buffer_b[idx0 + m]);
    let z2_raw = vec2<f64>(buffer_b[idx0 + 2u * m]);
    let z3_raw = vec2<f64>(buffer_b[idx0 + 3u * m]);
    let z4_raw = vec2<f64>(buffer_b[idx0 + 4u * m]);
    
    let twiddle_base = k * (n / (5u * m));
    let w1 = vec2<f64>(twiddles[twiddle_base]);
    let w2 = vec2<f64>(twiddles[2u * twiddle_base]);
    let w3 = vec2<f64>(twiddles[3u * twiddle_base]);
    let w4 = vec2<f64>(twiddles[4u * twiddle_base]);
    
    // --- 所有中间计算都使用 f64 ---
    let z1 = complex_mul_f64(z1_raw, w1);
    let z2 = complex_mul_f64(z2_raw, w2);
    let z3 = complex_mul_f64(z3_raw, w3);
    let z4 = complex_mul_f64(z4_raw, w4);
    
    let t1 = z1 + z4;
    let t2 = z2 + z3;
    // ... (此处省略与上面函数中完全相同的f64计算)
    let t3 = z1 - z4;
    let t4 = z2 - z3;
    
    let t5 = t1 + t2;
    let t6 = SQRT5_DIV4_D * (t1 - t2);
    let t5_div4 = vec2<f64>(t5.x * 0.25, t5.y * 0.25);
    let t7 = z0 - t5_div4;
    
    let t8 = t7 + t6;
    let t9 = t7 - t6;
    
    let t10 = vec2<f64>(
        SIN_2PI_DIV5_D * t3.x + SIN_2PI_DIV10_D * t4.x,
        SIN_2PI_DIV5_D * t3.y + SIN_2PI_DIV10_D * t4.y
    );
    
    let t11 = vec2<f64>(
        SIN_2PI_DIV10_D * t3.x - SIN_2PI_DIV5_D * t4.x,
        SIN_2PI_DIV10_D * t3.y - SIN_2PI_DIV5_D * t4.y
    );
    
    let x0 = z0 + t5;
    let x1 = vec2<f64>(t8.x - t10.y, t8.y + t10.x);
    let x2 = vec2<f64>(t9.x - t11.y, t9.y + t11.x);
    let x3 = vec2<f64>(t9.x + t11.y, t9.y - t11.x);
    let x4 = vec2<f64>(t8.x + t10.y, t8.y - t10.x);
    
    // 写回时从 f64 转换回 f32
    buffer_b[idx0] = vec2<f32>(x0);
    buffer_b[idx0 + m] = vec2<f32>(x1);
    buffer_b[idx0 + 2u * m] = vec2<f32>(x2);
    buffer_b[idx0 + 3u * m] = vec2<f32>(x3);
    buffer_b[idx0 + 4u * m] = vec2<f32>(x4);
}

// 移除f32的complex_mul，因为它不再被调用
// fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { ... }

// ... (main函数和其它辅助函数)
// 注意：main函数中调用radix5_butterfly和radix5_bit_reversal_and_butterfly的部分不需要修改
// 也不需要修改`buffer_a`和`buffer_b`的定义，它们仍然是`vec2<f32>`
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, 
        @builtin(num_workgroups) num_workgroups: vec3<u32>, 
        @builtin(local_invocation_index) local_invocation_index: u32) {
    
    let fft_len = consts.fft_len;
    let stage = consts.stage;
    
    let group_size = num_workgroups.x * num_workgroups.y;
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * 64u + local_invocation_index;
    
    let threads_per_fft = fft_len / 5u;

    if global_idx >= (arrayLength(&buffer_a) / fft_len) * threads_per_fft {
        return;
    }
    
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;
    
    if (stage == 0u) {
        radix5_bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
    } else {
        radix5_butterfly(local_idx, fft_len, batch_offset, stage);
    }
}

fn quinary_bit_reverse(n: u32, log5_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    // 根据常见长度进行循环展开
    switch log5_fft_len {
        case 1u: {
            reversed = n_copy % 5u;
        }
        case 2u: {
            reversed = (n_copy % 5u) * 5u + (n_copy / 5u) % 5u;
        }
        case 3u: {
            reversed = ((n_copy % 5u) * 25u) + 
                      ((n_copy / 5u) % 5u) * 5u + 
                      (n_copy / 25u) % 5u;
        }
        case 4u: {
            reversed = ((n_copy % 5u) * 125u) + 
                      ((n_copy / 5u) % 5u) * 25u + 
                      ((n_copy / 25u) % 5u) * 5u + 
                      (n_copy / 125u) % 5u;
        }
        default: {
            for (var i = 0u; i < log5_fft_len; i++) {
                reversed = reversed * 5u + n_copy % 5u;
                n_copy = n_copy / 5u;
            }
        }
    }
    
    return reversed;
}