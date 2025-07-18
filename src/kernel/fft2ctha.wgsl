
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
    
    if global_idx >= arrayLength(&buffer_a) / 2u {
        return;
    }
    
    let batch_idx = global_idx / fft_len;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % (fft_len / 2u);
    
    if (stage == 0u) {
        if (local_idx < fft_len / 2u) {
            bit_reversal_and_butterfly_f64(local_idx, fft_len, batch_offset);
        }
    } else {
        if (local_idx < fft_len / 2u) {
            fft_butterfly_f64(local_idx, fft_len, batch_offset, stage + 1u);
        }
    }
}

// 组合位反转和第一阶段蝶形运算的高精度实现
fn bit_reversal_and_butterfly_f64(idx: u32, n: u32, offset: u32) {
    let bits = u32(log2(f64(n))+0.4);
    
    let m = 1u;
    let step = 2u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    let a_idx_br = bit_reverse(block_idx * 2, bits) + offset;
    let b_idx_br = bit_reverse(block_idx * 2 + 1, bits) + offset;
    
    // 从 buffer_a 读取并转换为 f64
    let a = vec2<f64>(buffer_a[a_idx_br]);
    let b = vec2<f64>(buffer_a[b_idx_br]);
    
    // 高精度旋转因子
    let twiddle = vec2<f64>(1.0, 0.0);
    
    let out_idx_a = block_idx * 2 + offset;
    let out_idx_b = out_idx_a + 1;
    
    // 使用 f64 执行蝶形运算
    let b_twiddle = complex_mul_f64(b, twiddle);
    
    // 转回 f32 并写入 buffer_b
    buffer_b[out_idx_a] = vec2<f32>(a + b_twiddle);
    buffer_b[out_idx_b] = vec2<f32>(a - b_twiddle);
}

// 标准蝶形运算的高精度实现
fn fft_butterfly_f64(idx: u32, n: u32, offset: u32, stage: u32) {
    let m = 1u << (stage - 1u);
    let step = 2u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    let a_idx = block_idx * step + k + offset;
    let b_idx = a_idx + m;
    
    // 获取旋转因子并转换为 f64
    let twiddle_idx = k * (n / (2u * m));
    let twiddle = vec2<f64>(twiddles[twiddle_idx]); //因为没有f64的sin与cos，twiddle必须外置计算
    
    // 从 buffer_b 读取并转换为 f64
    let a = vec2<f64>(buffer_b[a_idx]);
    let b = vec2<f64>(buffer_b[b_idx]);
    
    // 使用 f64 执行复数乘法和蝶形运算
    let b_twiddle = complex_mul_f64(b, twiddle);
    
    // 转回 f32 并写回 buffer_b
    buffer_b[a_idx] = vec2<f32>(a + b_twiddle);
    buffer_b[b_idx] = vec2<f32>(a - b_twiddle);
}

// 辅助函数
fn bit_reverse(n: u32, bits: u32) -> u32 {
    var reversed = 0u;
    var n1 = n;
    for (var i = 0u; i < bits; i++) {
        reversed = (reversed << 1u) | (n1 & 1u);
        n1 >>= 1u;
    }
    return reversed;
}

// f64 版本的复数乘法
fn complex_mul_f64(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// f64 版本的优化复数乘法
