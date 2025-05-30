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
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    let fft_len = consts.fft_len;
    let stage = consts.stage;
    let index = (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x) * workgroup_len + local_invocation_index;
    let batch_idx = index / fft_len;
    let batch_offset = batch_idx * fft_len;
    let local_idx = index % (fft_len / 2u);
    if (stage == 0u) {
         //阶段0：结合位反转和第一阶段蝶形运算
        // 每个线程处理一对元素 - 需要 fft_len/2 个线程
        
        if (local_idx < fft_len / 2u) {
            bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
        }
    } else {
        // 阶段 1 及以上：标准蝶形运算
        // 注意：实际执行的是阶段 stage+1，因为阶段0已经包含了第一次蝶形运算
        if (local_idx < fft_len / 2u) {
            fft_butterfly(local_idx, fft_len, batch_offset, stage + 1u);
        }
    }
}

// 组合位反转和第一阶段蝶形运算的函数
fn bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    // 第一步：位反转 - 将 buffer_a 中的数据按位反转顺序读入临时数组
    let bits = u32(log2(f32(n)));
    
    // 蝶形运算的参数 (m=1 表示第一阶段)
    let m = 1u;  // 第一阶段蝶形运算的子问题大小
    let step = 2u * m; // 子问题之间的步长 = 2
    let k = idx % m;   // 在每个子问题中的位置 (始终为0，因为m=1)
    let block_idx = idx / m; // 子问题索引
    
    // 计算第一阶段蝶形运算的输入索引
    //let a_idx_br = bit_reverse(block_idx * step + k, bits) + offset;
    let a_idx_br = bit_reverse(block_idx * 2 , bits) + offset;
    let b_idx_br = bit_reverse(block_idx * 2  + 1, bits) + offset;
    
    // 从 buffer_a 读取位反转后的值
    let a = buffer_a[a_idx_br];
    let b = buffer_a[b_idx_br];
    
    // 获取旋转因子 (第一阶段是简单的 -1 旋转)
    //let twiddle = twiddles[k * (n / (2u * m))]; // 对第一阶段，k=0，所以取twiddles[0]
    //let twiddle=twiddles[0]; // 第一阶段的旋转因子是1
    let twiddle = vec2<f32>(1.0, 0.0); // 第一阶段的旋转因子是1
    // 计算输出位置 - 按正常顺序写入
   // let out_idx_a = block_idx * step + k + offset;
    let out_idx_a = block_idx * 2 + offset;
    //let out_idx_b = out_idx_a + m;
    let out_idx_b = out_idx_a + 1;
    
    // 执行蝶形运算并写入 buffer_b
    let b_twiddle = complex_mul(b, twiddle);
    buffer_b[out_idx_a] = a + b_twiddle;
    buffer_b[out_idx_b] = a - b_twiddle;
}

// 标准 Cooley-Tukey 蝶形运算 - 只在 buffer_b 中操作
fn fft_butterfly(idx: u32, n: u32, offset: u32, stage: u32) {
    // 计算当前阶段的蝶形运算参数
    let m = 1u << (stage - 1u); // 子问题大小
    let step = 2u * m;          // 子问题之间的步长，相当于block
    let k = idx % m;            // block内的位置
    let block_idx = idx / m;    // block索引
    
    // 计算输入/输出位置
    let a_idx = block_idx * step + k + offset;
    let b_idx = a_idx + m;
    
    // 获取旋转因子
    let twiddle_idx = k * (n / (2u * m));
    let twiddle = twiddles[twiddle_idx];
    
    // 执行蝶形运算 - 在 buffer_b 内读写
    let a = buffer_b[a_idx];
    let b = buffer_b[b_idx];
   //let a= vec2<f32>(1.0, 0.0);
   //let b= vec2<f32>(2.0, 0.0);
    let b_twiddle = complex_mul(b, twiddle);
    
    // 写回结果到同一缓冲区
    buffer_b[a_idx] = a + b_twiddle;

    buffer_b[b_idx] = a - b_twiddle;
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

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}