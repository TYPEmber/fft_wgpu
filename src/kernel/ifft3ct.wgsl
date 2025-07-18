@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

const PI: f32 = 3.14159265358979323846;
const SQRT3_DIV2: f32 = 0.866025403784439; // sin(π/3) = √3/2
const workgroup_len: u32 = 64u;

struct PushConstants { fft_len: u32, stage: u32 }
var<push_constant> consts: PushConstants;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, 
        @builtin(num_workgroups) num_workgroups: vec3<u32>, 
        @builtin(local_invocation_index) local_invocation_index: u32) {
    
    let fft_len = consts.fft_len;
    let stage = consts.stage;
    let final_stage = consts.stage == (u32(log2(f32(fft_len)) / log2(3.0) + 0.4)-1u);
    // Calculate global index
    let group_size = num_workgroups.x * num_workgroups.y;
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * workgroup_len + local_invocation_index;
    if global_idx >= arrayLength(&buffer_a) / 3u {
        return;
    }
    
    // Determine which batch and butterfly this thread processes
    // Radix-3 needs fft_len/3 threads (each thread processes 3 points)
    let threads_per_fft = fft_len / 3u;
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;

    if (stage == 0u) {
        // Stage 0: Combine ternary bit reversal and first stage butterfly
        if (local_idx < threads_per_fft) {
            inverse_radix3_bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
        }
    } else {
        // Stage 1 and above: Standard radix-3 butterfly
        if (local_idx < threads_per_fft) {
            inverse_radix3_butterfly(local_idx, fft_len, batch_offset, stage,  final_stage);
        }
    }
}

// Ternary bit reversal function (same as FFT)
fn ternary_bit_reverse(n: u32, log3_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < log3_fft_len; i++) {
        reversed = reversed * 3u + n_copy % 3u;
        n_copy = n_copy / 3u;
    }
    
    return reversed;
}

// Combine bit reversal and first stage radix-3 butterfly for IFFT
fn inverse_radix3_bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    // Compute log base 3 of n
    let log3_n = u32(log2(f32(n)) / log2(3.0) + 0.4);
    
    // Each work item processes a 3-point butterfly
    let block_idx = idx;
    
    // Calculate the three input points' bit-reversed indices
    let a_idx_br = ternary_bit_reverse(block_idx * 3u + 0u, log3_n) + offset;
    let b_idx_br = ternary_bit_reverse(block_idx * 3u + 1u, log3_n) + offset;
    let c_idx_br = ternary_bit_reverse(block_idx * 3u + 2u, log3_n) + offset;
    
    // Read bit-reversed values
    let z0 = buffer_a[a_idx_br];
    let z1 = buffer_a[b_idx_br];
    let z2 = buffer_a[c_idx_br];
    
    // Apply inverse radix-3 butterfly formula:
    // Note: For IFFT, we use conjugate of the twiddle factors,
    // which changes the sign of the imaginary part
    
    let t1 = z1 + z2;
    let t1_half = vec2<f32>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    let z1_minus_z2 = z1 - z2;
    
    // Calculate t3 = sin(π/3)(z1 - z2)
    // Note: For IFFT, we change the sign of the imaginary part
    let t3 = vec2<f32>(
        -SQRT3_DIV2 * z1_minus_z2.y,  // Note the sign change here
        SQRT3_DIV2 * z1_minus_z2.x    // Note the sign change here
    );
    
    // Calculate outputs
    let x0 = z0 + t1;
    let x1 = vec2<f32>(t2.x - t3.y, t2.y + t3.x);
    let x2 = vec2<f32>(t2.x + t3.y, t2.y - t3.x);
    
    // Write to output in sequential order
    let out_idx_a = block_idx * 3u + 0u + offset;
    let out_idx_b = block_idx * 3u + 1u + offset;
    let out_idx_c = block_idx * 3u + 2u + offset;

    if (log3_n==1u) {
        let scale = 1.0 / f32(n);
        x0 = vec2<f32>(x0.x * scale, x0.y * scale);
        x1 = vec2<f32>(x1.x * scale, x1.y * scale);
        x2 = vec2<f32>(x2.x * scale, x2.y * scale);
    }
    
    buffer_b[out_idx_a] = x0;
    buffer_b[out_idx_b] = x1;
    buffer_b[out_idx_c] = x2;
}

// Standard inverse radix-3 butterfly operation for stages > 0
fn inverse_radix3_butterfly(idx: u32, n: u32, offset: u32, stage: u32, is_final_stage: bool) {
    // Calculate current stage butterfly parameters
    let m = pow3(stage);
    let step = 3u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // Calculate three points' indices
    let a_idx = block_idx * step + k + offset;
    let b_idx = a_idx + m;
    let c_idx = a_idx + 2u * m;
    
    // Read values from buffer
    let z0 = buffer_b[a_idx];
    let z1 = buffer_b[b_idx];
    let z2 = buffer_b[c_idx];
    
    // Calculate twiddle factor indices
    let twiddle_base = k * (n / (3u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    
    // Get and apply twiddle factors (conjugate for IFFT)
    // Twiddle factors are stored as exp(-j2πk/N), but IFFT needs exp(j2πk/N)
    // So we take the complex conjugate by changing the sign of the imaginary part
    let w1 = vec2<f32>(twiddles[w1_idx].x, -twiddles[w1_idx].y);
    let w2 = vec2<f32>(twiddles[w2_idx].x, -twiddles[w2_idx].y);
    
    let z1_rot = complex_mul(z1, w1);
    let z2_rot = complex_mul(z2, w2);
    
    // Apply radix-3 butterfly formula
    let t1 = z1_rot + z2_rot;
    let t1_half = vec2<f32>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    let z1_minus_z2 = z1_rot - z2_rot;
    
    // Calculate t3 = sin(π/3)(z1 - z2)
    // Note the sign change for IFFT
    let t3 = vec2<f32>(
        -SQRT3_DIV2 * z1_minus_z2.y,
        SQRT3_DIV2 * z1_minus_z2.x
    );
    
    // Calculate outputs
    var x0 = z0 + t1;
    var x1 = vec2<f32>(t2.x - t3.y, t2.y + t3.x);
    var x2 = vec2<f32>(t2.x + t3.y, t2.y - t3.x);
    
    // Apply 1/N scaling on the final stage
    if (is_final_stage) {
        let scale = 1.0 / f32(n);
        x0 = vec2<f32>(x0.x * scale, x0.y * scale);
        x1 = vec2<f32>(x1.x * scale, x1.y * scale);
        x2 = vec2<f32>(x2.x * scale, x2.y * scale);
    }
    
    // Write back results
    buffer_b[a_idx] = x0;
    buffer_b[b_idx] = x1;
    buffer_b[c_idx] = x2;
}

// Helper to calculate 3^x (same as FFT)
fn pow3(x: u32) -> u32 {
    var result = 1u;
    for (var i = 0u; i < x; i++) {
        result *= 3u;
    }
    return result;
}

// Complex multiplication helper (same as FFT)
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}