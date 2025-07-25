@group(0) @binding(0)
var<storage, read_write> buffer_a: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> buffer_b: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

//const PI: f32 = 3.14159265358979323846;
const workgroup_len: u32 = 64u;

// Constants for radix-5 IFFT (same values as FFT)
const SQRT5_DIV4: f32 = 0.559016994374947; // √5/4
const SIN_2PI_DIV5: f32 = 0.9510565162951535;  // sin(2π/5)
const SIN_2PI_DIV10: f32 = 0.5877852522924731; // sin(2π/10) = sin(π/5)

struct PushConstants { fft_len: u32, stage: u32 }
var<push_constant> consts: PushConstants;

@compute @workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, 
        @builtin(num_workgroups) num_workgroups: vec3<u32>, 
        @builtin(local_invocation_index) local_invocation_index: u32) {
    
    let fft_len = consts.fft_len;
    let stage = consts.stage;
    
    // Calculate total stages and identify final stage
    let log5_n = u32(log2(f32(fft_len)) / log2(5.0) + 0.4);
    let final_stage = log5_n - 1u;
    
    // Calculate global index
    let group_size = num_workgroups.x * num_workgroups.y;
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * workgroup_len + local_invocation_index;
    if global_idx >= arrayLength(&buffer_a) / 5u {
        return;
    }
    
    // Determine which batch and butterfly this thread processes
    // Radix-5 needs fft_len/5 threads (each thread processes 5 points)
    let threads_per_fft = fft_len / 5u;
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;
    
    if (stage == 0u) {
        // Stage 0: Combine quinary bit reversal and first stage butterfly
        if (local_idx < threads_per_fft) {
            inverse_radix5_bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
        }
    } else {
        // Stage 1 and above: Standard radix-5 butterfly
        // Pass is_final_stage flag to apply 1/N scaling on last stage
        if (local_idx < threads_per_fft) {
            inverse_radix5_butterfly(local_idx, fft_len, batch_offset, stage, stage == final_stage);
        }
    }
}

// Base-5 digit reversal function (same as FFT)
fn quinary_bit_reverse(n: u32, log5_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < log5_fft_len; i++) {
        reversed = reversed * 5u + n_copy % 5u;
        n_copy = n_copy / 5u;
    }
    
    return reversed;
}

// Combine bit reversal and first stage radix-5 butterfly for IFFT
fn inverse_radix5_bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    // Compute log base 5 of n
    let log5_n = u32(log2(f32(n)) / log2(5.0) + 0.4);
    
    // Each work item processes a 5-point butterfly
    let block_idx = idx;
    
    // Calculate the five input points' bit-reversed indices
    let idx0_br = quinary_bit_reverse(block_idx * 5u + 0u, log5_n) + offset;
    let idx1_br = quinary_bit_reverse(block_idx * 5u + 1u, log5_n) + offset;
    let idx2_br = quinary_bit_reverse(block_idx * 5u + 2u, log5_n) + offset;
    let idx3_br = quinary_bit_reverse(block_idx * 5u + 3u, log5_n) + offset;
    let idx4_br = quinary_bit_reverse(block_idx * 5u + 4u, log5_n) + offset;
    
    // Read bit-reversed values
    let z0 = buffer_a[idx0_br];
    let z1 = buffer_a[idx1_br];
    let z2 = buffer_a[idx2_br];
    let z3 = buffer_a[idx3_br];
    let z4 = buffer_a[idx4_br];
    
    // Apply inverse radix-5 butterfly formula:
    // Initial calculations (same as FFT)
    let t1 = z1 + z4;
    let t2 = z2 + z3;
    let t3 = z1 - z4;
    let t4 = z2 - z3;
    
    // Further calculations (same as FFT)
    let t5 = t1 + t2;
    let t6 = SQRT5_DIV4 * (t1 - t2);
    let t5_div4 = vec2<f32>(t5.x * 0.25, t5.y * 0.25);
    let t7 = z0 - t5_div4;
    
    // Next steps (same as FFT)
    let t8 = t7 + t6;
    let t9 = t7 - t6;
    
    // Final calculations (same as FFT)
    let t10 = vec2<f32>(
        SIN_2PI_DIV5 * t3.x + SIN_2PI_DIV10 * t4.x,
        SIN_2PI_DIV5 * t3.y + SIN_2PI_DIV10 * t4.y
    );
    
    let t11 = vec2<f32>(
        SIN_2PI_DIV10 * t3.x - SIN_2PI_DIV5 * t4.x,
        SIN_2PI_DIV10 * t3.y - SIN_2PI_DIV5 * t4.y
    );
    
    // Results
    var x0 = z0 + t5;
    
    // For IFFT, multiply by -i instead of i (reverse rotation)
    // For x1 = t8 - i*t10 (opposite of FFT's t8 + i*t10)
    var x1 = vec2<f32>(t8.x + t10.y, t8.y - t10.x);
    
    // For x2 = t9 - i*t11 (opposite of FFT's t9 + i*t11)
    var x2 = vec2<f32>(t9.x + t11.y, t9.y - t11.x);
    
    // For x3 = t9 + i*t11 (opposite of FFT's t9 - i*t11)
    var x3 = vec2<f32>(t9.x - t11.y, t9.y + t11.x);
    
    // For x4 = t8 + i*t10 (opposite of FFT's t8 - i*t10)
    var x4 = vec2<f32>(t8.x - t10.y, t8.y + t10.x);
    
    // Write to output in sequential order
    let out_idx0 = block_idx * 5u + 0u + offset;
    let out_idx1 = block_idx * 5u + 1u + offset;
    let out_idx2 = block_idx * 5u + 2u + offset;
    let out_idx3 = block_idx * 5u + 3u + offset;
    let out_idx4 = block_idx * 5u + 4u + offset;

    
    buffer_b[out_idx0] = x0;
    buffer_b[out_idx1] = x1;
    buffer_b[out_idx2] = x2;
    buffer_b[out_idx3] = x3;
    buffer_b[out_idx4] = x4;
}

// Helper to calculate 5^x (same as FFT)
fn pow5(x: u32) -> u32 {
    var result = 1u;
    for (var i = 0u; i < x; i++) {
        result *= 5u;
    }
    return result;
}

// Standard inverse radix-5 butterfly operation for stages > 0
fn inverse_radix5_butterfly(idx: u32, n: u32, offset: u32, stage: u32, is_final_stage: bool) {
    // Calculate current stage butterfly parameters
    let m = pow5(stage);
    let step = 5u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // Calculate five points' indices
    let idx0 = block_idx * step + k + offset;
    let idx1 = idx0 + m;
    let idx2 = idx0 + 2u * m;
    let idx3 = idx0 + 3u * m;
    let idx4 = idx0 + 4u * m;
    
    // Read values from buffer
    let z0 = buffer_b[idx0];
    let z1_raw = buffer_b[idx1];
    let z2_raw = buffer_b[idx2];
    let z3_raw = buffer_b[idx3];
    let z4_raw = buffer_b[idx4];
    
    // Calculate twiddle factor indices
    let twiddle_base = k * (n / (5u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    let w3_idx = 3u * twiddle_base;
    let w4_idx = 4u * twiddle_base;
    
    // Get twiddle factors and take conjugate for IFFT
    // For IFFT, we use exp(j2πk/N) which is conjugate of FFT's exp(-j2πk/N)
    let w1 = vec2<f32>(twiddles[w1_idx].x, -twiddles[w1_idx].y);
    let w2 = vec2<f32>(twiddles[w2_idx].x, -twiddles[w2_idx].y);
    let w3 = vec2<f32>(twiddles[w3_idx].x, -twiddles[w3_idx].y);
    let w4 = vec2<f32>(twiddles[w4_idx].x, -twiddles[w4_idx].y);
    
    // Apply twiddle factors with complex multiplication
    let z1 = complex_mul(z1_raw, w1);
    let z2 = complex_mul(z2_raw, w2);
    let z3 = complex_mul(z3_raw, w3);
    let z4 = complex_mul(z4_raw, w4);
    
    // Apply inverse radix-5 butterfly formula with twiddle factors applied
    // Initial calculations (same as FFT)
    let t1 = z1 + z4;
    let t2 = z2 + z3;
    let t3 = z1 - z4;
    let t4 = z2 - z3;
    
    // Further calculations (same as FFT)
    let t5 = t1 + t2;
    let t6 = SQRT5_DIV4 * (t1 - t2);
    let t5_div4 = vec2<f32>(t5.x * 0.25, t5.y * 0.25);
    let t7 = z0 - t5_div4;
    
    // Next steps (same as FFT)
    let t8 = t7 + t6;
    let t9 = t7 - t6;
    
    // Final calculations (same as FFT)
    let t10 = vec2<f32>(
        SIN_2PI_DIV5 * t3.x + SIN_2PI_DIV10 * t4.x,
        SIN_2PI_DIV5 * t3.y + SIN_2PI_DIV10 * t4.y
    );
    
    let t11 = vec2<f32>(
        SIN_2PI_DIV10 * t3.x - SIN_2PI_DIV5 * t4.x,
        SIN_2PI_DIV10 * t3.y - SIN_2PI_DIV5 * t4.y
    );
    
    // Results
    var x0 = z0 + t5;
    
    // For IFFT, multiply by -i instead of i (reverse rotation)
    // For x1 = t8 - i*t10 (opposite of FFT's t8 + i*t10)
    var x1 = vec2<f32>(t8.x + t10.y, t8.y - t10.x);
    
    // For x2 = t9 - i*t11 (opposite of FFT's t9 + i*t11)
    var x2 = vec2<f32>(t9.x + t11.y, t9.y - t11.x);
    
    // For x3 = t9 + i*t11 (opposite of FFT's t9 - i*t11)
    var x3 = vec2<f32>(t9.x - t11.y, t9.y + t11.x);
    
    // For x4 = t8 + i*t10 (opposite of FFT's t8 - i*t10)
    var x4 = vec2<f32>(t8.x - t10.y, t8.y + t10.x);
    
    // Apply 1/N scaling on the final stage
    if (is_final_stage) {
        let scale = 1.0 / f32(n);
        x0 = vec2<f32>(x0.x * scale, x0.y * scale);
        x1 = vec2<f32>(x1.x * scale, x1.y * scale);
        x2 = vec2<f32>(x2.x * scale, x2.y * scale);
        x3 = vec2<f32>(x3.x * scale, x3.y * scale);
        x4 = vec2<f32>(x4.x * scale, x4.y * scale);
    }
    
    // Write results back to buffer
    buffer_b[idx0] = x0;
    buffer_b[idx1] = x1;
    buffer_b[idx2] = x2;
    buffer_b[idx3] = x3;
    buffer_b[idx4] = x4;
}

// Complex multiplication helper (same as FFT)
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}