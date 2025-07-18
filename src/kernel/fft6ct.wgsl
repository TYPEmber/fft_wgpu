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
    
    // Calculate global index
    let group_size = num_workgroups.x * num_workgroups.y;
    let group_idx = workgroup_id.x + workgroup_id.y * num_workgroups.x + 
                   workgroup_id.z * group_size;
    let global_idx = group_idx * workgroup_len + local_invocation_index;
    
    // Radix-6 needs fft_len/6 threads (each thread processes 6 points)
    let threads_per_fft = fft_len / 6u;
    if global_idx >= arrayLength(&buffer_a) / fft_len * threads_per_fft {
        return;
    }
    
    // Determine which batch and butterfly this thread processes
    let batch_idx = global_idx / threads_per_fft;
    let batch_offset = batch_idx * fft_len;
    let local_idx = global_idx % threads_per_fft;

    if (stage == 0u) {
        // Stage 0: Combine bit reversal and first stage butterfly
        if (local_idx < threads_per_fft) {
            radix6_bit_reversal_and_butterfly(local_idx, fft_len, batch_offset);
        }
    } else {
        // Stage 1 and above: Standard radix-6 butterfly
        if (local_idx < threads_per_fft) {
            radix6_butterfly(local_idx, fft_len, batch_offset, stage);
        }
    }
}

// Base-6 bit reversal function
fn base6_bit_reverse(n: u32, log6_fft_len: u32) -> u32 {
    var reversed = 0u;
    var n_copy = n;
    
    for (var i = 0u; i < log6_fft_len; i++) {
        reversed = reversed * 6u + n_copy % 6u;
        n_copy = n_copy / 6u;
    }
    
    return reversed;
}

// Combine bit reversal and first stage radix-6 butterfly using Prime Factor Algorithm
fn radix6_bit_reversal_and_butterfly(idx: u32, n: u32, offset: u32) {
    // Compute log base 6 of n (rounded down)
    let log6_n = u32(log2(f32(n)) / log2(6.0)+0.5);
    
    // Each work item processes a 6-point butterfly
    let block_idx = idx;
    
    // Calculate the six input points' bit-reversed indices
    let z0_idx_br = base6_bit_reverse(block_idx * 6u + 0u, log6_n) + offset;
    let z1_idx_br = base6_bit_reverse(block_idx * 6u + 1u, log6_n) + offset;
    let z2_idx_br = base6_bit_reverse(block_idx * 6u + 2u, log6_n) + offset;
    let z3_idx_br = base6_bit_reverse(block_idx * 6u + 3u, log6_n) + offset;
    let z4_idx_br = base6_bit_reverse(block_idx * 6u + 4u, log6_n) + offset;
    let z5_idx_br = base6_bit_reverse(block_idx * 6u + 5u, log6_n) + offset;
    
    // Read bit-reversed values
    let z0 = buffer_a[z0_idx_br];
    let z1 = buffer_a[z1_idx_br];
    let z2 = buffer_a[z2_idx_br];
    let z3 = buffer_a[z3_idx_br];
    let z4 = buffer_a[z4_idx_br];
    let z5 = buffer_a[z5_idx_br];
    
    // Apply radix-6 butterfly formula according to the Prime Factor Algorithm
    // t1 = z2 + z4
    let t1 = z2 + z4;
    
    // t2 = z0 - t1/2
    let t1_half = vec2<f32>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    
    // t3 = sin(π/3)(z2 - z4)
    let z2_minus_z4 = z2 - z4;
    let t3 = vec2<f32>(
        SQRT3_DIV2 * z2_minus_z4.x,
        SQRT3_DIV2 * z2_minus_z4.y
    );
    
    // t4 = z5 + z1
    let t4 = z5 + z1;
    
    // t5 = z3 - t4/2
    let t4_half = vec2<f32>(t4.x * 0.5, t4.y * 0.5);
    let t5 = z3 - t4_half;
    
    // t6 = sin(π/3)(z5 - z1)
    let z5_minus_z1 = z5 - z1;
    let t6 = vec2<f32>(
        SQRT3_DIV2 * z5_minus_z1.x,
        SQRT3_DIV2 * z5_minus_z1.y
    );
    
    // t7 = z0 + t1
    let t7 = z0 + t1;
    
    // t8 = t2 + i*t3
    let t8 = vec2<f32>(t2.x - t3.y, t2.y + t3.x); // Complex multiplication by i
    
    // t9 = t2 - i*t3
    let t9 = vec2<f32>(t2.x + t3.y, t2.y - t3.x); // Complex multiplication by -i
    
    // t10 = z3 + t4
    let t10 = z3 + t4;
    
    // t11 = t5 + i*t6
    let t11 = vec2<f32>(t5.x - t6.y, t5.y + t6.x); // Complex multiplication by i
    
    // t12 = t5 - i*t6
    let t12 = vec2<f32>(t5.x + t6.y, t5.y - t6.x); // Complex multiplication by -i
    
    // Final outputs
    // x0 = t7 + t10
    let x0 = t7 + t10;
    
    // x4 = t8 + t11
    let x4 = t8 + t11;
    
    // x2 = t9 + t12
    let x2 = t9 + t12;
    
    // x3 = t7 - t10
    let x3 = t7 - t10;
    
    // x1 = t8 - t11
    let x1 = t8 - t11;
    
    // x5 = t9 - t12
    let x5 = t9 - t12;
    
    // Write to output in sequential order
    let out_idx = block_idx * 6u + offset;
    buffer_b[out_idx + 0u] = x0;
    buffer_b[out_idx + 1u] = x1;
    buffer_b[out_idx + 2u] = x2;
    buffer_b[out_idx + 3u] = x3;
    buffer_b[out_idx + 4u] = x4;
    buffer_b[out_idx + 5u] = x5;
}

// Standard radix-6 butterfly operation for stages > 0
fn radix6_butterfly(idx: u32, n: u32, offset: u32, stage: u32) {
    // Calculate current stage butterfly parameters
    // For radix-6, we have 6^stage
    let m = pow6(stage);
    let step = 6u * m;
    let k = idx % m;
    let block_idx = idx / m;
    
    // Calculate six points' indices
    let z0_idx = block_idx * step + k + offset;
    let z1_idx = z0_idx + m;
    let z2_idx = z0_idx + 2u * m;
    let z3_idx = z0_idx + 3u * m;
    let z4_idx = z0_idx + 4u * m;
    let z5_idx = z0_idx + 5u * m;
    
    // Read values from buffer
    let z0_raw = buffer_b[z0_idx];
    let z1_raw = buffer_b[z1_idx];
    let z2_raw = buffer_b[z2_idx];
    let z3_raw = buffer_b[z3_idx];
    let z4_raw = buffer_b[z4_idx];
    let z5_raw = buffer_b[z5_idx];
    
    // Calculate twiddle factor indices
    let twiddle_base = k * (n / (6u * m));
    let w1_idx = twiddle_base;
    let w2_idx = 2u * twiddle_base;
    let w3_idx = 3u * twiddle_base;
    let w4_idx = 4u * twiddle_base;
    let w5_idx = 5u * twiddle_base;
    
    // Get and apply twiddle factors
    let w1 = twiddles[w1_idx % arrayLength(&twiddles)];
    let w2 = twiddles[w2_idx % arrayLength(&twiddles)];
    let w3 = twiddles[w3_idx % arrayLength(&twiddles)];
    let w4 = twiddles[w4_idx % arrayLength(&twiddles)];
    let w5 = twiddles[w5_idx % arrayLength(&twiddles)];
    
    // Apply twiddle factors
    let z0 = z0_raw;
    let z1 = complex_mul(z1_raw, w1);
    let z2 = complex_mul(z2_raw, w2);
    let z3 = complex_mul(z3_raw, w3);
    let z4 = complex_mul(z4_raw, w4);
    let z5 = complex_mul(z5_raw, w5);
    
    // Now apply Prime Factor Algorithm for radix-6 butterfly
    // t1 = z2 + z4
    let t1 = z2 + z4;
    
    // t2 = z0 - t1/2
    let t1_half = vec2<f32>(t1.x * 0.5, t1.y * 0.5);
    let t2 = z0 - t1_half;
    
    // t3 = sin(π/3)(z2 - z4)
    let z2_minus_z4 = z2 - z4;
    let t3 = vec2<f32>(
        SQRT3_DIV2 * z2_minus_z4.x,
        SQRT3_DIV2 * z2_minus_z4.y
    );
    
    // t4 = z5 + z1
    let t4 = z5 + z1;
    
    // t5 = z3 - t4/2
    let t4_half = vec2<f32>(t4.x * 0.5, t4.y * 0.5);
    let t5 = z3 - t4_half;
    
    // t6 = sin(π/3)(z5 - z1)
    let z5_minus_z1 = z5 - z1;
    let t6 = vec2<f32>(
        SQRT3_DIV2 * z5_minus_z1.x,
        SQRT3_DIV2 * z5_minus_z1.y
    );
    
    // t7 = z0 + t1
    let t7 = z0 + t1;
    
    // t8 = t2 + i*t3
    let t8 = vec2<f32>(t2.x - t3.y, t2.y + t3.x); // Complex multiplication by i
    
    // t9 = t2 - i*t3
    let t9 = vec2<f32>(t2.x + t3.y, t2.y - t3.x); // Complex multiplication by -i
    
    // t10 = z3 + t4
    let t10 = z3 + t4;
    
    // t11 = t5 + i*t6
    let t11 = vec2<f32>(t5.x - t6.y, t5.y + t6.x); // Complex multiplication by i
    
    // t12 = t5 - i*t6
    let t12 = vec2<f32>(t5.x + t6.y, t5.y - t6.x); // Complex multiplication by -i
    
    // Final outputs
    // x0 = t7 + t10
    let x0 = t7 + t10;
    
    // x4 = t8 + t11
    let x4 = t8 + t11;
    
    // x2 = t9 + t12
    let x2 = t9 + t12;
    
    // x3 = t7 - t10
    let x3 = t7 - t10;
    
    // x1 = t8 - t11
    let x1 = t8 - t11;
    
    // x5 = t9 - t12
    let x5 = t9 - t12;
    
    // Write back results
    buffer_b[z0_idx] = x0;
    buffer_b[z1_idx] = x1;
    buffer_b[z2_idx] = x2;
    buffer_b[z3_idx] = x3;
    buffer_b[z4_idx] = x4;
    buffer_b[z5_idx] = x5;
}

// Helper to calculate 6^x
fn pow6(x: u32) -> u32 {
    var result = 1u;
    for (var i = 0u; i < x; i++) {
        result *= 6u;
    }
    return result;
}

// Complex multiplication helper
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}