@group(0) @binding(0)
var<storage, read_write> input : array<vec2 < f32>>;
@group(0) @binding(1)
var<storage, read_write> output : array<vec2 < f32>>;
@group(0) @binding(2) //新增Twiddle因子缓冲区
var<storage, read> twiddles : array<vec2 < f32>>;

const PI : f32 = 3.14159265358979323846;
const workgroup_len : u32 = 256u;
struct PushConstants { fft_len : u32, stage : u32 }
    var<push_constant> consts : PushConstants;

@compute @workgroup_size(workgroup_len)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>

) {
    let workgroup_index=workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.y * num_workgroups.x;
    let offset=workgroup_index*consts.fft_len;
    fft(consts.fft_len,local_index,offset);
    }
    fn fft(fft_len:u32,local_index:u32,offset:u32)
    {
        let n = fft_len;
        //let tid = global_id.x;
        let local_tid = local_index;
        //let workgroup_size = 512u;

        // 初始化共享内存（带边界保护）
        
        workgroupBarrier();
    // 2. Stockham阶段处理
    for (var m = 1u; m < n; m *= 2u) {
        let stage = u32(log2(f32(m)));
        let J = m;    //半块大小
        let block_size = 2u * J;
        let num_blocks = n / block_size;    //块数
        let total_pairs = num_blocks * J;


        let active_threads = min(workgroup_len, total_pairs);   // 结合对数和工作组大小来确定活跃线程数
        let pairs_per_thread = (total_pairs + active_threads - 1u) / active_threads;//计算每个线程需要处理的工作组数
        // 任务分配（保证线程安全）
        //let pairs_per_thread = (total_pairs + workgroup_size - 1u) / workgroup_size;
        
        for (var p = 0u; p < pairs_per_thread; p++) {
            let pair_idx = local_tid * pairs_per_thread + p;
            if (pair_idx >= total_pairs) { break; }

            let s = pair_idx / J;  // 块索引
            let j = pair_idx % J;   // 块内索引

           // ==== 关键修改1：输入位置计算 ====
           // let base = s * block_size;
            //let pos1_in = base + j;
            //let pos2_in = base + j + J;
            //在 Stockham 阶段处理中
            let  var1 = s * J; // var = s * J
            let pos1_in = offset+var1 + j; // 对应 Python 中的 x[var + j]
            let pos2_in = offset+var1 + j + (n / 2u); // 对应 Python 中的 x[var + j + n/2]
            //==== 关键修改2：边界检查优化 ====
           // if (pos2_in >= n || pos1_in >= n) { continue; }


            let output_base =offset+s * block_size;
            let pos1_out = output_base + j;
            let pos2_out = output_base + j + J;

           // ==== 关键修改3：正确旋转因子计算 ====
              //let theta = -2.0 * PI * f32(J) * f32(s) / f32(n);

          // let theta = -PI * f32(j) / f32(J); // Stockham专用公式
           // let twiddle = vec2<f32>(cos(theta), sin(theta));
            let twiddle = twiddles[s*J];  
           // 蝶形运算
            if stage % 2u == 0u {
                let a = input[pos1_in];
            //let b = complex_mul(shared_in[pos2_in], twiddle);
                let b=input[pos2_in];
                 output[pos1_out] = a + b;
                output[pos2_out] = complex_mul((a - b),twiddle);
            }
         else {
                 let a = output[pos1_in];
            //let b = complex_mul(shared_in[pos2_in], twiddle);
                let b=output[pos2_in];
                 input[pos1_out] = a + b;
                input[pos2_out] = complex_mul((a - b),twiddle);
                }
        workgroupBarrier();
        }
        // 双缓冲交换（全范围覆盖）
        //for (var i = local_tid; i < n; i += workgroup_size) {
           // input[offset+i] = output[offset+i];
       // }
     //  workgroupBarrier();
   // }
    }
    // 3. 输出结果（带保护）
   // for (var i = local_tid; i < n; i += workgroup_size) {
       // output[offset+i] = input[offset+i];
    //}
    //for (var i = 0u; i < arrayLength(&input); i = i + 1u) {
        //output[i] = input[i];
    //}
   // if offset==256{
       // output[offset]=vec2<f32>(1.0,0.0);
        //}
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}