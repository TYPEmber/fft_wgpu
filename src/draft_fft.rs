use crate::typed_buffer;
use num_complex::Complex;
use wgpu::{
    BindGroup, BufferUsages, CommandEncoderDescriptor, ComputePipeline, Device, Queue,
    naga::FastHashMap,
    util::{self, DeviceExt},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Recipe {
    Radix2,
    Radix3,
    Radix4,
    Radix5,
    Radix6,
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Forward,
    Inverse,
}

pub struct Config {
    pub fft_len: u32,
    pub recipe: Recipe,
    pub direction: Direction,
}

pub struct Processor<'a> {
    config: Config,
    twiddles: typed_buffer::Array<Complex<f32>>,
    device: &'a Device,
    queue: &'a Queue,
    pipeline: ComputePipeline,
    bind_group_cache: FastHashMap<[usize; 2], BindGroup>,
}

impl<'a> Processor<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue, config: Config) -> Result<Self, FftError> {
        let pipeline = create_pipeline(device, &config.recipe, &config.direction)?;
        let twiddles = typed_buffer::Array::new(
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Twiddle Buffer"),
                contents: bytemuck::cast_slice(
                    &(0..config.fft_len)
                        .map(|k| -2.0 * std::f64::consts::PI * (k as f64) / (config.fft_len as f64))
                        .map(|theta| Complex::new(theta.cos() as f32, theta.sin() as f32))
                        .collect::<Vec<_>>(),
                ),
                usage: BufferUsages::STORAGE,
            }),
        );

        Ok(Self {
            config,
            twiddles,
            device,
            queue,
            pipeline,
            bind_group_cache: Default::default(),
        })
    }

    pub fn proc(
        &mut self,
        input: &typed_buffer::Array<Complex<f32>>,
        output: &typed_buffer::Array<Complex<f32>>,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("FFT Command Encoder"),
            });

        if self.bind_group_cache.len() > 1024 {
            self.bind_group_cache.clear();
        }
        let bind_group: &BindGroup = self
            .bind_group_cache
            .entry([input as *const _ as usize, output as *const _ as usize])
            // Do not use `or_insert`
            // It will new default value everytime.
            .or_insert_with(|| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FFT Bind Group"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.twiddles.inner.as_entire_binding(),
                        },
                    ],
                })
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Compute Pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            // 计算工作组数量
            let workgroup_size = 64;
            let base = match self.config.recipe {
                Recipe::Radix2 => 2,
                Recipe::Radix3 => 3,
                Recipe::Radix4 => 4,
                Recipe::Radix5 => 5,
                Recipe::Radix6 => 6,
            };

            let workgroup_count_per_fft = (self.config.fft_len / base / workgroup_size).max(1);

            // 计算数据批次数量
            let batch_count = (input.size() / self.config.fft_len as u64) as u32;
            let num_stages = self.config.fft_len.ilog(base);

            cpass.set_push_constants(0, &self.config.fft_len.to_le_bytes());
            // 设置FFT长度和当前阶段
            for stage in 0..num_stages {
                cpass.set_push_constants(4, &stage.to_le_bytes());
                cpass.dispatch_workgroups(workgroup_count_per_fft, batch_count, 1);
            }
        }

        self.queue.submit([encoder.finish()]);
    }
}

#[derive(Debug)]
pub enum FftError {
    InvalidRadix,
    InvalidLength,
    InvalidUsage,
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidRadix => write!(f, "Invalid FFT radix specified"),
            Self::InvalidLength => write!(f, "FFT length incompatible with selected radix"),
            Self::InvalidUsage => write!(f, "Buffer missing required usage flags"),
        }
    }
}

impl std::error::Error for FftError {}

fn create_pipeline(
    device: &Device,
    recipe: &Recipe,
    direction: &Direction,
) -> Result<ComputePipeline, FftError> {
    // 根据基底和方向选择WGSL文件
    let shader_source = match (recipe, direction) {
        (Recipe::Radix2, Direction::Forward) => include_str!("kernel/fftct.wgsl"),
        (Recipe::Radix2, Direction::Inverse) => include_str!("kernel/ifftct.wgsl"),
        (Recipe::Radix3, Direction::Forward) => include_str!("kernel/fft3ct.wgsl"),
        (Recipe::Radix3, Direction::Inverse) => include_str!("kernel/ifft3ct.wgsl"),
        (Recipe::Radix4, Direction::Forward) => include_str!("kernel/fft4ct.wgsl"),
        (Recipe::Radix4, Direction::Inverse) => include_str!("kernel/ifft4ct.wgsl"),
        (Recipe::Radix5, Direction::Forward) => include_str!("kernel/fft5ct.wgsl"),
        (Recipe::Radix5, Direction::Inverse) => include_str!("kernel/ifft5ct.wgsl"),
        (Recipe::Radix6, Direction::Forward) => include_str!("kernel/fft6ct.wgsl"),
        (Recipe::Radix6, Direction::Inverse) => include_str!("kernel/ifft6ct.wgsl"),
    };

    // 创建着色器模块
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FFT Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });

    // 创建绑定组布局
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FFT Bind Group Layout"),
        entries: &[
            // 输入缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 输出缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 旋转因子缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // 创建管道布局
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FFT Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8, // 用于传递fft_len和stage
        }],
    });

    // 创建计算管线
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Compute Pipeline"),
            layout: Some(&ppl),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        }),
    )
}

fn create_multiply_pipeline(device: &Device) -> Result<ComputePipeline, FftError> {
    // 选择 WGSL 文件
    let shader_source = include_str!("kernel/multiply.wgsl");

    // 创建着色器模块
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Multiply Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });

    // 创建绑定组布局
    let bgl: wgpu::BindGroupLayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Multiply Bind Group Layout"),
        entries: &[
            // 输入缓冲区 A
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,  // 可在计算阶段被访问
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 输入缓冲区 B
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 输出缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // 创建管道布局
    let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Multiply Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // 创建计算管线
    Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Multiply Pipeline"),
        layout: Some(&ppl),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    }))
}

pub struct MultiplyProcessor<'a> {
    device: &'a Device,
    queue: &'a Queue,
    pipeline: ComputePipeline,
    bind_group_cache: FastHashMap<[usize; 3], BindGroup>,
}

impl<'a> MultiplyProcessor<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue) -> Result<Self, FftError> {
        let pipeline = create_multiply_pipeline(device)?;

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_cache: Default::default(),
        })
    }

    pub fn proc(
        &mut self,
        input_a: &typed_buffer::Array<Complex<f32>>,
        input_b: &typed_buffer::Array<Complex<f32>>,
        output: &typed_buffer::Array<Complex<f32>>,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Multiply Encoder"),
            });

        if self.bind_group_cache.len() > 1024 {
            self.bind_group_cache.clear();
        }

        let bind_group: &BindGroup = self
            .bind_group_cache
            .entry([
                input_a as *const _ as usize,
                input_b as *const _ as usize,
                output as *const _ as usize,
            ])
            .or_insert_with(|| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Multiply Bind Group"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_a.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_b.inner.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output.inner.as_entire_binding(),
                        },
                    ],
                })
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Multiply Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            let workgroup_size = 64;
            let num_workgroups = (input_a.size() as u32 + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.queue.submit([encoder.finish()]);
    }
}


#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;
    use num_complex::Complex;
    use wgpu::{ExperimentalFeatures, hal::MemoryRange};
    use crate::typed_buffer;
    use super::{Processor, Config, Recipe, Direction};

   #[tokio::test]
    async fn test_f32_batch_truncation_bug_with_print() {
        println!("\n🚀 Starting FFT Batch Truncation Test (f32)...\n");

        // 1. 初始化 GPU 环境
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to find adapter");

        let mut required_features = adapter.features();
        // 如果你的 f32 shader 需要某些特性，可以在这里加，通常 f32 是核心功能不需要额外 feature
        
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits: adapter.limits(),
                experimental_features: unsafe { ExperimentalFeatures::enabled() },
                label: Some("GPU Device"),
                ..Default::default()
            })
            .await
            .unwrap();

        // 2. 准备测试参数
        let fft_len = 16u32; // FFT 点数
        let num_batches = 2; // Batch 数量 (测试多 Batch 能力)
        let total_elements = (fft_len * num_batches) as usize;

        // 3. 准备输入数据 (f32)
        // 构造简单的直流信号：
        // Batch 0: 全是 1.0 (Sum = 16.0)
        // Batch 1: 全是 2.0 (Sum = 32.0)
        let mut input_data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            let val = if i < fft_len as usize { 1.0 } else { 2.0 };
            input_data.push(Complex::new(val, 0.0f32)); // 显式 f32
        }

        // === 打印输入数据 ===
        println!("📝 --- Input Data (Before FFT) ---");
        for b in 0..num_batches {
            let start = (b * fft_len) as usize;
            let end = start + fft_len as usize;
            println!("Batch {}: {:?}", b, &input_data[start..end]);
        }
        println!("----------------------------------\n");

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let input_arr = typed_buffer::Array::new(input_buffer);

        // 4. 准备输出数据
        let output_size = (total_elements * std::mem::size_of::<Complex<f32>>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_arr = typed_buffer::Array::new(output_buffer);

        // 5. 创建 FFT Processor (f32)
        let config = Config {
            fft_len,
            recipe: Recipe::Radix2,
            direction: Direction::Forward,
        };
        // 注意：这里假设 Processor::new 内部使用的是 f32 的 shader
        let mut processor = Processor::new(&device, &queue, config).unwrap();

        // 6. 执行 FFT
        println!("⚙️  Running FFT on GPU...");
        processor.proc(&input_arr, &output_arr);

        // 7. 读取结果
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&output_arr.inner, 0, &readback_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        // 等待 GPU 完成
        // device.poll(wgpu::PollType::Wait).unwrap();
        while !device.poll(wgpu::PollType::Poll).unwrap().is_queue_empty() {}
        receiver.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: &[Complex<f32>] = bytemuck::cast_slice(&data);

        // === 打印输出数据 ===
        println!("\n📊 --- Output Data (After FFT) ---");
        for b in 0..num_batches as usize {
            let start = b * fft_len as usize;
            let end = start + fft_len as usize;
            println!("Batch {}:", b);
            // 为了显示整洁，只打印前几个和非零值，或者打印全部但紧凑些
            let batch_slice = &result[start..end];
            
            // 打印直流分量 (DC - Index 0)
            println!("  - DC Component (Index 0): re={:.4}, im={:.4}", batch_slice[0].re, batch_slice[0].im);
            
            // 打印全部数据（可选，如果太长可以注释掉）
            print!("  - All Data: [");
            for (i, val) in batch_slice.iter().enumerate() {
                if i > 0 { print!(", "); }
                // 简单的复数格式化
                print!("{:.1}+{:.1}i", val.re, val.im);
            }
            println!("]");
        }
        println!("----------------------------------\n");

        // 8. 验证逻辑
        println!("🔍 Verifying Results...");
        
        // 验证 Batch 0
        let b0_dc = result[0].re;
        let expected_b0 = 16.0;
        let b0_ok = (b0_dc - expected_b0).abs() < 0.1;
        
        if b0_ok {
            println!("✅ Batch 0 DC: {:.4} (Expected {:.4}) - PASS", b0_dc, expected_b0);
        } else {
            println!("❌ Batch 0 DC: {:.4} (Expected {:.4}) - FAIL", b0_dc, expected_b0);
        }

        // 验证 Batch 1 (Bug 核心)
        let b1_idx = fft_len as usize;
        let b1_dc = result[b1_idx].re;
        let expected_b1 = 32.0; // sum(2.0 * 16)
        
        // 特别检查是否为 0
        if b1_dc.abs() < 0.001 {
            println!("\n🚨 BUG DETECTED: Batch 1 DC is ZERO! (Expected {:.4})", expected_b1);
            println!("   -> The shader processed Batch 0 correctly but IGNORED Batch 1.");
            println!("   -> This confirms the 'Half-Truncation' bug in f32 shader logic.");
            panic!("Test Failed: Batch 1 Truncation Detected");
        } else if (b1_dc - expected_b1).abs() < 0.1 {
            println!("✅ Batch 1 DC: {:.4} (Expected {:.4}) - PASS", b1_dc, expected_b1);
            println!("\n🎉 SUCCESS: The f32 shader handles multiple batches correctly!");
        } else {
            println!("❓ Batch 1 DC: {:.4} (Expected {:.4}) - WRONG VALUE", b1_dc, expected_b1);
            panic!("Test Failed: Calculation Error");
        }
    }
}