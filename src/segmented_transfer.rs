use futures::channel::oneshot;
use std::sync::Arc;
use std::sync::mpsc;

/// 表示分段传输的配置和状态
pub struct SegmentedTransfer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // 上传相关资源
    upload_buffers: Vec<wgpu::Buffer>,
    upload_sizes: Vec<u64>,
    upload_elements: Vec<usize>,
    upload_segments: usize,

    // 下载相关资源
    download_buffers: Vec<wgpu::Buffer>,
    download_sizes: Vec<u64>,
    download_elements: Vec<usize>,
    download_segments: usize,

    // 总数据大小
    total_size: u64,
    element_size: u64,
}

impl SegmentedTransfer {
    /// 创建新的分段传输实例，允许上传和下载使用不同数量的段
    pub fn new<T, N>(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        total_elements: usize,
        upload_values: &[N],
        download_values: Option<&[N]>,
    ) -> Self
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
        N: Into<f64> + Copy,
    {
        let element_size = std::mem::size_of::<T>() as u64;
        let total_size = (total_elements as u64) * element_size;

        // 转换上传比例并归一化
        let upload_fractions = normalize_values(upload_values);
        println!(
            "Normalized upload fractions: {:?}",
            upload_fractions
        );
        let upload_segments = upload_values.len();

        // 处理下载比例 - 允许不同的段数
        let download_fractions = match download_values {
            Some(values) => {
                // 不再要求下载段数与上传段数相同
                normalize_values(values)
            }
            None => {
                // 默认使用上传顺序相反的比例
                let mut reversed = upload_fractions.clone();
                reversed.reverse();
                reversed
            }
        };
        let download_segments = download_fractions.len();

        // 计算上传段大小和元素数
        let mut upload_sizes = Vec::with_capacity(upload_segments);
        let mut upload_elements = Vec::with_capacity(upload_segments);

        for &fraction in &upload_fractions {
            let elements = ((total_elements as f64) * fraction) as usize;
            let size = (elements as u64) * element_size;
            upload_sizes.push(size);
            upload_elements.push(elements);
        }

        // 确保最后一段包含剩余所有元素
        let sum_elements: usize = upload_elements.iter().sum();
        if sum_elements < total_elements {
            let diff = total_elements - sum_elements;
            *upload_elements.last_mut().unwrap() += diff;
            let size_diff = (diff as u64) * element_size;
            *upload_sizes.last_mut().unwrap() += size_diff;
        }

        // 计算下载段大小和元素数
        let mut download_sizes = Vec::with_capacity(download_segments);
        let mut download_elements = Vec::with_capacity(download_segments);

        for &fraction in &download_fractions {
            let elements = ((total_elements as f64) * fraction) as usize;
            let size = (elements as u64) * element_size;
            download_sizes.push(size);
            download_elements.push(elements);
        }

        // 确保下载段也处理舍入误差
        let sum_elements: usize = download_elements.iter().sum();
        if sum_elements < total_elements {
            let diff = total_elements - sum_elements;
            *download_elements.last_mut().unwrap() += diff;
            let size_diff = (diff as u64) * element_size;
            *download_sizes.last_mut().unwrap() += size_diff;
        }

        // 创建上传缓冲区
        let mut upload_buffers = Vec::with_capacity(upload_segments);
        for (i, &size) in upload_sizes.iter().enumerate() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Upload Buffer {}", i + 1)),
                size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            upload_buffers.push(buffer);
        }

        // 创建下载缓冲区
        let mut download_buffers = Vec::with_capacity(download_segments);
        for (i, &size) in download_sizes.iter().enumerate() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Download Buffer {}", i + 1)),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            download_buffers.push(buffer);
        }

        Self {
            device,
            queue,
            upload_buffers,
            upload_sizes,
            upload_elements,
            upload_segments,
            download_buffers,
            download_sizes,
            download_elements,
            download_segments,
            total_size,
            element_size,
        }
    }

    /// 上传数据到GPU
    pub async fn upload_data<T>(&self, data: &[T], gpu_buffer: &wgpu::Buffer)
    where
        T: Copy + bytemuck::Pod + bytemuck::Zeroable,
    {
        let mut offset = 0;
        let mut gpu_offset = 0;

        for i in 0..self.upload_segments {
            if i == 0 {
                // 第一段：直接映射
                self.upload_buffers[0]
                    .slice(..)
                    .map_async(wgpu::MapMode::Write, |r| r.unwrap());
                while !self.device.poll(wgpu::MaintainBase::Poll).is_queue_empty() {}

                // 写入数据并提交命令
                {
                    let mut view = self.upload_buffers[0].slice(..).get_mapped_range_mut();
                    bytemuck::cast_slice_mut::<u8, T>(&mut view)
                        .copy_from_slice(&data[offset..offset + self.upload_elements[0]]);
                    drop(view);
                    self.upload_buffers[0].unmap();

                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some(&format!("Upload Encoder {}", i + 1)),
                            });
                    encoder.copy_buffer_to_buffer(
                        &self.upload_buffers[0],
                        0,
                        gpu_buffer,
                        gpu_offset,
                        self.upload_sizes[0],
                    );
                    self.queue.submit(Some(encoder.finish()));
                }

                offset += self.upload_elements[0];
                gpu_offset += self.upload_sizes[0];
            } else {
                // 其他段：使用异步通道
                let (tx, rx) = oneshot::channel();
                self.upload_buffers[i]
                    .slice(..)
                    .map_async(wgpu::MapMode::Write, move |result| {
                        let _ = tx.send(result);
                    });

                self.device.poll(wgpu::MaintainBase::Poll);
                let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("Upload Encoder {}", i + 1)),
                    });
                if let Ok(Ok(_)) = rx.await {
                    let mut view = self.upload_buffers[i].slice(..).get_mapped_range_mut();
                    bytemuck::cast_slice_mut::<u8, T>(&mut view)
                        .copy_from_slice(&data[offset..offset + self.upload_elements[i]]);
                    drop(view);
                    self.upload_buffers[i].unmap();

                   
                    encoder.copy_buffer_to_buffer(
                        &self.upload_buffers[i],
                        0,
                        gpu_buffer,
                        gpu_offset,
                        self.upload_sizes[i],
                    );
                    self.queue.submit(Some(encoder.finish()));

                    offset += self.upload_elements[i];
                    gpu_offset += self.upload_sizes[i];
                }
            }
        }
    }

    /// 从GPU下载数据
    pub async fn download_data<T>(
        &self,
        gpu_buffer: &wgpu::Buffer,
        output_vec: Option<&mut Vec<T>>,
    ) -> Vec<T>
    where
        T: Copy + bytemuck::Pod + bytemuck::Zeroable,
    {
        // 确定输出目标
        let total_elements = (self.total_size / self.element_size) as usize;
        let mut owned_output = Vec::new();
        let output_slice = match output_vec {
            Some(vec_slice) => {
                assert!(vec_slice.len() >= total_elements, "提供的输出向量太小");
                vec_slice
            }
            None => {
                owned_output = vec![T::zeroed(); total_elements];
                &mut owned_output
            }
        };

        // 下载阶段：多段并行下载
        let mut result_offset = 0;
        let mut gpu_offset = 0;

        if self.download_segments > 0 {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Download Encoder 1"),
                });
            encoder.copy_buffer_to_buffer(
                gpu_buffer,
                0,
                &self.download_buffers[0],
                0,
                self.download_sizes[0],
            );
            self.queue.submit(Some(encoder.finish()));
        }

        // 处理所有段
        for i in 0..self.download_segments {
            // 如果不是最后一段，提前开始下一段的操作
            // 等待并处理当前段
            let (tx, rx) = mpsc::channel();
            self.download_buffers[i]
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).unwrap();
                });

            while !self.device.poll(wgpu::MaintainBase::Poll).is_queue_empty() {}
            if i < self.download_segments - 1 {
                let next_gpu_offset = gpu_offset + self.download_sizes[i];
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(&format!("Download Encoder {}", i + 2)),
                        });
                encoder.copy_buffer_to_buffer(
                    gpu_buffer,
                    next_gpu_offset,
                    &self.download_buffers[i + 1],
                    0,
                    self.download_sizes[i + 1],
                );
                self.queue.submit(Some(encoder.finish()));
            }

            if rx.recv().unwrap().is_ok() {
                let view = self.download_buffers[i].slice(..).get_mapped_range();
                output_slice[result_offset..result_offset + self.download_elements[i]]
                    .copy_from_slice(&bytemuck::cast_slice(&view)[..self.download_elements[i]]);
                drop(view);
                self.download_buffers[i].unmap();

                result_offset += self.download_elements[i];
                gpu_offset += self.download_sizes[i];
            }
        }
        owned_output
    }
}

fn normalize_values<N>(values: &[N]) -> Vec<f64>
where
    N: Into<f64> + Copy,
{
    // 计算所有值的总和
    let sum: f64 = values.iter().map(|&v| v.into()).sum();
    assert!(sum > 0.0, "比例值总和必须大于0");

    // 归一化每个值
    values.iter().map(|&v| v.into() / sum).collect()
}
