use crate::common::config::{CacheConfig, ModelConfig, ParallelConfig};

use anyhow::Result;
use candle_core::cuda_backend::cudarc::driver::sys::CUevent;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::CudaDevice;
use candle_core::{Device, Tensor};
use std::sync::Arc;

use std::collections::HashMap;

pub type KVCache = (Tensor, Tensor);
pub type KeyBlockShape = (usize, usize, usize, usize);
pub type ValueBlockShape = (usize, usize, usize);
pub(super) struct CacheEngine {
    cuda_device: Device,
    cache_config: CacheConfig,
    model_config: Arc<ModelConfig>,
    parallel_config: ParallelConfig,
    // cache_ops: CacheOps,
    num_heads: usize,
    num_layers: usize,
    gpu_cache: Vec<KVCache>,
    cpu_cache: Vec<KVCache>,
    cache_stream: CudaStream,
    events: Vec<CUevent>,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
}

impl CacheEngine {
    pub fn new(
        cuda_device: &CudaDevice,
        cache_config: &CacheConfig,
        model_config: Arc<ModelConfig>,
        parallel_config: &ParallelConfig,
        // cache_ops: CacheOps,
    ) -> Result<Self> {
        let num_heads = model_config.get_num_kv_heads(parallel_config);
        let num_layers = model_config.get_num_layers(parallel_config);

        let mut events = Vec::new();
        let cache_stream = cuda_device.fork_default_stream()?;
        for _i in 0..num_layers {
            let mut event: CUevent = std::ptr::null_mut();
            unsafe {
                candle_core::cuda_backend::cudarc::driver::sys::cuEventCreate(&mut event, 0);
            }
            events.push(event);
        }

        let cache = Self {
            cuda_device: Device::Cuda(cuda_device.clone()),
            cache_config: cache_config.clone(),
            model_config,
            // model_config: model_config.clone(),
            parallel_config: parallel_config.clone(),
            // cache_ops,
            num_heads,
            num_layers,
            gpu_cache: Vec::new(),
            cpu_cache: Vec::new(),
            cache_stream,
            events,
            num_gpu_blocks: 0,
            num_cpu_blocks: 0,
        };
        Ok(cache)
    }

    pub fn get_num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }
    pub fn get_num_cpu_blocks(&self) -> usize {
        self.num_cpu_blocks
    }

    pub fn init(&mut self, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Result<()> {
        self.num_cpu_blocks = num_cpu_blocks;
        self.num_gpu_blocks = num_gpu_blocks;
        self.cpu_cache = self.allocate_cpu_cache()?;

        self.gpu_cache = self.allocate_gpu_cache()?;
        Ok(())
    }

    pub fn get_cache_block_size(&self, block_size: usize) -> usize {
        let head_size = self.model_config.head_size();
        let num_heads = self.model_config.get_num_kv_heads(&self.parallel_config);
        let num_layers = self.model_config.get_num_layers(&self.parallel_config);

        let key_cache_block = block_size * num_heads * head_size;
        let value_cache_block = key_cache_block;
        let total = num_layers * (key_cache_block + value_cache_block);
        tracing::info!(
            "head_size:{},num_heads:{},num_layers:{},total:{}",
            head_size,
            num_heads,
            num_layers,
            total,
        );
        let dtype_size = self.model_config.get_dtype().size_in_bytes();
        dtype_size * total
    }

    fn get_key_block_shape(&self) -> KeyBlockShape {
        let element_size = self.model_config.get_dtype().size_in_bytes();
        let x = 16 / element_size;

        (
            self.num_heads,
            self.model_config.head_size() / x,
            self.cache_config.get_block_size(),
            x,
        )
    }
    fn get_value_block_shape(&self) -> ValueBlockShape {
        (
            self.num_heads,
            self.model_config.head_size(),
            self.cache_config.get_block_size(),
        )
    }

    fn allocate_gpu_cache(&self) -> Result<Vec<KVCache>> {
        let key_block_shape = self.get_key_block_shape();
        let value_block_shape = self.get_value_block_shape();

        let mut gpu_cache = Vec::new();
        for _ in 0..self.model_config.num_hidden_layers() {
            let key_blocks = Tensor::zeros(
                (
                    self.get_num_gpu_blocks(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                self.model_config.get_dtype(),
                &self.cuda_device,
            )?;

            let value_blocks = Tensor::zeros(
                (
                    self.get_num_gpu_blocks(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                self.model_config.get_dtype(),
                &self.cuda_device,
            )?;

            let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
            tracing::info!("After init_cache, GPU free:{} bytes, total:{}", free, total);
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn allocate_cpu_cache(&self) -> Result<Vec<KVCache>> {
        let key_block_shape = self.get_key_block_shape();
        let value_block_shape = self.get_value_block_shape();
        let mut cpu_cache = Vec::new();
        for _ in 0..self.model_config.num_hidden_layers() {
            let device = Device::Cpu;
            let key_blocks = Tensor::zeros(
                (
                    self.get_num_cpu_blocks(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                self.model_config.get_dtype(),
                &device,
            )?;
            let value_blocks = Tensor::zeros(
                (
                    self.get_num_cpu_blocks(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                self.model_config.get_dtype(),
                &device,
            )?;

            cpu_cache.push((key_blocks, value_blocks));
        }
        Ok(cpu_cache)
    }

    pub fn get_gpu_cache(&self) -> &Vec<(Tensor, Tensor)> {
        &self.gpu_cache
    }

    // fn get_mut_gpu_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
    //     loop {
    //         if let Ok(v) = self.gpu_cache.try_lock() {
    //             return v;
    //         }
    //     }
    // }
    pub fn swap_in(&self, block_mapping: &HashMap<u32, u32>) -> Result<()> {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
            // let mut gpu_cache = self.get_mut_gpu_cache();
            // let (dst_key_cache, dst_value_cache) = gpu_cache.get_mut(i).unwrap();
            // Self::swap_blocks(
            //     src_key_cache.clone(),
            //     dst_key_cache,
            //     block_mapping,
            //     &self.cache_stream,
            // )?;
            // Self::swap_blocks(
            //     src_value_cache.clone(),
            //     dst_value_cache,
            //     block_mapping,
            //     &self.cache_stream,
            // )?;
            unsafe {
                candle_core::cuda_backend::cudarc::driver::sys::cuEventRecord(
                    self.events[i],
                    self.cache_stream.stream,
                );
            }
        }
        todo!("swap_in");
        Ok(())
    }
    pub fn swap_out(&self, block_mapping: &HashMap<u32, u32>) -> Result<()> {
        for i in 0..self.num_layers {
            // let gpu_cache = self.get_mut_gpu_cache();
            // let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
            // drop(gpu_cache);

            // let (dst_key_cache, dst_value_cache) = self.cpu_cache.get_mut(i).unwrap();
            // Self::swap_blocks(
            //     src_key_cache.clone(),
            //     dst_key_cache,
            //     block_mapping,
            //     &self.cache_stream,
            // )?;
            // Self::swap_blocks(
            //     src_value_cache.clone(),
            //     dst_value_cache,
            //     block_mapping,
            //     &self.cache_stream,
            // )?;
            unsafe {
                candle_core::cuda_backend::cudarc::driver::sys::cuEventRecord(
                    self.events[i],
                    self.cache_stream.stream,
                );
            }
        }
        todo!("swap_out");
        Ok(())
    }
    pub fn copy(&self, block_mapping: &HashMap<u32, Vec<u32>>) -> Result<()> {
        // let mut gpu_cache = self.get_mut_gpu_cache();
        // let caches: (Vec<&mut Tensor>, Vec<&mut Tensor>) =
        //     gpu_cache.iter_mut().map(|(a, b)| (a, b)).unzip();
        // let (key_caches, value_caches) = caches;

        // for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        //     key_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
        //     value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
        //   }
        todo!("copy");
        Ok(())
    }
}
