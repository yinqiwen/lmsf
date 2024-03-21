use std::{collections::HashMap, ops::Deref};

use anyhow::{anyhow, Result};
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::Storage;
use candle::{
    cuda_backend::cudarc::driver::{
        result::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async},
        CudaFunction, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig,
    },
    CudaDevice, DType, Device, Tensor,
};

use crate::{dispatch_cuda_func, get_cuda_slice};

use super::get_tensor_cuda_slice_ptr_int;

pub struct CacheOps {
    copy_blocks: Vec<CudaFunction>,
    reshape_and_cache: Vec<CudaFunction>,
}

impl CacheOps {
    pub fn new(device: &CudaDevice) -> candle::Result<Self> {
        Ok(Self {
            copy_blocks: vec![
                device.get_or_load_func("copy_blocks_f16", kernels::CACHE_KERNELS)?,
                device.get_or_load_func("copy_blocks_f32", kernels::CACHE_KERNELS)?,
            ],
            reshape_and_cache: vec![
                device.get_or_load_func("reshape_and_cache_f16", kernels::CACHE_KERNELS)?,
                device.get_or_load_func("reshape_and_cache_f32", kernels::CACHE_KERNELS)?,
            ],
        })
    }

    fn swap_blocks(
        src: Tensor,
        dst: &mut Tensor,
        block_mapping: &HashMap<usize, usize>,
        stream: &CudaStream,
    ) -> Result<()> {
        let block_size_in_bytes = src.dtype().size_in_bytes() * src.dims()[0];
        match (src.device(), dst.device()) {
            (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
                if src_dev.ordinal() != dst_dev.ordinal() {
                    return Err(anyhow!("Tensors must be on the same device to copy, got ordinals {} (src) and {} (dst).", src_dev.ordinal(), dst_dev.ordinal()));
                }
                let (src_storage, _) = src.storage_and_layout();
                let (dst_storage, _) = dst.storage_and_layout();

                match (src_storage.deref(), dst_storage.deref()) {
                    (Storage::Cuda(src_storage), Storage::Cuda(dst_storage)) => {
                        let src_ptr = src_storage.as_cuda_slice::<u8>()?.device_ptr();
                        let dst_ptr = dst_storage.as_cuda_slice::<u8>()?.device_ptr();
                        // let stream = ManuallyDrop::new(src_dev.cu_stream());
                        for (src_block_number, dst_block_number) in block_mapping {
                            let src_offset = src_block_number * block_size_in_bytes;
                            let dst_offset = dst_block_number * block_size_in_bytes;
                            unsafe {
                                memcpy_dtod_async(
                                    dst_ptr + dst_offset as u64,
                                    src_ptr + src_offset as u64,
                                    block_size_in_bytes,
                                    stream.stream,
                                )
                            }?
                        }
                    }
                    _ => {
                        //todo
                    }
                }
            }
            (Device::Cpu, Device::Cuda(dst_dev)) => {
                let (src_storage, _) = src.storage_and_layout();
                let (dst_storage, _) = dst.storage_and_layout();
                match (src_storage.deref(), dst_storage.deref()) {
                    (Storage::Cpu(src_storage), Storage::Cuda(dst_storage)) => {
                        let src_slice = src_storage.as_slice::<u8>()?;
                        let dst_ptr = dst_storage.as_cuda_slice::<u8>()?.device_ptr();

                        for (src_block_number, dst_block_number) in block_mapping {
                            let dst_offset = dst_block_number * block_size_in_bytes;
                            let src_offset = src_block_number * block_size_in_bytes;
                            let src_slice =
                                &src_slice[src_offset..src_offset + block_size_in_bytes];
                            unsafe {
                                memcpy_htod_async(
                                    dst_ptr + dst_offset as u64,
                                    src_slice,
                                    stream.stream,
                                )
                            }?
                        }
                    }
                    _ => {
                        //todo
                    }
                }
            }
            (Device::Cuda(src_dev), Device::Cpu) => {
                let (src_storage, _) = src.storage_and_layout();
                let (dst_storage, _) = dst.storage_and_layout();
                match (src_storage.deref(), dst_storage.deref()) {
                    (Storage::Cuda(src_storage), Storage::Cpu(dst_storage)) => {
                        let src_ptr = src_storage.as_cuda_slice::<u8>()?.device_ptr();
                        let dst_ptr = dst_storage.as_slice::<u8>()?;

                        for (src_block_number, dst_block_number) in block_mapping {
                            let dst_offset = dst_block_number * block_size_in_bytes;
                            let src_offset = src_block_number * block_size_in_bytes;
                            let dst_slice = &dst_ptr[dst_offset..dst_offset + block_size_in_bytes];
                            let dst_slice_ptr: *mut u8 = dst_slice.as_ptr() as *mut u8;

                            unsafe {
                                let dst_slice = std::slice::from_raw_parts_mut(
                                    dst_slice_ptr,
                                    block_size_in_bytes,
                                );
                                memcpy_dtoh_async(
                                    dst_slice,
                                    src_ptr + src_offset as u64,
                                    stream.stream,
                                )
                            }?
                        }
                    }
                    _ => {
                        //todo
                    }
                }
            }
            _ => {
                return Err(anyhow!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst)."));
            }
        }

        Ok(())
    }

    pub fn copy(
        &mut self,
        device: &Device,
        stream: &CudaStream,
        key_caches: Vec<&mut Tensor>,
        value_caches: Vec<&mut Tensor>,
        block_mapping: &HashMap<usize, Vec<usize>>,
    ) -> Result<()> {
        let num_layers = key_caches.len();

        let mut key_ptrs = Vec::with_capacity(num_layers);
        let mut value_ptrs = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let key_ptr_int = get_tensor_cuda_slice_ptr_int(key_caches[layer_idx])?;
            let value_ptr_int = get_tensor_cuda_slice_ptr_int(value_caches[layer_idx])?;
            key_ptrs.push(key_ptr_int);
            value_ptrs.push(value_ptr_int);
        }

        // Create block mapping array.
        let mut block_mapping_vec = Vec::with_capacity(block_mapping.len());
        for (src_block_number, v) in block_mapping {
            for dst_block_number in v {
                block_mapping_vec.push(*src_block_number as i64);
                block_mapping_vec.push(*dst_block_number as i64);
            }
        }
        let num_pairs = block_mapping_vec.len() / 2;
        let key_cache_ptrs_tensor = Tensor::from_vec(key_ptrs, num_layers, device)?;
        let value_cache_ptrs_tensor = Tensor::from_vec(value_ptrs, num_layers, device)?;
        let block_mapping_tensor = Tensor::from_vec(block_mapping_vec, num_pairs * 2, device)?;

        //   // Launch the kernel.
        let numel_per_block = key_caches[0].get(0)?.elem_count();
        let mut cfg = LaunchConfig::for_num_elems(numel_per_block as u32);
        cfg.grid_dim.0 = num_layers as u32;
        cfg.grid_dim.1 = num_pairs as u32;
        cfg.grid_dim.2 = 1;
        cfg.block_dim.0 = std::cmp::min(1024, numel_per_block) as u32;
        cfg.block_dim.1 = 1;
        cfg.block_dim.2 = 1;

        let key_cache_ptrs_tensor_data = get_cuda_slice!(key_cache_ptrs_tensor);
        let value_cache_ptrs_tensor_data = get_cuda_slice!(value_cache_ptrs_tensor);
        let block_mapping_tensor_data = get_cuda_slice!(block_mapping_tensor);

        let params = (
            key_cache_ptrs_tensor_data,
            value_cache_ptrs_tensor_data,
            block_mapping_tensor_data,
            numel_per_block,
        );
        let result = dispatch_cuda_func!(self.copy_blocks, key_caches[0].dtype(), cfg, params);
        result.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok(())
    }

    pub fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> candle_core::Result<()> {
        let num_tokens = key.shape().dims()[0];
        let num_heads = key.shape().dims()[1];
        let head_size = key.shape().dims()[2];
        let block_size = key_cache.shape().dims()[3];
        let x = key_cache.shape().dims()[4];
        let key_stride = key.stride()[0];
        let value_stride = value.stride()[0];

        let mut cfg = LaunchConfig::for_num_elems(num_tokens as u32);
        cfg.grid_dim.0 = num_tokens as u32;
        cfg.grid_dim.1 = 1;
        cfg.grid_dim.2 = 1;
        cfg.block_dim.0 = std::cmp::min(num_heads * head_size, 512) as u32;
        cfg.block_dim.1 = 1;
        cfg.block_dim.2 = 1;

        let key_data = get_cuda_slice!(key);
        let value_data = get_cuda_slice!(value);
        let key_cache_data = get_cuda_slice!(key_cache);
        let value_cache_data = get_cuda_slice!(value_cache);
        let slot_mapping_data = get_cuda_slice!(slot_mapping);
        let params = (
            key_data,
            value_data,
            key_cache_data,
            value_cache_data,
            slot_mapping_data,
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size,
            x,
        );
        let result = dispatch_cuda_func!(self.reshape_and_cache, key.dtype(), cfg, params);
        result.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok(())
    }
}
