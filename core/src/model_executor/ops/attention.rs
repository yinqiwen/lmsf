use std::ops::Deref;

use candle::{
    cuda_backend::cudarc::driver::{
        result::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async},
        CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
    },
    CudaDevice, DType, Device, Storage, Tensor,
};

use crate::{dispatch_cuda_func, get_cuda_device, get_cuda_slice};
pub struct PagedAttentionOps {
    paged_attention_v1: Vec<CudaFunction>,
    paged_attention_v2: Vec<CudaFunction>,
    paged_attention_reduce_v2: Vec<CudaFunction>,
}

const WARP_SIZE: usize = 32;
const NUM_THREADS: usize = 128;
const PARTITION_SIZE: usize = 512;

#[repr(C)]
struct PagedAttentionKernelParams {
    block_size: i32,
    head_size: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
    num_kv_heads: i32,
    max_num_blocks_per_seq: i32,
    max_num_partitions: i32,
}
/// We have to implement this to send it to cuda!
unsafe impl DeviceRepr for PagedAttentionKernelParams {}

impl PagedAttentionOps {
    pub fn new(device: &CudaDevice) -> candle::Result<Self> {
        Ok(Self {
            paged_attention_v1: vec![
                device.get_or_load_func("paged_attention_v1_f16", kernels::ATTENTION_KERNELS)?,
                device.get_or_load_func("paged_attention_v1_f32", kernels::ATTENTION_KERNELS)?,
            ],
            paged_attention_v2: vec![
                device.get_or_load_func("paged_attention_v2_f16", kernels::ATTENTION_KERNELS)?,
                device.get_or_load_func("paged_attention_v2_f32", kernels::ATTENTION_KERNELS)?,
            ],
            paged_attention_reduce_v2: vec![
                device.get_or_load_func(
                    "paged_attention_v2_reduce_f16",
                    kernels::ATTENTION_KERNELS,
                )?,
                device.get_or_load_func(
                    "paged_attention_v2_reduce_f32",
                    kernels::ATTENTION_KERNELS,
                )?,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_v1(
        &self,
        block_size: usize,
        out: &mut Tensor,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        num_kv_heads: usize,
        scale: f32,
        block_tables: &Tensor,
        context_lens: &Tensor,
        max_context_len: usize,
        alibi_slopes: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        let num_seqs = query.shape().dims()[0];
        let num_heads = query.shape().dims()[1];
        let head_size = query.shape().dims()[2];
        let max_num_blocks_per_seq = block_tables.shape().dims()[1];
        let q_stride = query.stride()[0];
        let kv_block_stride = key_cache.stride()[0];
        let kv_head_stride = key_cache.stride()[1];

        // let thread_group_size = std::cmp::max(1, WARP_SIZE / block_size);
        let num_warps = NUM_THREADS / WARP_SIZE;
        let padded_max_context_len = max_context_len.div_ceil(block_size) * block_size;
        let logits_size = padded_max_context_len * 4;
        let outputs_size = (num_warps / 2) * head_size * 4;
        let shared_mem_size = std::cmp::max(logits_size, outputs_size);

        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (NUM_THREADS as u32, 1, 1),
            shared_mem_bytes: shared_mem_size as u32,
        };

        let query_data = get_cuda_slice!(query);
        let out_data = get_cuda_slice!(out);
        let key_cache_data = get_cuda_slice!(key_cache);
        let value_cache_data = get_cuda_slice!(value_cache);
        let block_tables_data = get_cuda_slice!(block_tables);
        let context_lens_data = get_cuda_slice!(context_lens);

        let ext_params = PagedAttentionKernelParams {
            block_size: block_size as i32,
            head_size: head_size as i32,
            q_stride: q_stride as i32,
            kv_block_stride: kv_block_stride as i32,
            kv_head_stride: kv_head_stride as i32,
            num_kv_heads: num_kv_heads as i32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as i32,
            max_num_partitions: 0,
        };

        let result = if let Some(alibi_slopes) = alibi_slopes {
            let alibi_slopes_data = get_cuda_slice!(alibi_slopes);
            let params = (
                out_data,
                query_data,
                key_cache_data,
                value_cache_data,
                scale,
                block_tables_data,
                context_lens_data,
                alibi_slopes_data,
                ext_params,
            );
            dispatch_cuda_func!(self.paged_attention_v1, query.dtype(), cfg, params)
        } else {
            let cuda_device = get_cuda_device!(out.device());

            let alibi_slopes_null_data = cuda_device
                .null::<i64>()
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            let params = (
                out_data,
                query_data,
                key_cache_data,
                value_cache_data,
                scale,
                block_tables_data,
                context_lens_data,
                &alibi_slopes_null_data,
                ext_params,
            );
            dispatch_cuda_func!(self.paged_attention_v1, query.dtype(), cfg, params)
        };
        result.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_v2(
        &self,
        out: &mut Tensor,
        tmp_out: &mut Tensor,
        exp_sums: &Tensor,
        max_logits: &Tensor,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        num_kv_heads: usize,
        scale: f32,
        block_tables: &Tensor,
        context_lens: &Tensor,
        block_size: usize,
        max_context_len: usize,
        alibi_slopes: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        let num_seqs = query.shape().dims()[0];
        let num_heads = query.shape().dims()[1];
        let head_size = query.shape().dims()[2];
        let max_num_blocks_per_seq = block_tables.shape().dims()[1];
        let q_stride = query.stride()[0];
        let kv_block_stride = key_cache.stride()[0];
        let kv_head_stride = key_cache.stride()[1];

        let num_warps = NUM_THREADS / WARP_SIZE;
        let max_num_partitions = max_context_len.div_ceil(PARTITION_SIZE);
        let logits_size = PARTITION_SIZE * 4;
        let outputs_size = (num_warps / 2) * head_size * 4;
        let shared_mem_size = std::cmp::max(logits_size, outputs_size);
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, max_num_partitions as u32),
            block_dim: (num_heads as u32, num_seqs as u32, 1),
            shared_mem_bytes: shared_mem_size as u32,
        };
        let reduce_shared_mem_size = 2 * max_num_partitions * 4;
        let reduce_cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (num_heads as u32, num_seqs as u32, 1),
            shared_mem_bytes: reduce_shared_mem_size as u32,
        };

        let ext_params = PagedAttentionKernelParams {
            block_size: block_size as i32,
            head_size: head_size as i32,
            q_stride: q_stride as i32,
            kv_block_stride: kv_block_stride as i32,
            kv_head_stride: kv_head_stride as i32,
            num_kv_heads: num_kv_heads as i32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as i32,
            max_num_partitions: max_num_partitions as i32,
        };
        let exp_sums_data = get_cuda_slice!(exp_sums);
        let max_logits_data = get_cuda_slice!(max_logits);
        let out_data = get_cuda_slice!(out);
        let tmp_out_data = get_cuda_slice!(tmp_out);
        let query_data = get_cuda_slice!(query);
        let key_cache_data = get_cuda_slice!(key_cache);
        let value_cache_data = get_cuda_slice!(value_cache);
        let block_tables_data = get_cuda_slice!(block_tables);
        let context_lens_data = get_cuda_slice!(context_lens);

        let result = if let Some(alibi_slopes) = alibi_slopes {
            let alibi_slopes_data = get_cuda_slice!(alibi_slopes);
            let params = (
                exp_sums_data.clone(),
                max_logits_data.clone(),
                out_data.clone(),
                tmp_out_data.clone(),
                query_data,
                key_cache_data,
                value_cache_data,
                scale,
                block_tables_data,
                context_lens_data.clone(),
                alibi_slopes_data,
                ext_params,
            );
            dispatch_cuda_func!(self.paged_attention_v2, query.dtype(), cfg, params)
        } else {
            let cuda_device = if let candle_core::Device::Cuda(cuda_dev) = out.device() {
                cuda_dev
            } else {
                unimplemented!("unreach");
            };
            let alibi_slopes_null_data = cuda_device
                .null::<i64>()
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            let params = (
                out_data.clone(),
                query_data,
                key_cache_data,
                value_cache_data,
                scale,
                block_tables_data,
                context_lens_data.clone(),
                &alibi_slopes_null_data,
                ext_params,
            );
            dispatch_cuda_func!(self.paged_attention_v2, query.dtype(), cfg, params)
        };
        result.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;

        let reduce_params = (
            exp_sums_data,
            max_logits_data,
            out_data,
            tmp_out_data,
            context_lens_data,
            block_size as i32,
            head_size as i32,
            max_num_partitions as i32,
        );
        let reduce_result = dispatch_cuda_func!(
            self.paged_attention_reduce_v2,
            query.dtype(),
            reduce_cfg,
            reduce_params
        );
        reduce_result.map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok(())
    }
}
