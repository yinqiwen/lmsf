use candle_core::{DType, Device, Tensor, D};

use crate::model_executor::input_metadata::InputMetadata;

use super::Cache;

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle_core::Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> candle_core::Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

const _PARTITION_SIZE: usize = 512;
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    scale: f32,
    num_kv_heads: usize,
    alibi_slopes: Option<Tensor>,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    head_mapping: Tensor,
    // cache_ops: CacheOps,
    // attention_ops: PagedAttentionOps,
    use_flash_attn: bool,
    cache: Cache,
}

// const _SUPPORTED_HEAD_SIZES: Vec<usize> = vec![64, 80, 96, 112, 128, 256];

impl PagedAttention {
    pub fn new(
        device: &Device,
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        alibi_slopes: Option<Vec<f64>>,
    ) -> candle_core::Result<Self> {
        let cuda_device = if let Device::Cuda(cuda_dev) = &device {
            cuda_dev
        } else {
            return Err(candle_core::bail!("no cuda device"));
        };
        // let cache_ops = CacheOps::new(cuda_device)?;
        // let attention_ops = PagedAttentionOps::new(cuda_device)?;
        let num_kv_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_kv_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        let cache = Cache::new(device)?;
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_kv_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            head_mapping: Tensor::arange(0u32, num_kv_heads as u32, device)?
                .repeat(num_queries_per_kv)?,
            alibi_slopes,
            // cache_ops,
            // attention_ops,
            use_flash_attn: false,
            cache,
        })
    }

    fn repeat_kv(&self, x: Tensor) -> candle_core::Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prompt_attention(
        &self,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        query: Tensor, //[batch_size*seq_len, num_attention_heads,  head_size]
        key: Tensor,   //[batch_size*seq_len, num_attention_heads,  head_size]
        value: Tensor, //[batch_size*seq_len, num_attention_heads,  head_size]
    ) -> candle_core::Result<Tensor> {
        let query = query
            .reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = key
            .reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value: Tensor = value
            .reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self.repeat_kv(key)?;
        let key = self.repeat_kv(key)?;
        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let query = query.transpose(1, 2)?;
            let key = key.transpose(1, 2)?;
            let value = value.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&query, &key, &value, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = query.dtype();
            let q = query.to_dtype(DType::F32)?;
            let k: Tensor = key.to_dtype(DType::F32)?;
            let v = value.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?;
        Ok(y)
    }

    // fn vllm_propmpt_run(){
    //     if self.num_attention_heads != self.num_key_value_heads {
    //         // # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
    //         // # project the key and value tensors to the desired number of
    //         // # heads.
    //         // # TODO(woosuk): Use MQA/GQA kernels for higher performance.
    //         query = query.reshape((
    //             query.shape().dims()[0],
    //             self.num_key_value_heads,
    //             self.num_queries_per_kv,
    //             *query.shape().dims().last().unwrap(),
    //         ))?;
    //         key = key.unsqueeze(2)?;
    //         key = key.reshape((
    //             key.shape().dims()[0],
    //             self.num_key_value_heads,
    //             self.num_queries_per_kv,
    //             *key.shape().dims().last().unwrap(),
    //         ))?;

    //         value = value.unsqueeze(2)?;
    //         value = value.reshape((
    //             value.shape().dims()[0],
    //             self.num_key_value_heads,
    //             self.num_queries_per_kv,
    //             *value.shape().dims().last().unwrap(),
    //         ))?;
    //     }

    //     // # Set attention bias if not provided. This typically happens at the
    //     // # very attention layer of every iteration.
    //     // # FIXME(woosuk): This is a hack.
    //     if input_metadata.attn_bias.is_none() {
    //         if let Some(alibi_slopes) = &self.alibi_slopes {
    //             //     input_metadata.attn_bias = _make_alibi_bias(
    //             //         self.alibi_slopes, self.num_kv_heads, batch_size,
    //             //         seq_len, query.dtype)
    //             let attn_bias = make_alibi_bias(
    //                 alibi_slopes,
    //                 self.num_key_value_heads,
    //                 batch_size,
    //                 seq_len,
    //                 dtype,
    //                 &device,
    //             )?;
    //             input_metadata.attn_bias = Some(Box::new(attn_bias));
    //         } else {
    //             let attn_bias = BlockDiagonalMask::from_seqlens(
    //                 [seq_len.try_into().unwrap()].repeat(batch_size),
    //                 None,
    //             )?;

    //             if let Some(sliding_window) = self.sliding_window {
    //                 let attn_bias = attn_bias.make_local_attention(sliding_window);
    //                 input_metadata.attn_bias = Some(Box::new(attn_bias));
    //             } else {
    //                 input_metadata.attn_bias = Some(Box::new(attn_bias));
    //             }
    //             //     attn_bias = BlockDiagonalCausalMask.from_seqlens(
    //             //         [seq_len] * batch_size)
    //             //     if self.sliding_window is not None:
    //             //         attn_bias = attn_bias.make_local_attention(
    //             //             self.sliding_window)
    //             //     input_metadata.attn_bias = attn_bias
    //         }
    //     }

    //     if let Some(alibi_slopes) = &self.alibi_slopes {
    //         let mut q_new_shape = vec![batch_size, seq_len];
    //         q_new_shape.extend(&query.shape().dims()[1..]);
    //         query = query.reshape(q_new_shape)?;

    //         let mut k_new_shape = vec![batch_size, seq_len];
    //         k_new_shape.extend(&key.shape().dims()[1..]);
    //         key = key.reshape(k_new_shape)?;

    //         let mut v_new_shape = vec![batch_size, seq_len];
    //         v_new_shape.extend(&value.shape().dims()[1..]);
    //         value = value.reshape(v_new_shape)?;
    //         // query = query.unflatten(0, (batch_size, seq_len))
    //         // key = key.unflatten(0, (batch_size, seq_len))
    //         // value = value.unflatten(0, (batch_size, seq_len))
    //     } else {
    //         query = query.unsqueeze(0)?;
    //         key = key.unsqueeze(0)?;
    //         value = value.unsqueeze(0)?;
    //     }
    //     memory_efficient_attention_forward(
    //         &query,
    //         &key,
    //         &value,
    //         input_metadata.attn_bias.as_ref().unwrap(),
    //         0.0,
    //         self.scale,
    //     );
    // }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        query: &Tensor, //[batch_size, seq_len, num_heads * head_size]
        key: &Tensor,   //[batch_size, seq_len, num_heads * head_size]
        value: &Tensor, //[batch_size, seq_len, num_heads * head_size]
        mut key_cache: Option<&Tensor>,
        mut value_cache: Option<&Tensor>,
        input_metadata: &mut InputMetadata,
        dtype: DType,
        device: Device,
        log_enable: bool,
    ) -> candle_core::Result<Tensor> {
        if log_enable {
            // tracing::info!(
            //     "query:{}, key:{},value:{}:is_prompt:{}",
            //     query.to_string(),
            //     key.to_string(),
            //     value.to_string(),
            //     input_metadata.is_prompt
            // );
        }
        let (batch_size, seq_len, hidden_size) = query.shape().dims3()?;
        let mut query = query.reshape(((), self.num_attention_heads, self.head_dim))?;
        let mut key = key.reshape(((), self.num_kv_heads, self.head_dim))?;
        let mut value = value.reshape(((), self.num_kv_heads, self.head_dim))?;
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let slot_mapping = input_metadata.slot_mapping.flatten_all()?;
            // candle_paged_attention::reshape_and_cache(
            //     &key,
            //     &value,
            //     key_cache.unwrap(),
            //     value_cache.unwrap(),
            //     &slot_mapping,
            // )?;
            vllm::cache::apply_reshape_and_cache(
                &key,
                &value,
                key_cache.unwrap(),
                value_cache.unwrap(),
                &slot_mapping,
            )?;
        }

        let output: Tensor = if input_metadata.is_prompt {
            // Prompt run.
            self.prompt_attention(batch_size, seq_len, hidden_size, query, key, value)?
        } else {
            match (key_cache, value_cache) {
                (Some(key_cache), Some(value_cache)) => {
                    let max_context_len = input_metadata.max_context_len.unwrap();
                    // candle_paged_attention::paged_attention(
                    //     &query,
                    //     key_cache,
                    //     value_cache,
                    //     input_metadata.block_tables.as_ref().unwrap(),
                    //     input_metadata.context_lens.as_ref().unwrap(),
                    //     max_context_len,
                    //     self.scale,
                    // )?
                    self.paged_attention(
                        &query,
                        key_cache,
                        value_cache,
                        input_metadata,
                        log_enable,
                    )?
                }
                _ => {
                    // tracing::info!("####empty key val cahche");
                    query.zeros_like()?
                }
            }
        };
        output.reshape((batch_size, seq_len, hidden_size))
    }

    fn paged_attention(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        input_metadata: &mut InputMetadata,
        log_enable: bool,
    ) -> candle_core::Result<Tensor> {
        let num_seqs = query.shape().dims()[0];
        let num_heads = query.shape().dims()[1];
        let max_context_len = input_metadata.max_context_len.unwrap();
        let max_num_partitions = (max_context_len + _PARTITION_SIZE - 1) / _PARTITION_SIZE;

        // # NOTE(woosuk): We use a simple heuristic to decide whether to use
        // # PagedAttention V1 or V2. If the number of partitions is 1, we use
        // # V1 to avoid the overhead of reduction. Also, if the number of
        // # sequences or heads is large, we use V1 since there is enough work
        // # to parallelize.
        // # TODO(woosuk): Tune this heuristic.
        // # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        let use_v1 =
            max_context_len <= 8192 && (max_num_partitions == 1 || num_seqs * num_heads > 512);
        if log_enable {
            // tracing::info!(
            //     "use_v1:{}, block_tables:{},context_lens:{}",
            //     use_v1,
            //     input_metadata.block_tables.as_ref().unwrap().to_string(),
            //     input_metadata.context_lens.as_ref().unwrap().to_string()
            // );
        }
        let output = if use_v1 {
            vllm::attention::apply_paged_attention_v1(
                self.scale,
                max_context_len,
                self.num_kv_heads,
                query,
                key_cache,
                value_cache,
                input_metadata.block_tables.as_ref().unwrap(),
                input_metadata.context_lens.as_ref().unwrap(),
                self.alibi_slopes.as_ref(),
            )?
        } else {
            vllm::attention::apply_paged_attention_v2(
                self.scale,
                max_context_len,
                self.num_kv_heads,
                query,
                key_cache,
                value_cache,
                input_metadata.block_tables.as_ref().unwrap(),
                input_metadata.context_lens.as_ref().unwrap(),
                self.alibi_slopes.as_ref(),
            )?
        };
        Ok(output)
    }
}

// fn make_alibi_bias(
//     alibi_slopes: &Tensor,
//     num_kv_heads: usize,
//     batch_size: usize,
//     seq_len: usize,
//     dtype: DType,
//     device: &Device,
// ) -> candle_core::Result<LowerTriangularMaskWithTensorBias> {
//     let bias = Tensor::arange(0, seq_len as u32, device)?.to_dtype(dtype)?;
//     // # NOTE(zhuohan): HF uses
//     // #     `bias = bias[None, :].repeat(prompt_len, 1)`
//     // # here. We find that both biases give the same results, but
//     // # the bias below more accurately follows the original ALiBi
//     // # paper.
//     let bias = (bias.unsqueeze(0)? - bias.unsqueeze(1)?)?;
//     // bias = bias[None, :] - bias[:, None]
//     let padded_len = ((seq_len + 7) / 8) * 8;
//     let num_heads = alibi_slopes.shape().dims()[0];
//     // num_heads = alibi_slopes.shape[0]
//     let bias_tmp = Tensor::zeros(
//         (batch_size, num_heads, seq_len, padded_len),
//         dtype,
//         alibi_slopes.device(),
//     )?
//     .i((.., .., .., ..seq_len))?;
//     let bias_tmp = bias_tmp.slice_assign(&[.., .., .., ..], &bias)?;
//     let t = alibi_slopes
//         .i(..)?
//         .unsqueeze(D::Minus1)?
//         .unsqueeze(D::Minus1)?;
//     let bias = bias_tmp.mul(&t)?;

//     let bias = if num_heads != num_kv_heads {
//         // bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
//         let mut new_shape = vec![
//             bias.shape().dims()[0],
//             num_kv_heads,
//             num_heads / num_kv_heads,
//         ];
//         new_shape.extend(&bias.shape().dims()[2..]);
//         bias.reshape(new_shape)?
//     } else {
//         bias
//     };
//     let attn_bias = LowerTriangularMaskWithTensorBias::from(bias);
//     Ok(attn_bias)
// }
