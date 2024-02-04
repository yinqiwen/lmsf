use candle_core::{DType, Device, Tensor, D};
use std::clone;
use std::sync::Arc;

// use crate::model_executor::ops::PosEncoding;
use std::sync::OnceLock;

fn get_cos_sin_cache(
    base: f32,
    rotary_dim: usize,
    max_position_embeddings: usize,
    dtype: DType,
    device: &Device,
) -> Arc<Tensor> {
    static CACHE: OnceLock<Arc<Tensor>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let t = vllm::pos_encoding::compute_cos_sin_cache(
                base,
                rotary_dim,
                max_position_embeddings,
                dtype,
                device,
            )
            .unwrap();
            Arc::new(t)
        })
        .clone()
}

pub struct RotaryEmbedding {
    head_size: usize,
    num_key_value_heads: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f32,
    is_neox: bool,
    // pos_encoding: PosEncoding,
    // cos_sin_cache: Tensor,
    // cos: Tensor,
    // sin: Tensor,
    cos_sin_cache: Arc<Tensor>,
}

// fn compute_cos_sin_cache(
//     rotary_dim: usize,
//     base: f32,
//     max_position_embeddings: usize,
//     device: &Device,
//     dtype: DType,
// ) -> candle_core::Result<Tensor> {
//     let inv_freq: Vec<_> = (0..rotary_dim)
//         .step_by(2)
//         .map(|i| 1f32 / base.powf(i as f32 / rotary_dim as f32))
//         .collect();
//     let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
//     let t = Tensor::arange(0, max_position_embeddings as u32, device)?
//         .to_dtype(DType::F32)?
//         .reshape((max_position_embeddings, 1))?;
//     let inv_freq_n = inv_freq.elem_count();
//     let inv_freq = inv_freq.reshape((1, inv_freq_n))?;
//     let freqs = t.matmul(&inv_freq)?;
//     let cos = freqs.cos()?;
//     let sin = freqs.sin()?;
//     let last = cos.dims().len() - 1;
//     Tensor::cat(&[cos, sin], last)
//     // freqs = torch.einsum("i,j -> ij", t, inv_freq)
//     // cos = freqs.cos()
//     // sin = freqs.sin()
//     // cache = torch.cat((cos, sin), dim=-1)
// }

impl RotaryEmbedding {
    pub fn new(
        device: &Device,
        dtype: DType,
        head_size: usize,
        num_key_value_heads: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f32,
        is_neox_style: bool,
    ) -> candle_core::Result<Self> {
        // get_cos_sin_cache(base, rotary_dim, max_position_embeddings, dtype, device)
        // let cos_sin_cache = compute_cos_sin_cache(
        //     rotary_dim,
        //     rope_theta,
        //     max_position_embeddings,
        //     device,
        //     dtype,
        // )?;

        // tracing::info!("cos_sin_cache shape:{:?}", cos_sin_cache.shape());
        // Create inv freqs
        // let inv_freqs = candle_rotary::inv_freqs(rotary_dim, 10000f32, device)?;
        // // Create an over-sized cos sin cache like you would usually do
        // let (cos, sin) = candle_rotary::cos_sin(32, &inv_freqs, dtype)?;

        Ok(Self {
            head_size,
            num_key_value_heads,
            rotary_dim,
            max_position_embeddings,
            rope_theta,
            is_neox: is_neox_style,
            // pos_encoding,
            // cos_sin_cache,
            // cos,
            // sin,
            cos_sin_cache: get_cos_sin_cache(
                10000f32,
                rotary_dim,
                max_position_embeddings,
                dtype,
                device,
            ),
        })
    }
    pub fn forward(
        &self,
        position: &Tensor,
        query: &mut Tensor,
        key: &mut Tensor,
    ) -> candle_core::Result<()> {
        // let num_tokens = (query.elem_count() / query.shape().dims().last().unwrap()) as i32;
        // //   int64_t num_tokens = query.numel() / query.size(-1);
        // let rot_dim = self.cos_sin_cache.shape().dims()[1] as i32;
        // //   int rot_dim = cos_sin_cache.size(1);
        // let num_heads = (query.shape().dims().last().unwrap() / self.head_size) as i32;
        // //   int num_heads = query.size(-1) / head_size;
        // let num_kv_heads = (key.shape().dims().last().unwrap() / self.head_size) as i32;
        // //   int num_kv_heads = key.size(-1) / head_size;
        // let query_strides = query.stride();
        // let query_stride = query_strides[query_strides.len() - 2] as i64;
        // //   int64_t query_stride = query.stride(-2);
        // let key_strides = key.stride();
        // let key_stride = key_strides[key_strides.len() - 2] as i64;
        // let params = vllm_kernels::RotaryEmbeddingKernelParams {
        //     stream: std::ptr::null_mut(),
        //     scalar_type: vllm_kernels::get_scalar_type(query.dtype()),
        //     is_neox: self.is_neox,
        //     query_stride,
        //     key_stride,
        //     num_tokens,
        //     rot_dim,
        //     num_heads,
        //     num_kv_heads,
        //     head_size: self.head_size as i32,
        // };
        // vllm_kernels::rotary_embedding_tensor(position, query, key, &self.cos_sin_cache, params)?;
        // Filter cos and sin
        // let (b_sz, seq_len) = position.dims2()?;
        // let select_pos = position.reshape(b_sz * seq_len)?;
        // let cos = self.cos.index_select(&select_pos, 0)?;
        // let sin = self.sin.index_select(&select_pos, 0)?;

        // // q,k shape  //[batch_size, seq_len, num_heads * head_size]
        let (b_sz, seq_len, hidden_size) = query.dims3()?;
        let fwd_q = query.reshape((b_sz * seq_len, self.num_key_value_heads, self.head_size))?;
        let fwd_k = key.reshape((b_sz * seq_len, self.num_key_value_heads, self.head_size))?;
        // // Inplace
        // candle_rotary::apply_rotary_inplace(&fwd_q, &fwd_k, &cos, &sin, self.is_neox)?;

        // tracing::info!(
        //     "cos sin cache:{:?}, {:?}, {:?}",
        //     self.cos_sin_cache.shape(),
        //     query.shape(),
        //     key.shape()
        // );
        vllm::pos_encoding::apply_rotary_embedding(
            position,
            &fwd_q,
            &fwd_k,
            &self.cos_sin_cache.as_ref(),
            self.head_size,
            self.is_neox,
        )?;

        Ok(())

        // self.pos_encoding.rotary_embedding(
        //     position,
        //     query,
        //     key,
        //     self.head_size,
        //     &self.cos_sin_cache,
        //     self.is_neox,
        // )
    }
}
