use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
// use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::model_executor::input_metadata::InputMetadata;
use crate::model_executor::layers::Cache;
use crate::model_executor::layers::Embedding;
use crate::model_executor::layers::{PagedAttention, QKVLinear, RotaryEmbedding};
use crate::model_executor::models::Model;
use crate::tensor::{cuda_add_, TensorArena};
use common::{DefaultTensorCreator, TensorCreator};

pub const MAX_SEQ_LEN: usize = 4096;

#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
        }
    }
}

pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}

struct RmsNorm {
    // inner: candle_nn::RmsNorm,
    inner: tops::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        // let inner = candle_nn::rms_norm(size, eps, vb)?;
        let inner = tops::RmsNorm::load(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

struct CausalSelfAttention {
    // q_proj: Linear,
    // k_proj: Linear,
    // v_proj: Linear,
    qkv_proj: QKVLinear,
    o_proj: crate::model_executor::layers::Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
    rotary_emb: RotaryEmbedding,
    span: tracing::Span,
    span_rot: tracing::Span,
}

// #[cfg(feature = "flash-attn")]
// fn flash_attn(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     softmax_scale: f32,
//     causal: bool,
// ) -> Result<Tensor> {
//     candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
// }

// #[cfg(not(feature = "flash-attn"))]
// fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
//     unimplemented!("compile with '--features flash-attn'")
// }

impl CausalSelfAttention {
    fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        input_metadata: &mut InputMetadata,
        cache: Option<(&Tensor, &Tensor)>,
        log_enable: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        // let (b_sz, seq_len, hidden_size) = x.dims3()?;
        // let mut q = self.q_proj.forward(x)?; //[batch_size, seq_len, num_heads * head_size]
        // let mut k = self.k_proj.forward(x)?; //[batch_size, seq_len, num_heads * head_size]
        // let v = self.v_proj.forward(x)?; //[batch_size, seq_len, num_heads * head_size]
        //let debug_file = if log_enable { Some("test_qkv") } else { None };
        let (q, k, v) = self.qkv_proj.forward(x)?;

        // let mut q = q
        //     .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
        //     .transpose(1, 2)?;

        // let mut k = k
        //     .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
        //     .transpose(1, 2)?;

        // let v = v
        //     .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
        //     .transpose(1, 2)?;

        // if log_enable {
        //     tracing::info!("before rotary_emb: query:{:?}/{:?}", q.shape(), q.stride());
        // }

        //rotary_emb accept shape [batch_size, seq_len, num_heads * head_size]

        self.rotary_emb.forward(positions, &q, &k)?;

        let dtype = q.dtype();

        let mut default_creator = DefaultTensorCreator {};
        let attn_output = self.attn.forward(
            &q,
            &k,
            &v,
            cache.map(|(k, _)| k),
            cache.map(|(_, v)| v),
            input_metadata,
            dtype,
            log_enable,
            &mut default_creator,
        )?;

        self.o_proj.forward(&attn_output)
    }
    fn forward_<F: TensorCreator>(
        &self,
        tensor_creator: &mut F,
        x: &Tensor,
        positions: &Tensor,
        input_metadata: &mut InputMetadata,
        cache: Option<(&Tensor, &Tensor)>,
        log_enable: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (q, k, v) = self.qkv_proj.forward_(x, tensor_creator)?;

        self.rotary_emb.forward(positions, &q, &k)?;

        let dtype = q.dtype();
        // let device = q.device().clone();

        let attn_output = self.attn.forward(
            &q,
            &k,
            &v,
            cache.map(|(k, _)| k),
            cache.map(|(_, v)| v),
            input_metadata,
            dtype,
            log_enable,
            tensor_creator,
        )?;
        if log_enable {
            // tracing::info!("after attn: :{}", attn_output.to_string());
            // todo!("aaa");
        }
        self.o_proj.forward_(&attn_output, tensor_creator)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
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

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        // let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        // let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        // let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let qkv_proj = QKVLinear::load_qkv(
            &vb, size_in, size_q, "q_proj", size_kv, "k_proj", size_kv, "v_proj",
        )?;

        //let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        let o_proj =
            crate::model_executor::layers::linear_no_bias(size_q, size_in, vb.pp("o_proj"))?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            // q_proj,
            // k_proj,
            // v_proj,
            qkv_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            attn: PagedAttention::new(
                vb.device(),
                cfg.num_attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(cfg.num_key_value_heads),
                None,
                None,
            )?,
            rotary_emb: RotaryEmbedding::new(
                vb.device(),
                vb.dtype(),
                head_dim,
                cfg.num_key_value_heads,
                head_dim,
                MAX_SEQ_LEN,
                10000.0,
                true,
            )?,
            span,
            span_rot,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

struct Mlp {
    // c_fc1: crate::model_executor::layers::Linear,
    // c_fc2: crate::model_executor::layers::Linear,
    gate_up: crate::model_executor::layers::Linear,
    c_proj: crate::model_executor::layers::Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward_<F: TensorCreator>(
        &self,
        x: &Tensor,
        log_enable: bool,
        tensor_creator: &mut F,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = self.gate_up.forward_(x, tensor_creator)?;

        if log_enable {
            // tracing::info!("after gateup:{}\n{}", x1.to_string(), x2.to_string());
        }
        let x = vllm::silu_and_mul_(&x, tensor_creator)?;
        self.c_proj.forward_(&x, tensor_creator)
    }
    fn forward(&self, x: &Tensor, log_enable: bool) -> Result<Tensor> {
        let _enter = self.span.enter();
        // let x1 = self.c_fc1.forward(x)?;
        // let x2 = self.c_fc2.forward(x)?;
        let x = self.gate_up.forward(x)?;
        // println!("x shape:{:?}", x.shape());
        // let dim0 = *x.dims().last().unwrap();
        // let x1 = x.i((.., .., 0..dim0 / 2))?;
        // let x2 = x.i((.., .., (dim0 / 2)..))?;
        // let x1 = x.narrow(0, 0, dim0 / 2)?;
        // let x2 = x.narrow(0, dim0 / 2, dim0 / 2)?;
        if log_enable {
            // tracing::info!("after gateup:{}\n{}", x1.to_string(), x2.to_string());
        }
        let x = vllm::silu_and_mul(&x)?;

        // tops::cuda_silu_activation(&x1, &x2, std::ptr::null_mut())?;
        // let x = x1;
        // if log_enable {
        //     tracing::info!("after cuda_silu_activation:{}", x.to_string());
        // }
        //let x = (candle_nn::ops::silu(&x1)? * x2)?;
        // let x = candle_nn::ops::silu(&(x1 * x2)?)?;
        //let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up = crate::model_executor::layers::Linear::load_multi(
            &vb,
            h_size,
            i_size,
            &["gate_proj", "up_proj"],
        )?;
        // println!("gate_up shape:{:?}", gate_up.weight.shape());
        // drop(gate_up);

        // let c_fc1 =
        //     crate::model_executor::layers::linear_no_bias(h_size, i_size, vb.pp("gate_proj"))?;
        // let c_fc2 =
        //     crate::model_executor::layers::linear_no_bias(h_size, i_size, vb.pp("up_proj"))?;
        // println!(
        //     "fc1/2 shape:{:?}/{:?}",
        //     c_fc1.weight.shape(),
        //     c_fc2.weight.shape()
        // );
        let c_proj =
            crate::model_executor::layers::linear_no_bias(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            // c_fc1,
            // c_fc2,
            gate_up,
            c_proj,
            span,
        })
    }
}

// struct Block {
//     rms_1: RmsNorm,
//     attn: CausalSelfAttention,
//     rms_2: RmsNorm,
//     mlp: Mlp,
//     span: tracing::Span,
//     idx: usize,
// }

// impl Block {
//     fn forward(
//         &self,
//         x: &Tensor,
//         positions: &Tensor,
//         cache: Option<(&Tensor, &Tensor)>,
//         input_metadata: &mut InputMetadata,
//     ) -> Result<Tensor> {
//         let cuda_dev = match x.device() {
//             Device::Cuda(cuda) => cuda.clone(),
//             _ => {
//                 candle_core::bail!("")
//             }
//         };

//         let _enter = self.span.enter();
//         let residual = x;
//         let x = self.rms_1.forward(x)?;
//         // cuda_dev.synchronize();
//         // tracing::info!("Block 0 cost {:?}", start.elapsed(),);
//         // cuda_dev.synchronize();
//         // let start = std::time::Instant::now();
//         // let x = (self
//         //     .attn
//         //     .forward(&x, positions, input_metadata, cache, self.idx == 0)?
//         //     + residual)?;
//         let x = self
//             .attn
//             .forward(&x, positions, input_metadata, cache, self.idx == 0)?;
//         cuda_add_(&x, &residual)?;
//         // cuda_dev.synchronize();
//         // tracing::info!("Block attn cost {:?}", start.elapsed(),);
//         let residual = &x;
//         //let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
//         let x = self.mlp.forward(&self.rms_2.forward(&x)?, false)?;
//         cuda_add_(&x, &residual)?;
//         // cuda_dev.synchronize();
//         // tracing::info!("Block mlp cost {:?}", start.elapsed(),);
//         Ok(x)
//     }

//     fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, idx: usize) -> Result<Self> {
//         let span = tracing::span!(tracing::Level::TRACE, "block");
//         let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg)?;
//         let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
//         let rms_1 = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
//         let rms_2 = RmsNorm::load(
//             cfg.hidden_size,
//             cfg.rms_norm_eps,
//             vb.pp("post_attention_layernorm"),
//         )?;
//         Ok(Self {
//             rms_1,
//             attn,
//             rms_2,
//             mlp,
//             span,
//             idx,
//         })
//     }
// }

struct Block2 {
    rms_1: vllm::RmsNorm,
    attn: CausalSelfAttention,
    rms_2: vllm::RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
    idx: usize,
}

impl Block2 {
    fn forward_<F: TensorCreator>(
        &self,
        tensor_creator: &mut F,
        mut hidden_states: Tensor,
        positions: &Tensor,
        residual: Option<Tensor>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = match residual {
            Some(residual) => {
                self.rms_1.forward_residual_(&hidden_states, &residual)?;
                (hidden_states, residual)
            }
            None => {
                // if self.idx == 0 {
                //     tracing::info!(
                //         "enter layer:{}/ {}",
                //         hidden_states.to_string(),
                //         residual.is_none()
                //     );
                // }
                let new_hidden_states =
                    self.rms_1
                        .forward_(&hidden_states, tensor_creator, self.idx == 0)?;
                // if self.idx == 0 {
                //     tracing::info!("after rms1:{}", new_hidden_states.to_string());
                // }
                let residual = hidden_states;
                (new_hidden_states, residual)
            }
        };
        // if self.idx == 0 {
        //     tracing::info!("before attn:{}", hidden_states.to_string());
        // }

        let hidden_states = self.attn.forward_(
            tensor_creator,
            &hidden_states,
            positions,
            input_metadata,
            cache,
            self.idx == 0,
        )?;
        // if self.idx == 0 {
        //     tracing::info!("after attn:{}", hidden_states.to_string());
        // }
        self.rms_2.forward_residual_(&hidden_states, &residual)?;

        // let start = std::time::Instant::now();
        let hidden_states = self
            .mlp
            .forward_(&hidden_states, self.idx == 0, tensor_creator)?;
        Ok((hidden_states, residual))
    }
    fn forward(
        &self,
        mut hidden_states: Tensor,
        positions: &Tensor,
        residual: Option<Tensor>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = match residual {
            Some(residual) => {
                self.rms_1.forward_residual_(&hidden_states, &residual)?;
                (hidden_states, residual)
            }
            None => {
                let new_hidden_states = self.rms_1.forward(&hidden_states)?;
                let residual = hidden_states;
                (new_hidden_states, residual)
            }
        };

        let hidden_states = self.attn.forward(
            &hidden_states,
            positions,
            input_metadata,
            cache,
            self.idx == 0,
        )?;
        self.rms_2.forward_residual_(&hidden_states, &residual)?;

        // let start = std::time::Instant::now();
        let hidden_states = self.mlp.forward(&hidden_states, self.idx == 0)?;
        Ok((hidden_states, residual))
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, idx: usize) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 =
            vllm::RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
            idx,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block2>,
    ln_f: vllm::RmsNorm,
    lm_head: crate::model_executor::layers::Linear,
    block_arena: [TensorArena; 2],
}

impl Llama {
    pub fn do_forward(
        &mut self,
        x: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        // let cuda_dev = match x.device() {
        //     Device::Cuda(cuda) => cuda.clone(),
        //     _ => {
        //         candle_core::bail!("")
        //     }
        // };
        // let start = std::time::Instant::now();

        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;

        // cuda_dev.synchronize();
        // tracing::info!("wte cost {:?}", start.elapsed(),);
        let mut residual: Option<Tensor> = None;
        let start = std::time::Instant::now();

        if let Some(kv_caches) = kv_caches {
            for (idx, block) in self.blocks.iter().enumerate() {
                let arena_idx = idx % 2;
                self.block_arena[1 - arena_idx].reset();
                let (hidden_states, new_residual) = block.forward_(
                    &mut self.block_arena[arena_idx],
                    x,
                    positions,
                    residual,
                    Some((&kv_caches[idx].0, &kv_caches[idx].1)),
                    input_metadata,
                )?;
                x = hidden_states;
                residual = Some(new_residual);
            }
        } else {
            // self.block_arena[0].reset();
            for (idx, block) in self.blocks.iter().enumerate() {
                let arena_idx = idx % 2;
                self.block_arena[1 - arena_idx].reset();
                let (hidden_states, new_residual) = block.forward_(
                    &mut self.block_arena[arena_idx],
                    x,
                    positions,
                    residual,
                    None,
                    input_metadata,
                )?;
                x = hidden_states;
                residual = Some(new_residual);
            }
        }
        // cuda_dev.synchronize();
        // tracing::info!("block cost {:?}", start.elapsed(),);
        // cuda_dev.synchronize();
        // let start = std::time::Instant::now();
        // tracing::info!("x0 shape:{:?}/{}", x.shape(), x.to_string());
        // tracing::info!("after layers:{:?}", x.to_string());
        //let x = self.ln_f.forward(&x)?;
        // tracing::info!("before ln_f0:{:?}", x.to_string());
        self.ln_f
            .forward_residual_(&x, residual.as_ref().unwrap())?;
        // cuda_dev.synchronize();
        // tracing::info!("ln_f cost {:?}", start.elapsed(),);

        let x = x.i((.., seq_len - 1, ..))?;
        // tracing::info!("after ln_f1:{:?}", x.to_string());

        // tracing::info!("x1 shape:{:?}/{}", x.shape(), x.to_string());
        // tracing::info!("x2 shape:{:?}", x.shape());
        let logits = self.lm_head.forward_(&x, &mut self.block_arena[1])?;
        // let logits = logits.to_dtype(DType::F32)?;
        // cuda_dev.synchronize();
        // tracing::info!("lm_head cost {:?}", start.elapsed(),);
        Ok(logits)
    }

    pub fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
        //let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let lm_head = crate::model_executor::layers::linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
        )?;
        let ln_f = vllm::RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block2::load(vb.pp(&format!("model.layers.{i}")), cache, cfg, i).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            block_arena: [TensorArena::new(vb.device()), TensorArena::new(vb.device())],
        })
    }
}

impl Model for Llama {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        mut input_metadata: InputMetadata,
    ) -> anyhow::Result<candle_core::Tensor> {
        self.do_forward(
            &input_tokens,
            &input_positions,
            kv_cache,
            &mut input_metadata,
        )
        .map_err(|e| anyhow::anyhow!("{}", e))
    }
}
