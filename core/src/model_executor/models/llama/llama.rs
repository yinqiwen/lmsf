use candle::{IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};
// use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear};

use super::LlamaConfig as Config;
use crate::model_executor::input_metadata::InputMetadata;
use crate::model_executor::layers::{Cache, Layer};
use crate::model_executor::layers::{
    ColumnParallelLinear, LinearWeights, MergedColumnParallelLinear, PagedAttention,
    QKVParallelLinear, RotaryEmbedding, RowParallelLinear,
};
use crate::model_executor::layers::{Embedding, UnquantizedLinearWeights};
use crate::model_executor::models::Model;
use crate::model_executor::parallel::ParallelState;
use crate::tensor::TensorArena;
use common::{DefaultTensorCreator, TensorCreator};

pub const MAX_SEQ_LEN: usize = 4096;

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

struct CausalSelfAttention<W: LinearWeights> {
    // q_proj: Linear,
    // k_proj: Linear,
    // v_proj: Linear,
    // qkv_proj: QKVLinear,
    // o_proj: crate::model_executor::layers::Linear,
    qkv_proj: QKVParallelLinear<W>,
    o_proj: RowParallelLinear<W>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
    rotary_emb: RotaryEmbedding,
    span: tracing::Span,
    span_rot: tracing::Span,
    // t: Option<W>,
}

impl<W: LinearWeights> CausalSelfAttention<W> {
    fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        input_metadata: &mut InputMetadata,
        cache: Option<(&Tensor, &Tensor)>,
        log_enable: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (q, k, v) = self.qkv_proj.forward(x)?;

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
        if log_enable {
            // tracing::info!("before qkv_proj:{}", x.to_string());
        }
        let (q, k, v) = self.qkv_proj.forward_(x, tensor_creator)?;
        if log_enable {
            // tracing::info!("after qkv_proj:{}", q.to_string());
        }

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

    fn load(
        vb: VarBuilder,
        _cache: &Cache,
        cfg: &Config,
        parallel_state: &ParallelState,
        config: W::Config,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        // let size_in = cfg.hidden_size;
        // let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        // let _size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let hidden_size = cfg.hidden_size;
        let tp_size = parallel_state.get_tensor_model_parallel_world_size();
        let total_num_heads = cfg.num_attention_heads;
        let _num_heads = total_num_heads / tp_size;
        let total_num_kv_heads = cfg.num_key_value_heads;

        let _num_kv_heads = std::cmp::max(1, total_num_kv_heads / tp_size);
        let head_dim = hidden_size / total_num_heads;

        // self.q_size = self.num_heads * self.head_dim
        // self.kv_size = self.num_kv_heads * self.head_dim
        // self.scaling = self.head_dim**-0.5
        // self.rope_theta = rope_theta
        // self.max_position_embeddings = max_position_embeddings

        let _head_size = cfg.hidden_size / cfg.num_attention_heads;
        let qkv_proj = QKVParallelLinear::<W>::new(
            hidden_size,
            head_dim,
            total_num_heads,
            Some(total_num_kv_heads),
            parallel_state,
        )
        .load(&vb, &["q_proj", "k_proj", "v_proj"], config.clone(), true)?;

        let o_proj =
            RowParallelLinear::<W>::new(total_num_heads * head_dim, hidden_size, parallel_state)
                .load(&vb.pp("o_proj"), config)?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
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
                head_dim,
                MAX_SEQ_LEN,
                10000.0,
                true,
            )?,
            span,
            span_rot,
            // t: None,
        })
    }
}

struct Mlp<W: LinearWeights> {
    // c_fc1: crate::model_executor::layers::Linear,
    // c_fc2: crate::model_executor::layers::Linear,
    // gate_up: crate::model_executor::layers::Linear,
    // c_proj: crate::model_executor::layers::Linear,
    gate_up: MergedColumnParallelLinear<W>,
    c_proj: RowParallelLinear<W>,
    span: tracing::Span,
    // t: Option<W>,
}

impl<W: LinearWeights> Mlp<W> {
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

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        parallel_state: &ParallelState,
        config: W::Config,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up = MergedColumnParallelLinear::<W>::new(
            h_size,
            &[i_size, i_size],
            parallel_state,
        )
        .load(&vb, &["gate_proj", "up_proj"], config.clone(), true)?;
        let c_proj = RowParallelLinear::<W>::new(i_size, h_size, parallel_state)
            .load(&vb.pp("down_proj"), config)?;
        Ok(Self {
            gate_up,
            c_proj,
            span,
        })
    }
}

struct Block<W: LinearWeights> {
    rms_1: vllm::RmsNorm,
    attn: CausalSelfAttention<W>,
    rms_2: vllm::RmsNorm,
    mlp: Mlp<W>,
    span: tracing::Span,
    idx: usize,
}

impl<W: LinearWeights> Block<W> {
    fn forward_<F: TensorCreator>(
        &self,
        tensor_creator: &mut F,
        hidden_states: Tensor,
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
        if self.idx == 0 {
            // tracing::info!("before attn:{}", hidden_states.to_string());
        }

        let hidden_states = self.attn.forward_(
            tensor_creator,
            &hidden_states,
            positions,
            input_metadata,
            cache,
            self.idx == 0,
        )?;
        if self.idx == 0 {
            // tracing::info!("after attn:{}", hidden_states.to_string());
        }
        self.rms_2.forward_residual_(&hidden_states, &residual)?;

        // let start = std::time::Instant::now();
        let hidden_states = self
            .mlp
            .forward_(&hidden_states, self.idx == 0, tensor_creator)?;
        Ok((hidden_states, residual))
    }

    fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        idx: usize,
        parallel_state: &ParallelState,
        config: W::Config,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");

        let attn = CausalSelfAttention::<W>::load(
            vb.pp("self_attn"),
            cache,
            cfg,
            parallel_state,
            config.clone(),
        )?;

        let mlp = Mlp::<W>::load(vb.pp("mlp"), cfg, parallel_state, config)?;

        let rms_1 = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            None,
            vb.pp("input_layernorm"),
        )?;

        let rms_2 = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            None,
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

pub struct Llama<W: LinearWeights> {
    wte: Embedding,
    blocks: Vec<Block<W>>,
    ln_f: vllm::RmsNorm,
    // lm_head: crate::model_executor::layers::Linear,
    lm_head: ColumnParallelLinear<UnquantizedLinearWeights>,
    block_arena: [TensorArena; 2],
}

impl<W: LinearWeights> Llama<W> {
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
        // tracing::info!("wte:{}", x.to_string());
        let mut residual: Option<Tensor> = None;
        let _start = std::time::Instant::now();

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

    pub fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        parallel_state: &ParallelState,
        linear_config: W::Config,
    ) -> Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
        //let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let lm_head = ColumnParallelLinear::<UnquantizedLinearWeights>::new(
            cfg.hidden_size,
            cfg.vocab_size,
            parallel_state,
        )
        .load(&vb.pp("lm_head"), None)?;

        let ln_f =
            vllm::RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, None, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::<W>::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cache,
                    cfg,
                    i,
                    parallel_state,
                    linear_config.clone(),
                )
                .unwrap()
            })
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

impl<W: LinearWeights> Model for Llama<W> {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        mut input_metadata: InputMetadata,
    ) -> anyhow::Result<Tensor> {
        self.do_forward(
            &input_tokens,
            &input_positions,
            kv_cache,
            &mut input_metadata,
        )
        .map_err(|e| anyhow::anyhow!("{}", e))
    }
}
