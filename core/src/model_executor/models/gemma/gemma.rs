use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::GemmaConfig as Config;

use crate::model_executor::input_metadata::InputMetadata;
use crate::model_executor::layers::{Cache, Layer};
use crate::model_executor::layers::{
    MergedColumnParallelLinear, PagedAttention, QKVParallelLinear, RotaryEmbedding,
    RowParallelLinear, UnquantizedLinearWeights,
};
use crate::model_executor::models::Model;
use crate::model_executor::parallel::ParallelState;
use crate::tensor::TensorArena;
use common::TensorCreator;

struct GemmaMLP {
    gate_up_proj: MergedColumnParallelLinear<UnquantizedLinearWeights>,
    down_proj: RowParallelLinear<UnquantizedLinearWeights>,
}

impl GemmaMLP {
    pub fn forward<C: TensorCreator>(&self, x: &Tensor, tensor_creator: &mut C) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward_(x, tensor_creator)?;
        let x = vllm::gelu_and_mul_(&gate_up, tensor_creator)?;
        self.down_proj.forward_(&x, tensor_creator)
    }
    fn load(vb: VarBuilder, cfg: &Config, parallel_state: &ParallelState) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = MergedColumnParallelLinear::<UnquantizedLinearWeights>::new(
            h_size,
            &[i_size, i_size],
            parallel_state,
        )
        .load(&vb, &["gate_proj", "up_proj"], None, true)?;

        let down_proj =
            RowParallelLinear::<UnquantizedLinearWeights>::new(i_size, h_size, parallel_state)
                .load(&vb.pp("down_proj"), None)?;

        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }
}

struct GemmaAttention {
    qkv_proj: QKVParallelLinear<UnquantizedLinearWeights>,
    o_proj: RowParallelLinear<UnquantizedLinearWeights>,
    attn: PagedAttention,
    rotary_emb: RotaryEmbedding,
}

impl GemmaAttention {
    pub fn forward<C: TensorCreator>(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
        tensor_creator: &mut C,
    ) -> Result<Tensor> {
        let (q, k, v) = self.qkv_proj.forward_(hidden_states, tensor_creator)?;

        self.rotary_emb.forward(positions, &q, &k)?;

        let dtype = q.dtype();
        let attn_output = self.attn.forward(
            &q,
            &k,
            &v,
            cache.map(|(k, _)| k),
            cache.map(|(_, v)| v),
            input_metadata,
            dtype,
            false,
            tensor_creator,
        )?;

        self.o_proj.forward_(&attn_output, tensor_creator)
    }

    fn load(
        vb: VarBuilder,
        _cache: &Cache,
        cfg: &Config,
        parallel_state: &ParallelState,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let _num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let total_num_kv_heads = num_kv_heads;
        let total_num_heads = num_heads;

        let qkv_proj = QKVParallelLinear::<UnquantizedLinearWeights>::new(
            hidden_sz,
            head_dim,
            total_num_heads,
            Some(total_num_kv_heads),
            parallel_state,
        )
        .load(&vb, &["q_proj", "k_proj", "v_proj"], None, true)?;

        let o_proj = RowParallelLinear::<UnquantizedLinearWeights>::new(
            total_num_heads * head_dim,
            hidden_sz,
            parallel_state,
        )
        .load(&vb.pp("o_proj"), None)?;

        Ok(Self {
            qkv_proj,
            o_proj,
            attn: PagedAttention::new(
                vb.device(),
                num_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(num_kv_heads),
                None,
                None,
            )?,
            rotary_emb: RotaryEmbedding::new(
                vb.device(),
                vb.dtype(),
                head_dim,
                head_dim,
                cfg.max_position_embeddings,
                10000.0,
                true,
            )?,
        })
    }
}

struct GemmaDecoderLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMLP,
    input_layernorm: vllm::RmsNorm,
    post_attention_layernorm: vllm::RmsNorm,
    idx: usize,
}

impl GemmaDecoderLayer {
    fn forward_<C: TensorCreator>(
        &self,
        hidden_states: Tensor,
        positions: &Tensor,
        residual: Option<Tensor>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
        tensor_creator: &mut C,
    ) -> Result<(Tensor, Tensor)> {
        if self.idx == 0 {}
        let (hidden_states, residual) = match residual {
            Some(residual) => {
                self.input_layernorm
                    .forward_residual_(&hidden_states, &residual)?;
                (hidden_states, residual)
            }
            None => {
                let new_hidden_states =
                    self.input_layernorm
                        .forward_(&hidden_states, tensor_creator, self.idx == 0)?;
                let residual = hidden_states;
                (new_hidden_states, residual)
            }
        };
        if self.idx == 0 {}
        let hidden_states = self.self_attn.forward(
            positions,
            &hidden_states,
            cache,
            input_metadata,
            tensor_creator,
        )?;

        self.post_attention_layernorm
            .forward_residual_(&hidden_states, &residual)?;

        let hidden_states = self.mlp.forward(&hidden_states, tensor_creator)?;
        if self.idx == 0 {
            //tracing::info!("####after mlp, hidden_states:{}", hidden_states.to_string());
        }
        Ok((hidden_states, residual))
    }
    fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        idx: usize,
        parallel_state: &ParallelState,
    ) -> Result<Self> {
        let self_attn = GemmaAttention::load(vb.pp("self_attn"), cache, cfg, parallel_state)?;
        let mlp = GemmaMLP::load(vb.pp("mlp"), cfg, parallel_state)?;
        let input_layernorm = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            Some(1.0),
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            Some(1.0),
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            idx,
        })
    }
}

pub struct Gemma {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<GemmaDecoderLayer>,
    norm: vllm::RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    block_arena: [TensorArena; 2],
}

impl Gemma {
    pub fn do_forward(
        &mut self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let x = self.embed_tokens.forward(input_ids)?;
        let mut x = (x * (self.hidden_size as f64).sqrt())?;

        let mut residual: Option<Tensor> = None;
        if let Some(kv_caches) = kv_caches {
            for (idx, layer) in self.layers.iter().enumerate() {
                let arena_idx = idx % 2;
                self.block_arena[1 - arena_idx].reset();
                let (hidden_states, new_residual) = layer.forward_(
                    x,
                    positions,
                    residual,
                    Some((&kv_caches[idx].0, &kv_caches[idx].1)),
                    input_metadata,
                    &mut self.block_arena[arena_idx],
                )?;
                x = hidden_states;
                residual = Some(new_residual);
            }
        } else {
            for (idx, layer) in self.layers.iter().enumerate() {
                let arena_idx = idx % 2;
                self.block_arena[1 - arena_idx].reset();
                let (hidden_states, new_residual) = layer.forward_(
                    x,
                    positions,
                    residual,
                    None,
                    input_metadata,
                    &mut self.block_arena[arena_idx],
                )?;
                x = hidden_states;
                residual = Some(new_residual);
            }
        }

        self.norm
            .forward_residual_(&x, residual.as_ref().unwrap())?;

        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;

        Ok(logits)
    }

    pub fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        parallel_state: &ParallelState,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = GemmaDecoderLayer::load(
                vb.pp(&format!("model.layers.{layer_idx}")),
                cache,
                cfg,
                layer_idx,
                parallel_state,
            )?;
            layers.push(layer)
        }
        let norm = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            Some(1.0),
            vb_m.pp("norm"),
        )?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            block_arena: [TensorArena::new(vb.device()), TensorArena::new(vb.device())],
        })
    }
}

impl Model for Gemma {
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
