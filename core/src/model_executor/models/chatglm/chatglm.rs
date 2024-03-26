use std::rc::Rc;

use super::GLMConfig as Config;

use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{linear_b as linear, Linear, VarBuilder};

use crate::model_executor::input_metadata::InputMetadata;
use crate::model_executor::layers::{Cache, Layer, MergedColumnParallelLinear};
use crate::model_executor::layers::{
    LinearWeights, PagedAttention, QKVParallelLinear, RotaryEmbedding, RowParallelLinear,
};
use crate::model_executor::models::{Model, ModelConfig};
use crate::model_executor::parallel::ParallelState;
use crate::tensor::TensorArena;

use common::TensorCreator;

pub struct GLMAttention<W: LinearWeights> {
    query_key_value: QKVParallelLinear<W>,
    dense: RowParallelLinear<W>,
    attn: PagedAttention,
    rotary_emb: RotaryEmbedding,
}

impl<W: LinearWeights> GLMAttention<W> {
    fn load(
        vb: VarBuilder,
        cfg: &Config,
        parallel_state: &ParallelState,
        config: W::Config,
    ) -> candle::Result<Self> {
        let hidden_size = cfg.hidden_size;
        let tp_size = parallel_state.get_tensor_model_parallel_world_size();
        let total_num_heads = cfg.num_attention_heads;
        let num_heads = total_num_heads / tp_size;
        let _multi_query_attention = cfg.multi_query_attention;
        let total_num_kv_heads = if cfg.multi_query_attention {
            cfg.multi_query_group_num
        } else {
            cfg.num_attention_heads
        };
        let num_kv_heads = std::cmp::max(1, total_num_kv_heads / tp_size);
        let head_dim = cfg.hidden_size / total_num_heads;
        let _q_size = num_heads * head_dim;
        let _kv_size = num_kv_heads * head_dim;

        let scaling = 1. / ((head_dim as f32).sqrt());

        let query_key_value = QKVParallelLinear::<W>::new(
            hidden_size,
            head_dim,
            total_num_heads,
            Some(total_num_kv_heads),
            parallel_state,
        )
        .with_bias(true)
        .load(&vb, &["query_key_value"], config.clone(), false)?;

        let dense =
            RowParallelLinear::<W>::new(total_num_heads * head_dim, hidden_size, parallel_state)
                .load(&vb.pp("o_proj"), config)?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            query_key_value,
            dense,
            attn: PagedAttention::new(
                vb.device(),
                num_heads,
                head_dim,
                scaling,
                Some(num_kv_heads),
                None,
                None,
            )?,
            rotary_emb: RotaryEmbedding::new(
                vb.device(),
                vb.dtype(),
                head_dim,
                cfg.max_model_len(),
                head_dim / 2,
                10000.0,
                false,
            )?,
        })
    }

    fn forward_<F: TensorCreator>(
        &self,
        tensor_creator: &mut F,
        x: &Tensor,
        positions: &Tensor,
        input_metadata: &mut InputMetadata,
        cache: Option<(&Tensor, &Tensor)>,
        log_enable: bool,
    ) -> candle::Result<Tensor> {
        let (q, k, v) = self.query_key_value.forward_(x, tensor_creator)?;
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
            log_enable,
            tensor_creator,
        )?;
        self.dense.forward_(&attn_output, tensor_creator)
    }
}

pub struct GLMMLP<W: LinearWeights> {
    dense_h_to_4h: MergedColumnParallelLinear<W>,
    dense_4h_to_h: RowParallelLinear<W>,
}
impl<W: LinearWeights> GLMMLP<W> {
    fn load(
        vb: VarBuilder,
        cfg: &Config,
        parallel_state: &ParallelState,
        config: W::Config,
    ) -> candle::Result<Self> {
        let dense_h_to_4h = MergedColumnParallelLinear::<W>::new(
            cfg.hidden_size,
            &[cfg.ffn_hidden_size * 2],
            parallel_state,
        )
        .load(&vb, &["dense_h_to_4h"], config.clone(), false)?;
        let dense_4h_to_h =
            RowParallelLinear::<W>::new(cfg.ffn_hidden_size, cfg.hidden_size, parallel_state)
                .load(&vb.pp("dense_4h_to_h"), config)?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward_<F: TensorCreator>(
        &self,
        hidden_states: &Tensor,
        _log_enable: bool,
        tensor_creator: &mut F,
    ) -> Result<Tensor> {
        let intermediate_parallel = self.dense_h_to_4h.forward_(hidden_states, tensor_creator)?;
        let intermediate_parallel = vllm::silu_and_mul_(&intermediate_parallel, tensor_creator)?;
        self.dense_4h_to_h
            .forward_(&intermediate_parallel, tensor_creator)
    }
}

pub struct GLMBlock<W: LinearWeights> {
    self_attention: GLMAttention<W>,
    mlp: GLMMLP<W>,
    input_layernorm: vllm::RmsNorm,
    post_attention_layernorm: vllm::RmsNorm,
    apply_residual_connection_post_layernorm: bool,
    idx: usize,
}
impl<W: LinearWeights> GLMBlock<W> {
    fn load(
        vb: &VarBuilder,
        cfg: &Config,
        idx: usize,
        parallel_state: &ParallelState,
        config: W::Config,
    ) -> candle::Result<Self> {
        let apply_residual_connection_post_layernorm = cfg.apply_residual_connection_post_layernorm;
        let self_attention =
            GLMAttention::<W>::load(vb.pp("self_attention"), cfg, parallel_state, config.clone())?;
        let mlp = GLMMLP::<W>::load(vb.pp("mlp"), cfg, parallel_state, config.clone())?;
        let input_layernorm = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.layernorm_epsilon,
            None,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.layernorm_epsilon,
            None,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            apply_residual_connection_post_layernorm,
            idx,
        })
    }

    fn forward_<F: TensorCreator>(
        &self,
        hidden_states: Tensor,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
        tensor_creator: &mut F,
        log_enable: bool,
    ) -> Result<Tensor> {
        let layernorm_output =
            self.input_layernorm
                .forward_(&hidden_states, tensor_creator, log_enable)?;
        let attention_output = self.self_attention.forward_(
            tensor_creator,
            &layernorm_output,
            positions,
            input_metadata,
            cache,
            log_enable,
        )?;

        let residual = if self.apply_residual_connection_post_layernorm {
            layernorm_output
        } else {
            hidden_states
        };
        let layernorm_input = residual.add(&attention_output)?;
        let layernorm_output =
            self.post_attention_layernorm
                .forward_(&layernorm_input, tensor_creator, log_enable)?;
        let layernorm_output = Rc::new(layernorm_output);
        let layernorm_input = Rc::new(layernorm_input);

        let residual = if self.apply_residual_connection_post_layernorm {
            layernorm_output.clone()
        } else {
            layernorm_input.clone()
        };

        let output = self
            .mlp
            .forward_(layernorm_output.as_ref(), log_enable, tensor_creator)?
            .add(residual.as_ref())?;
        Ok(output)
    }
}

pub struct ChatGLMModel<W: LinearWeights> {
    embedding: candle_nn::Embedding,
    blocks: Vec<GLMBlock<W>>,
    final_layernorm: vllm::RmsNorm,
    output_layer: Linear,
    block_arena: [TensorArena; 2],
}
impl<W: LinearWeights> ChatGLMModel<W> {
    pub fn load(
        vb: VarBuilder,
        _cache: &Cache,
        cfg: &Config,
        parallel_state: &ParallelState,
        linear_config: W::Config,
    ) -> candle::Result<Self> {
        let vb_m = vb.pp("transformer");
        let embedding = candle_nn::embedding(
            cfg.padded_vocab_size,
            cfg.hidden_size,
            vb_m.pp("embedding").pp("word_embeddings"),
        )?;

        let blocks: Vec<_> = (0..cfg.num_layers)
            .map(|i| {
                GLMBlock::<W>::load(
                    &vb_m.pp(&format!("encoder.layers.{i}")),
                    cfg,
                    i,
                    parallel_state,
                    linear_config.clone(),
                )
                .unwrap()
            })
            .collect();
        let final_layernorm = vllm::RmsNorm::load(
            cfg.hidden_size,
            cfg.layernorm_epsilon,
            None,
            vb_m.pp("encoder.final_layernorm"),
        )?;
        let output_layer = linear(
            cfg.padded_vocab_size,
            cfg.hidden_size,
            false,
            vb_m.pp("output_layer"),
        )?;

        Ok(Self {
            embedding,
            blocks,
            output_layer,
            final_layernorm,
            block_arena: [TensorArena::new(vb.device()), TensorArena::new(vb.device())],
        })
    }
    pub fn do_forward(
        &mut self,
        x: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        let mut hidden_states = self.embedding.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for (idx, block) in self.blocks.iter().enumerate() {
                let arena_idx = idx % 2;
                self.block_arena[1 - arena_idx].reset();
                hidden_states = block.forward_(
                    hidden_states,
                    positions,
                    Some((&kv_caches[idx].0, &kv_caches[idx].1)),
                    input_metadata,
                    &mut self.block_arena[arena_idx],
                    idx == 0,
                )?;
            }
        } else {
            // self.block_arena[0].reset();
            for (idx, block) in self.blocks.iter().enumerate() {
                let arena_idx = idx % 2;
                self.block_arena[1 - arena_idx].reset();
                hidden_states = block.forward_(
                    hidden_states,
                    positions,
                    None,
                    input_metadata,
                    &mut self.block_arena[arena_idx],
                    idx == 0,
                )?;
            }
        }
        let output = self.final_layernorm.forward(&hidden_states)?;
        let x = output.i((.., seq_len - 1, ..))?;
        let logits = self.output_layer.forward(&x)?;
        Ok(logits)
    }
}
impl<W: LinearWeights> Model for ChatGLMModel<W> {
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
