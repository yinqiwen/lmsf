use anyhow::{anyhow, Result};
use candle_core::DType;
use candle_core::Tensor;

use crate::model_executor::input_metadata::InputMetadata;

use super::llama::Llama;
use super::model::Model;
use super::model::PretrainedModelConfig;
use super::model::TokenizerConfig;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct LlamaEosTokenConfig {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
}
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(crate) struct LlamaTokenizerConfig {
    eos_token: LlamaEosTokenConfig,
}
impl TokenizerConfig for LlamaTokenizerConfig {
    fn get_eos_token(&self) -> &str {
        &self.eos_token.content
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(crate) struct LlamaConfig {
    bos_token_id: u32,
    eos_token_id: u32,
    hidden_act: String,
    hidden_size: usize,
    initializer_range: f32,
    intermediate_size: u32,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    // pad_token_id: u32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    torch_dtype: String,
    use_cache: bool,
    vocab_size: u32,
}

impl PretrainedModelConfig for LlamaConfig {
    fn get_dtype(&self) -> Result<DType> {
        match self.torch_dtype.as_str() {
            "float16" | "half" => Ok(DType::F16),
            "float32" => Ok(DType::F32),
            _ => Err(anyhow!("invalid dtype:{}", self.torch_dtype)),
        }
    }
    fn get_model_type(&self) -> &str {
        &self.model_type
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn max_model_len(&self) -> usize {
        self.max_position_embeddings
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
}

pub(crate) struct LlamaModel {
    pub(crate) llama: Llama,
}

impl Model for LlamaModel {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        mut input_metadata: InputMetadata,
    ) -> anyhow::Result<candle_core::Tensor> {
        self.llama
            .forward(
                &input_tokens,
                &input_positions,
                kv_cache,
                &mut input_metadata,
            )
            .map_err(|e| anyhow!("{}", e))
    }
}
