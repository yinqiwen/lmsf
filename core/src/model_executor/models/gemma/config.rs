use anyhow::{anyhow, Result};
use candle::DType;

use crate::model_executor::models::{ModelConfig, ModelType};

#[derive(Debug, serde::Deserialize, Clone)]
pub(crate) struct GemmaConfig {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub torch_dtype: String,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}
fn default_max_position_embeddings() -> usize {
    4096
}

impl ModelConfig for GemmaConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    // fn get_bos_token_id(&self) -> u32 {
    //     self.bos_token_id
    // }
    fn get_eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
    fn get_dtype(&self) -> Result<DType> {
        match self.torch_dtype.as_str() {
            "float16" | "half" => Ok(DType::F16),
            "float32" => Ok(DType::F32),
            "bfloat16" => Ok(DType::BF16),
            _ => Err(anyhow!("invalid dtype:{}", self.torch_dtype)),
        }
    }
    fn get_model_type(&self) -> ModelType {
        ModelType::GEMMA
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
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

    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
