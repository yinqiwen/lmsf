use anyhow::{anyhow, Result};
use candle::DType;

use crate::model_executor::models::{ModelConfig, ModelType};

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(crate) struct LlamaConfig {
    pub(crate) bos_token_id: u32,
    pub(crate) eos_token_id: u32,
    pub(crate) hidden_act: String,
    pub(crate) hidden_size: usize,
    pub(crate) initializer_range: f32,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) model_type: String,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_hidden_layers: usize,
    // pad_token_id: u32,
    pub(crate) rms_norm_eps: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) torch_dtype: String,
    pub(crate) use_cache: bool,
    pub(crate) vocab_size: usize,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
}
fn default_rope() -> f32 {
    10_000.0
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            bos_token_id: 1,
            eos_token_id: 2,
            hidden_act: String::from("silu"),
            hidden_size: Default::default(),
            initializer_range: 0.02,
            intermediate_size: 11008,
            max_position_embeddings: 4096,
            model_type: String::from("llama"),
            num_attention_heads: Default::default(),
            num_key_value_heads: Default::default(),
            num_hidden_layers: Default::default(),
            rms_norm_eps: 1e-05,
            tie_word_embeddings: false,
            torch_dtype: String::from("float16"),
            use_cache: true,
            vocab_size: 32000,
            rope_theta: 10_000.0,
        }
    }
}

impl ModelConfig for LlamaConfig {
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
            _ => Err(anyhow!("invalid dtype:{}", self.torch_dtype)),
        }
    }
    fn get_model_type(&self) -> ModelType {
        ModelType::LLAMA
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    // fn set_num_attention_heads(&self, v: usize) {
    //     self.num_attention_heads
    //         .store(v, std::sync::atomic::Ordering::SeqCst);
    // }
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }
    // fn set_num_key_value_heads(&self, v: usize) {
    //     self.num_key_value_heads
    //         .store(v, std::sync::atomic::Ordering::SeqCst);
    // }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    // fn set_hidden_size(&self, v: usize) {
    //     self.hidden_size
    //         .store(v, std::sync::atomic::Ordering::SeqCst);
    // }
    fn max_model_len(&self) -> usize {
        self.max_position_embeddings
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    // fn set_num_hidden_layers(&self, v: usize) {
    //     self.num_hidden_layers
    //         .store(v, std::sync::atomic::Ordering::SeqCst);
    // }
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
