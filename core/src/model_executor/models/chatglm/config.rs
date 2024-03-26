use anyhow::{anyhow, Result};
use candle::DType;

use crate::model_executor::models::{ModelConfig, ModelType};

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GLMConfig {
    // pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub num_layers: usize,
    pub padded_vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub kv_channels: usize,
    pub num_attention_heads: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
    pub rmsnorm: bool,
    pub apply_residual_connection_post_layernorm: bool,
    pub post_layer_norm: bool,
    pub add_bias_linear: bool,
    pub add_qkv_bias: bool,
    pub bias_dropout_fusion: bool,
    pub multi_query_attention: bool,
    pub multi_query_group_num: usize,
    pub apply_query_key_layer_scaling: bool,
    pub attention_softmax_in_fp32: bool,
    pub fp32_residual_connection: bool,
    pub torch_dtype: String,
}

impl GLMConfig {
    pub fn glm3_6b() -> Self {
        Self {
            // bos_token_id: 1,
            eos_token_id: 2,
            num_layers: 28,
            padded_vocab_size: 65024,
            hidden_size: 4096,
            ffn_hidden_size: 13696,
            kv_channels: 128,
            num_attention_heads: 32,
            seq_length: 8192,
            layernorm_epsilon: 1e-5,
            rmsnorm: true,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            add_bias_linear: false,
            add_qkv_bias: true,
            bias_dropout_fusion: true,
            multi_query_attention: true,
            multi_query_group_num: 2,
            apply_query_key_layer_scaling: true,
            attention_softmax_in_fp32: true,
            fp32_residual_connection: false,
            torch_dtype: String::from("float16"),
        }
    }
}

impl ModelConfig for GLMConfig {
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
        ModelType::CHATGLM
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn num_key_value_heads(&self) -> usize {
        if self.multi_query_attention {
            self.multi_query_group_num
        } else {
            self.num_attention_heads
        }
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn max_model_len(&self) -> usize {
        self.seq_length
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_layers
    }

    fn get_vocab_size(&self) -> usize {
        self.padded_vocab_size
    }
}
