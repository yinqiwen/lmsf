use anyhow::Result;
use candle_core::DType;
use candle_core::Tensor;
use std::sync::Arc;

use crate::model_executor::input_metadata::InputMetadata;

pub trait ModelConfig: std::fmt::Debug + Send + Sync {
    fn get_dtype(&self) -> Result<DType>;
    fn get_model_type(&self) -> &str;
    fn num_attention_heads(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> usize {
        self.num_attention_heads()
    }
    fn max_model_len(&self) -> usize;
    fn get_sliding_window(&self) -> Option<usize> {
        None
    }
}

pub trait TokenizerConfig: std::fmt::Debug + Send + Sync {
    fn get_eos_token(&self) -> &str;
    fn get_chat_template(&self) -> Option<&str>;
}

pub trait Model: Send {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: InputMetadata,
    ) -> anyhow::Result<Tensor>;
}
