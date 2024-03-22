use anyhow::Result;
use candle::DType;
use candle::Tensor;


use super::ModelType;
use crate::model_executor::input_metadata::InputMetadata;

pub trait ModelConfig: std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn get_dtype(&self) -> Result<DType>;
    fn get_model_type(&self) -> ModelType;
    fn num_attention_heads(&self) -> usize;
    // fn set_num_attention_heads(&self, v: usize) {
    //     unimplemented!("set_num_attention_heads unimplemented")
    // }
    fn hidden_size(&self) -> usize;
    // fn set_hidden_size(&self, v: usize) {
    //     unimplemented!("set_hidden_size unimplemented")
    // }
    fn num_hidden_layers(&self) -> usize;
    // fn set_num_hidden_layers(&self, v: usize) {
    //     unimplemented!("set_num_hidden_layers unimplemented")
    // }
    fn num_key_value_heads(&self) -> usize {
        self.num_attention_heads()
    }
    // fn set_num_key_value_heads(&self, v: usize) {
    //     unimplemented!("set_num_key_value_heads unimplemented")
    // }
    fn max_model_len(&self) -> usize;
    fn get_sliding_window(&self) -> Option<usize> {
        None
    }
    fn get_vocab_size(&self) -> usize;
    fn get_bos_token_id(&self) -> u32;
    fn get_eos_token_id(&self) -> u32;
}

// pub trait TokenizerConfig: std::fmt::Debug + Send + Sync {
//     fn get_eos_token(&self) -> &str;
//     fn get_chat_template(&self) -> Option<&str>;
// }

pub trait Model {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: InputMetadata,
    ) -> anyhow::Result<Tensor>;
}
