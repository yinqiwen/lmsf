use candle::Tensor;

// use super::ops::AttentionBias;

pub struct InputMetadata {
    pub(crate) prompt_lens: Vec<usize>,
    pub(crate) slot_mapping: Tensor,
    pub(crate) max_context_len: Option<usize>,
    pub(crate) context_lens: Option<Tensor>,
    pub(crate) block_tables: Option<Tensor>,
    use_cuda_graph: bool,
    pub(crate) is_prompt: bool,
    // pub(crate) attn_bias: Option<Box<dyn AttentionBias>>,
}
impl InputMetadata {
    pub fn new(
        prompt_lens: Vec<usize>,
        slot_mapping: Tensor,
        max_context_len: Option<usize>,
        context_lens: Option<Tensor>,
        block_tables: Option<Tensor>,
        use_cuda_graph: bool,
    ) -> Self {
        let is_prompt = !prompt_lens.is_empty();
        Self {
            prompt_lens,
            slot_mapping,
            max_context_len,
            context_lens,
            block_tables,
            use_cuda_graph,
            is_prompt,
            // attn_bias: None,
        }
    }
}
