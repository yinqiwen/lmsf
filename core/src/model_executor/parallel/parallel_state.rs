pub struct ParallelState {
    tensor_model_parallel_size: usize,
    pipeline_model_parallel_size: usize,
    tensor_model_parallel_rank: usize,
}

impl Default for ParallelState {
    fn default() -> Self {
        Self {
            tensor_model_parallel_size: 1,
            pipeline_model_parallel_size: 1,
            tensor_model_parallel_rank: 0,
        }
    }
}

impl ParallelState {
    pub fn get_tensor_model_parallel_world_size(&self) -> usize {
        self.tensor_model_parallel_size
    }
    pub fn get_pipeline_model_parallel_size(&self) -> usize {
        self.pipeline_model_parallel_size
    }

    pub fn get_tensor_model_parallel_rank(&self) -> usize {
        self.tensor_model_parallel_rank
    }
}
