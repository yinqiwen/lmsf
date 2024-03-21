use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use candle::DType;
use clap::{Parser, ValueEnum};

use crate::model_executor::models::{
    ModelConfig as PretrainedModelConfig, ModelFactory, ModelType, QuantizeType,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct EngineArgs {
    #[clap(default_value = "4", long)]
    pub threads: usize,

    #[clap(
        default_value = "facebook/opt-125m",
        long,
        help = "path of the huggingface model to use"
    )]
    model: String,

    // #[clap(long, help = "specify model_type")]
    // model_type: Option<ModelType>,
    #[clap(long, help = "specify quantize method")]
    quantization: Option<QuantizeType>,

    #[clap(default_value = "", long, help = "chat template config file")]
    chat_template: Option<String>,

    // Parallel arguments
    #[clap(default_value = "1", long = "pp", help = "number of pipeline stages")]
    pipeline_parallel_size: usize,
    #[clap(
        default_value = "1",
        long = "tp",
        help = "number of tensor parallel replicas"
    )]
    tensor_parallel_size: usize,

    #[clap(long, help = "maximum number of batched tokens per iteration")]
    max_num_batched_tokens: Option<usize>,

    #[clap(
        default_value = "256",
        long,
        help = "maximum number of sequences per iteration"
    )]
    max_num_seqs: usize,

    #[clap(
        default_value = "256",
        long,
        help = "maximum number of paddings in a batch"
    )]
    max_paddings: usize,

    #[clap(
        long,
        help = "model context length. If unspecified, will be automatically derived from the model"
    )]
    max_model_len: Option<usize>,

    // kv cache
    #[clap(default_value = "16", long, help = "token block size")]
    block_size: usize,

    #[clap(default_value = "0", long, help = "random seed")]
    seed: u64,
    #[clap(default_value = "4", long, help = "CPU swap space size (GiB) per GPU")]
    swap_space: usize,

    #[clap(
        default_value = "0.9",
        long,
        help = "'the percentage of GPU memory to be used for the model executor"
    )]
    gpu_memory_utilization: f32,

    #[clap(long, help = "log dir")]
    pub log_dir: Option<String>,

    #[clap(default_value = "false", long, help = "print log into stderr")]
    pub alsologtostderr: bool,
}

impl EngineArgs {
    pub fn create_engine_configs(
        &self,
    ) -> Result<(ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig)> {
        let model_cfg = ModelConfig::new(
            self.model.as_str(),
            self.quantization,
            self.max_model_len,
            self.seed,
        )?;
        tracing::info!("{:?}", model_cfg);
        let parallel_cfg = ParallelConfig {
            pipeline_parallel_size: self.pipeline_parallel_size,
            tensor_parallel_size: self.tensor_parallel_size,
        };
        let scheduler_cfg = SchedulerConfig::new(
            self.max_num_batched_tokens,
            self.max_num_seqs,
            model_cfg.get_max_model_len(),
            self.max_paddings,
        );

        let cache_cfg = CacheConfig::new(
            self.block_size,
            self.gpu_memory_utilization,
            self.swap_space,
            model_cfg.get_sliding_window(),
        )?;
        Ok((model_cfg, cache_cfg, parallel_cfg, scheduler_cfg))
    }
}

#[derive(Debug)]
pub struct ModelConfig {
    cfg: Box<dyn PretrainedModelConfig>,
    quantize_type: Option<QuantizeType>,
    model_weight_files: Vec<String>,
    path: String,
    dtype: DType,
    max_model_len: usize,
    pub(crate) seed: u64,
}

impl ModelConfig {
    pub fn new(
        model_path: &str,
        quantize_type: Option<QuantizeType>,
        max_model_len: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        let model_weight_files = ModelFactory::get_model_weight_files(model_path, quantize_type)?;

        let modle_cfg = ModelFactory::new_model_config(model_path, quantize_type)?;

        let dtype = modle_cfg.get_dtype()?;
        let max_len = if let Some(n) = max_model_len {
            if n > modle_cfg.max_model_len() {
                return Err(
                    anyhow!("User-specified max_model_len ({}) is greater than the derived max_model_len:{} in model's config.json). This may lead to incorrect model
                outputs or CUDA errors. Make sure the value is correct and
                within the model context size.",n, modle_cfg.max_model_len()),
                );
            }
            n
        } else {
            modle_cfg.max_model_len()
        };

        Ok(Self {
            cfg: modle_cfg,
            quantize_type,
            model_weight_files,
            path: String::from(model_path),
            dtype,
            max_model_len: max_len,
            seed,
        })
    }

    pub fn inner(&self) -> &dyn PretrainedModelConfig {
        self.cfg.as_ref()
    }

    pub fn path(&self) -> &str {
        self.path.as_str()
    }

    pub fn get_model_weight_files(&self) -> &Vec<String> {
        &self.model_weight_files
    }
    pub fn get_dtype(&self) -> DType {
        self.dtype
    }
    pub fn get_max_model_len(&self) -> usize {
        self.max_model_len
    }

    pub fn head_size(&self) -> usize {
        self.cfg.hidden_size() / self.cfg.num_attention_heads()
    }
    pub fn num_hidden_layers(&self) -> usize {
        self.cfg.num_hidden_layers()
    }

    pub fn get_num_layers(&self, parallel_config: &ParallelConfig) -> usize {
        let total_num_hidden_layers = self.num_hidden_layers();
        total_num_hidden_layers / parallel_config.pipeline_parallel_size
    }
    pub fn get_num_kv_heads(&self, parallel_config: &ParallelConfig) -> usize {
        let total_num_kv_heads: usize = self.cfg.num_key_value_heads();
        (total_num_kv_heads / parallel_config.tensor_parallel_size).max(1)
    }
    pub fn get_sliding_window(&self) -> Option<usize> {
        self.cfg.get_sliding_window()
    }

    pub fn get_vocab_size(&self) -> usize {
        self.cfg.get_vocab_size()
    }

    pub fn get_bos_token_id(&self) -> u32 {
        self.cfg.get_bos_token_id()
    }
    pub fn get_eos_token_id(&self) -> u32 {
        self.cfg.get_eos_token_id()
    }
    pub fn get_model_type(&self) -> ModelType {
        self.cfg.get_model_type()
    }
    pub fn get_quantize_type(&self) -> Option<QuantizeType> {
        self.quantize_type
    }
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub(crate) block_size: usize,
    pub(crate) gpu_memory_utilization: f32,
    pub(crate) swap_space_bytes: usize,
    pub(crate) sliding_window: Option<usize>,
}

impl CacheConfig {
    pub fn new(
        block_size: usize,
        gpu_memory_utilization: f32,
        swap_space_gb: usize,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        if gpu_memory_utilization > 1.0 {
            return Err(anyhow!(
                "GPU memory utilization must be less than 1.0. Got {}",
                gpu_memory_utilization
            ));
        }
        Ok(Self {
            block_size,
            gpu_memory_utilization,
            swap_space_bytes: swap_space_gb * 1024 * 1024 * 1024,
            sliding_window,
        })
    }

    pub fn verify_with_parallel_config(&self, parallel_config: &ParallelConfig) -> Result<()> {
        //total_cpu_memory = get_cpu_memory();

        // let num_gpus_per_node = parallel_config.tensor_parallel_size;
        // let cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node;

        // msg = (f"{cpu_memory_usage / _GB:.2f} GiB out of "
        //        f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
        //        "allocated for the swap space.")
        // if cpu_memory_usage > 0.7 * total_cpu_memory:
        //     raise ValueError("Too large swap space. " + msg)
        // elif cpu_memory_usage > 0.4 * total_cpu_memory:
        //     logger.warning("Possibly too large swap space. " + msg)
        Ok(())
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size
    }
}

#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pipeline_parallel_size: usize,
    tensor_parallel_size: usize,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub(crate) max_num_batched_tokens: usize,
    pub(crate) max_num_seqs: usize,
    pub(crate) max_model_len: usize,
    pub(crate) max_paddings: usize,
}

impl SchedulerConfig {
    pub fn new(
        max_num_batched_tokens: Option<usize>,
        max_num_seqs: usize,
        max_model_len: usize,
        max_paddings: usize,
    ) -> Self {
        let max_num_batched_tokens = if let Some(max_num_batched_tokens) = max_num_batched_tokens {
            max_num_batched_tokens
        } else {
            // # If max_model_len is too short, use 2048 as the default value for
            // # higher throughput.
            std::cmp::max(max_model_len, 2048)
        };
        Self {
            max_num_batched_tokens,
            max_num_seqs,
            max_model_len,
            max_paddings,
        }
    }
    // def _verify_args(self) -> None:
    // if self.max_num_batched_tokens < self.max_model_len:
    //     raise ValueError(
    //         f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
    //         f"smaller than max_model_len ({self.max_model_len}). "
    //         "This effectively limits the maximum sequence length to "
    //         "max_num_batched_tokens and makes vLLM reject longer "
    //         "sequences. Please increase max_num_batched_tokens or "
    //         "decrease max_model_len.")
    // if self.max_num_batched_tokens < self.max_num_seqs:
    //     raise ValueError(
    //         f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
    //         "be greater than or equal to max_num_seqs "
    //         f"({self.max_num_seqs}).")
}

// pub struct TokenizerConfig {
//     pub eos_token_id: u32,
// }

// impl TokenizerConfig {
//     pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
//         let content = read_to_string(file)?;
//         let config = serde_json::from_str(&content)?;
//         Ok(config)
//     }
// }
