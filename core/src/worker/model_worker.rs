use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};

use crate::common::config::{CacheConfig, ModelConfig, ParallelConfig};
use crate::common::sampling_params::SamplingType;
use crate::common::sequence::{SamplerOutput, SequenceGroup, SequenceGroupMetadata, SequenceState};
use crate::model_executor::input_metadata::InputMetadata;
use crate::model_executor::layers::Sampler;
use crate::model_executor::models::{Model, ModelFactory};
use crate::model_executor::ops::tensor::make_tensor_with_pad;
use crate::model_executor::sampling_metadata::SamplingMetadata;
use crate::sched::scheduler::SchedulerOutputs;
use crate::SamplingParams;

use std::collections::HashMap;
use std::sync::Arc;

use super::cache_engine::CacheEngine;

const _PAD_SLOT_ID: i64 = -1;
pub struct Worker {
    device: candle_core::Device,
    cache_config: CacheConfig,
    model_config: Arc<ModelConfig>,
    cache_engine: CacheEngine,
    model: Box<dyn Model>,
    sampler: Sampler,
    block_size: usize,
}

impl Worker {
    pub fn from(
        cache_config: &CacheConfig,
        model_config: Arc<ModelConfig>,
        parallel_config: &ParallelConfig,
        rank: usize,
    ) -> Result<Self> {
        let device = candle_core::Device::new_cuda(rank)?;
        let mut filenames: Vec<String> = vec![];
        for rfilename in model_config.get_safetensors() {
            let path = format!("{}/{}", model_config.dir(), rfilename);
            filenames.push(path);
        }
        tracing::info!("files:{:?}", filenames);
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &filenames,
                model_config.get_dtype(),
                &device,
            )?
        };
        let cuda_device = if let Device::Cuda(cuda_dev) = &device {
            cuda_dev
        } else {
            return Err(anyhow!("invalid device"));
        };
        // let cache_ops = CacheOps::new(cuda_device)?;
        //CacheEngine::new(cache_config, model_config.clone(), parallel_config.clone())?;
        let model = ModelFactory::load_model("llama", model_config.inner(), vb)?;
        let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
        tracing::info!("GPU free:{}, total:{} after load model.", free, total);

        let sampler = Sampler::new(
            256,
            model_config.get_max_model_len(),
            model_config.get_dtype(),
            &device,
        )?;
        let cache_engine = CacheEngine::new(
            cuda_device,
            cache_config,
            model_config.clone(),
            parallel_config,
            // cache_ops,
        )?;

        Ok(Self {
            device,
            cache_config: cache_config.clone(),
            model_config,
            cache_engine,
            model,
            sampler,
            block_size: 0,
        })
    }

    pub fn get_num_gpu_blocks(&self) -> usize {
        self.cache_engine.get_num_gpu_blocks()
    }
    pub fn get_num_cpu_blocks(&self) -> usize {
        self.cache_engine.get_num_cpu_blocks()
    }
    fn set_block_size(&mut self, block_size: usize) {
        self.block_size = block_size;
        //         max_num_blocks = (self.max_context_len_to_capture + block_size -
        //             1) // block_size
        // self.graph_block_tables = np.zeros(
        // (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)
    }

    pub fn init_cache(&mut self) -> Result<()> {
        let (num_gpu_blocks, num_cpu_blocks) = self.profile_num_available_blocks(
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.swap_space_bytes,
        )?;

        let max_seq_len = self.cache_config.block_size * num_gpu_blocks;
        if self.model_config.get_max_model_len() > max_seq_len {
            return Err(anyhow!("The model's max seq len ({}) is larger than the maximum number of tokens that can be stored in KV cache ({}). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.",self.model_config.get_max_model_len(),max_seq_len));
        }

        self.cache_engine.init(num_gpu_blocks, num_cpu_blocks)?;
        // self.device.synchronize()?;
        self.set_block_size(self.cache_config.block_size);
        let (free, _) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
        tracing::info!("After init_cache, GPU free:{}bytes", free);
        Ok(())
    }

    fn profile_num_available_blocks(
        &self,
        block_size: usize,
        gpu_memory_utilization: f32,
        cpu_swap_space: usize,
    ) -> Result<(usize, usize)> {
        // # Profile the memory usage of the model and get the maximum number of
        // # cache blocks that can be allocated with the remaining free memory.
        // torch.cuda.empty_cache()

        // # Execute a forward pass with dummy inputs to profile the memory usage
        // # of the model.
        // self.model_runner.profile_run()

        // # Calculate the number of blocks that can be allocated with the
        // # profiled peak memory.
        // torch.cuda.synchronize()
        // self.device.synchronize()?;
        let (free, total_gpu_memory) =
            candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
        let peak_memory = total_gpu_memory - free;
        let cache_block_size = self.cache_engine.get_cache_block_size(block_size);

        let num_gpu_blocks = (total_gpu_memory as f64 * gpu_memory_utilization as f64
            - peak_memory as f64) as usize
            / cache_block_size;
        let num_cpu_blocks = cpu_swap_space / cache_block_size;
        tracing::info!(
            "Cache init with cache_block_size:{},num_cpu_blocks:{},num_gpu_blocks:{}",
            cache_block_size,
            num_cpu_blocks,
            num_gpu_blocks
        );
        Ok((num_gpu_blocks, num_cpu_blocks))
    }

    pub fn execute_model(
        &mut self,
        seq_group_metadata_list: Vec<SequenceGroupMetadata>,
        sched_output: &SchedulerOutputs,
    ) -> Result<Option<SamplerOutput>> {
        if !sched_output.blocks_to_swap_in.is_empty() {
            self.cache_engine.swap_in(&sched_output.blocks_to_swap_in)?;
        }
        if !sched_output.blocks_to_swap_out.is_empty() {
            self.cache_engine
                .swap_out(&sched_output.blocks_to_swap_out)?;
        }
        if !sched_output.blocks_to_copy.is_empty() {
            self.cache_engine.copy(&sched_output.blocks_to_copy)?;
        }
        let prepare_start = std::time::Instant::now();
        let (input_tokens, input_positions, input_metadata) = if sched_output.prompt_run {
            self.prepare_prompt(&seq_group_metadata_list)?
        } else {
            self.prepare_decode(&seq_group_metadata_list)?
        };
        //tracing::info!("prepare cost {:?}", prepare_start.elapsed());
        let sampling_metadata =
            self.prepare_sample(&seq_group_metadata_list, &input_metadata.prompt_lens)?;

        let logits = self.model.forward(
            input_tokens,
            input_positions,
            Some(self.cache_engine.get_gpu_cache()),
            input_metadata,
        )?;
        //tracing::info!("modle forward cost {:?}", start.elapsed());
        //let cpu = Device::Cpu;
        // tracing::info!("logits data:{:?}", logits.to_device(&cpu)?.to_string());
        //Ok(logits)
        match logits.device() {
            Device::Cuda(cuda) => {
                cuda.synchronize()?;
                tracing::info!("modle forward cost {:?}", prepare_start.elapsed());
            }
            _ => {
                //
            }
        }
        self.sampler
            .forward(logits, sampling_metadata)
            .map_err(|e| anyhow!("{}", e))
    }
    fn prepare_prompt(
        &self,
        seq_group_metadata_list: &[SequenceGroupMetadata],
    ) -> Result<(Tensor, Tensor, InputMetadata)> {
        let mut prompt_lens = Vec::new();
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut slot_mapping = Vec::new();
        for seq_group in seq_group_metadata_list.iter() {
            for seq in seq_group.seq_data.values() {
                let prompt_len = seq.borrow().get_prompt_len();
                prompt_lens.push(prompt_len);
                input_tokens.push(seq.borrow().get_token_ids());
                // # NOTE(woosuk): Here we assume that the first token in the prompt
                // # is always the first token in the sequence.
                input_positions.push((0..prompt_len as i64).collect::<Vec<_>>());

                if seq_group.block_tables.is_empty() {
                    slot_mapping.push([_PAD_SLOT_ID].repeat(prompt_len));
                    continue;
                }
                // # Compute the slot mapping.
                //slot_mapping.push(Vec::new());
                let mut seq_slot_mapping = Vec::new();
                let block_table = seq_group.block_tables.get(&seq.borrow().seq_id).unwrap();
                // # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
                // # where start_idx is max(0, prompt_len - sliding_window).
                // # For example, if the prompt len is 10, sliding window is 8, and
                // # block size is 4, the first two tokens are masked and the slot
                // # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
                let start_idx = if let Some(sliding_window) = self.model_config.get_sliding_window()
                {
                    //0.min(prompt_len - sliding_window)
                    std::cmp::max(0, prompt_len as i64 - sliding_window as i64)
                } else {
                    0
                } as usize;

                for i in 0..prompt_len {
                    if i < start_idx {
                        seq_slot_mapping.push(_PAD_SLOT_ID);
                        continue;
                    }
                    let block_number = block_table[i / self.block_size];
                    let block_offset = i % self.block_size;
                    let slot = block_number as usize * self.block_size + block_offset;
                    seq_slot_mapping.push(slot as i64);
                }
                slot_mapping.push(seq_slot_mapping);
            }
        }
        let max_prompt_len = prompt_lens.iter().max().unwrap();
        let input_tokens = make_tensor_with_pad(
            &self.device,
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
        )?;
        let input_positions =
            make_tensor_with_pad(&self.device, input_positions, *max_prompt_len, 0)?;
        let slot_mapping_tensor =
            make_tensor_with_pad(&self.device, slot_mapping, *max_prompt_len, _PAD_SLOT_ID)?;
        Ok((
            input_tokens,
            input_positions,
            InputMetadata::new(prompt_lens, slot_mapping_tensor, None, None, None, false),
        ))
    }
    fn prepare_decode(
        &self,
        seq_group_metadata_list: &[SequenceGroupMetadata],
    ) -> Result<(Tensor, Tensor, InputMetadata)> {
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut context_lens = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut decode_block_tables = Vec::new();

        for seq_group in seq_group_metadata_list.iter() {
            for seq in seq_group.seq_data.values() {
                let generation_token = seq.borrow().get_last_token_id();
                input_tokens.push(vec![generation_token as i64]);
                let seq_len = seq.borrow().get_len();
                let position = seq_len - 1;
                input_positions.push(vec![position as i64]);

                let context_len =
                    if let Some(sliding_window) = self.model_config.get_sliding_window() {
                        std::cmp::min(seq_len, sliding_window)
                    } else {
                        seq_len
                    };
                context_lens.push(context_len as u32);

                let block_table = seq_group.block_tables.get(&seq.borrow().seq_id).unwrap();
                let block_number = block_table[position / self.block_size];
                let block_offset = position % self.block_size;
                let slot = block_number as usize * self.block_size + block_offset;
                slot_mapping.push(vec![slot as i64]);

                if let Some(sliding_window) = self.model_config.get_sliding_window() {
                    let sliding_window_blocks = sliding_window / self.block_size;
                    decode_block_tables.push(
                        block_table
                            .get(block_table.len() - sliding_window_blocks..)
                            .unwrap()
                            .to_vec(),
                    );
                } else {
                    decode_block_tables.push(block_table.clone());
                }
            }
        }

        let batch_size = input_tokens.len();
        let max_context_len = *context_lens.iter().max().unwrap();

        let input_tokens_tensor = make_tensor_with_pad(&self.device, input_tokens, 1, 0)?;
        let input_positions_tensor = make_tensor_with_pad(&self.device, input_positions, 1, 0)?;
        let slot_mapping_tensor =
            make_tensor_with_pad(&self.device, slot_mapping, 1, _PAD_SLOT_ID)?;
        let context_lens_size = context_lens.len();
        let context_lens_tensor =
            Tensor::from_vec(context_lens, (context_lens_size,), &self.device)?;

        let max_block_table_len = decode_block_tables.iter().map(|x| x.len()).max().unwrap();
        let decode_block_tables_tensor = make_tensor_with_pad(
            &self.device,
            decode_block_tables,
            max_context_len as usize,
            0,
        )?;

        //tracing::info!("prepare_decode result: slot_mapping_tensor:{},max_context_len:{},context_lens_tensor:{},block_table:{} ",slot_mapping_tensor.to_string(),max_context_len,context_lens_tensor.to_string(),decode_block_tables_tensor.to_string());
        Ok((
            input_tokens_tensor,
            input_positions_tensor,
            InputMetadata::new(
                vec![],
                slot_mapping_tensor,
                Some(max_context_len as usize),
                Some(context_lens_tensor),
                Some(decode_block_tables_tensor),
                false,
            ),
        ))
    }

    fn prepare_sample(
        &self,
        seq_group_metadata_list: &[SequenceGroupMetadata],
        prompt_lens: &[usize],
    ) -> Result<SamplingMetadata> {
        let mut seq_groups: Vec<(Vec<u64>, SamplingParams)> = Vec::new();
        let mut categorized_sample_indices_start_idx: u32 = 0;
        let mut selected_token_start_idx: i64 = 0;
        let mut categorized_sample_indices: Vec<Vec<u32>> = Vec::new();
        let mut selected_token_indices: Vec<i64> = Vec::new();
        for _ in 0..SamplingType::Beam as usize {
            categorized_sample_indices.push(Vec::new());
        }
        let max_prompt_len = if let Some(n) = prompt_lens.iter().max() {
            *n
        } else {
            1
        };
        for (idx, seq_group_metadata) in seq_group_metadata_list.iter().enumerate() {
            let seq_ids: Vec<u64> = seq_group_metadata.seq_data.keys().cloned().collect();
            let num_seqs = seq_ids.len();
            seq_groups.push((seq_ids, seq_group_metadata.sampling_params.clone()));
            let sampling_type = seq_group_metadata.sampling_params.sampling_type();
            if seq_group_metadata.is_prompt {
                assert!(num_seqs == 1);
                let prompt_len = prompt_lens[idx];
                if seq_group_metadata.sampling_params.prompt_logprobs.is_some() {
                    categorized_sample_indices_start_idx += prompt_len as u32 - 1
                }
                categorized_sample_indices[sampling_type as usize]
                    .push(categorized_sample_indices_start_idx);
                categorized_sample_indices_start_idx += 1;

                if seq_group_metadata.sampling_params.prompt_logprobs.is_some() {
                    let range =
                        selected_token_start_idx..selected_token_start_idx + prompt_len as i64 - 1;
                    let range_vec: Vec<i64> = range.collect();
                    selected_token_indices.extend(range_vec);
                }
                selected_token_indices.push(selected_token_start_idx + prompt_len as i64 - 1);
                selected_token_start_idx += max_prompt_len as i64;
            } else {
                let range = selected_token_start_idx..selected_token_start_idx + num_seqs as i64;
                let range_vec: Vec<i64> = range.collect();
                selected_token_indices.extend(range_vec);
                selected_token_start_idx += num_seqs as i64;

                let range = categorized_sample_indices_start_idx
                    ..categorized_sample_indices_start_idx + num_seqs as u32;
                let range_vec: Vec<u32> = range.collect();
                categorized_sample_indices[sampling_type as usize].extend(range_vec);
                categorized_sample_indices_start_idx += num_seqs as u32;
            }
        }
        let selected_token_indices = Tensor::new(selected_token_indices, &self.device)?;
        let mut categorized_sample_indices_map: HashMap<SamplingType, Tensor> = HashMap::new();
        for (t, seq_ids) in categorized_sample_indices.into_iter().enumerate() {
            if !seq_ids.is_empty() {
                let seq_ids_tensor = Tensor::new(seq_ids, &self.device)?;
                let sample_type = SamplingType::from_int(t).unwrap();
                categorized_sample_indices_map.insert(sample_type, seq_ids_tensor);
            }
        }

        let mut seq_data = HashMap::new();
        for seq_group_metadata in seq_group_metadata_list {
            seq_data.extend(seq_group_metadata.seq_data.clone());
        }

        let sample_meta = SamplingMetadata::new(
            seq_groups,
            seq_data,
            prompt_lens,
            selected_token_indices,
            categorized_sample_indices_map,
            true,
        );
        Ok(sample_meta)
    }
}
