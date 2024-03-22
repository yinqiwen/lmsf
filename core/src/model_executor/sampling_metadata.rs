use candle::{DType, Device, Tensor};
use std::{collections::HashMap, sync::Arc};

use crate::common::{
    sampling_params::{SamplingParams, SamplingType},
    sequence::SequenceDataRef,
};
use common::TensorCreator;

#[allow(dead_code)]
pub struct SamplingMetadata {
    pub(crate) seq_groups: Vec<(Vec<u64>, Arc<SamplingParams>)>,
    pub(crate) seq_data: HashMap<u64, SequenceDataRef>,
    pub(crate) prompt_lens: Vec<usize>,
    pub(crate) selected_token_indices: Tensor,
    pub(crate) categorized_sample_indices: HashMap<SamplingType, Tensor>,
    pub(crate) perform_sampling: bool,
    pub(crate) num_prompts: usize,
}
impl SamplingMetadata {
    pub fn new(
        seq_groups: Vec<(Vec<u64>, Arc<SamplingParams>)>,
        seq_data: HashMap<u64, SequenceDataRef>,
        prompt_lens: &[usize],
        selected_token_indices: Tensor,
        categorized_sample_indices: HashMap<SamplingType, Tensor>,
        perform_sampling: bool,
    ) -> Self {
        let num_prompts = prompt_lens.len();
        Self {
            seq_groups,
            seq_data,
            prompt_lens: Vec::from(prompt_lens),
            selected_token_indices,
            categorized_sample_indices,
            perform_sampling,
            num_prompts,
        }
    }
}
const _SAMPLING_EPS: f32 = 1e-5;

pub struct SamplingTensors {
    pub(crate) temperatures: Tensor,
    pub(crate) top_ps: Tensor,
    pub(crate) top_ks: Tensor,
    pub(crate) min_ps: Tensor,
    pub(crate) presence_penalties: Tensor,
    pub(crate) frequency_penalties: Tensor,
    pub(crate) repetition_penalties: Tensor,
    pub(crate) prompt_tokens: Tensor,
    pub(crate) output_tokens: Tensor,
}

impl SamplingTensors {
    pub fn from_sampling_metadata<F: TensorCreator>(
        sampling_metadata: &SamplingMetadata,
        vocab_size: usize,
        device: &Device,
        dtype: DType,
        tensor_creator: &mut F,
    ) -> candle::Result<(Self, bool, bool, bool)> {
        let mut prompt_tokens: Vec<Vec<u32>> = Vec::new();
        let mut output_tokens: Vec<Vec<u32>> = Vec::new();
        let mut top_ks: Vec<u32> = Vec::new();
        let mut temperatures: Vec<f32> = Vec::new();
        let mut top_ps: Vec<f32> = Vec::new();
        let mut min_ps: Vec<f32> = Vec::new();
        let mut presence_penalties: Vec<f32> = Vec::new();
        let mut frequency_penalties: Vec<f32> = Vec::new();
        let mut repetition_penalties: Vec<f32> = Vec::new();
        let mut do_penalties = false;
        let mut do_top_p_top_k = false;
        let mut do_min_p = false;
        //let start = std::time::Instant::now();
        for (i, (seq_ids, sampling_params)) in sampling_metadata.seq_groups.iter().enumerate() {
            let mut temperature = sampling_params.temperature;
            let p = sampling_params.presence_penalty;
            let f = sampling_params.frequency_penalty;
            let r = sampling_params.repetition_penalty;
            let top_p = sampling_params.top_p;
            let min_p = sampling_params.min_p;

            // # k should not be greater than the vocab size.
            let mut top_k = std::cmp::min(sampling_params.top_k, vocab_size as i32);
            if top_k == -1 {
                top_k = vocab_size as i32;
            }

            if temperature < _SAMPLING_EPS {
                // # NOTE: Zero temperature means deterministic sampling
                // # (i.e., greedy sampling or beam search).
                // # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            }
            if !do_top_p_top_k && (top_p < 1.0 - _SAMPLING_EPS || top_k != vocab_size as i32) {
                do_top_p_top_k = true;
            }
            if !do_min_p && min_p > _SAMPLING_EPS {
                do_min_p = true;
            }
            if !do_penalties
                && (p.abs() >= _SAMPLING_EPS
                    || f.abs() >= _SAMPLING_EPS
                    || (r - 1.0).abs() >= _SAMPLING_EPS)
            {
                do_penalties = true;
            }
            if i < sampling_metadata.num_prompts && sampling_params.prompt_logprobs.is_some() {
                // For tokens in the prompt that we only need to get their logprobs
                let prompt_len = sampling_metadata.prompt_lens[i];
                temperatures.extend_from_slice(&[temperature].repeat(prompt_len - 1));
                top_ps.extend_from_slice(&[top_p].repeat(prompt_len - 1));
                top_ks.extend_from_slice(&[top_k as u32].repeat(prompt_len - 1));
                min_ps.extend_from_slice(&[min_p].repeat(prompt_len - 1));
                presence_penalties.extend_from_slice(&[0.0].repeat(prompt_len - 1));
                frequency_penalties.extend_from_slice(&[0.0].repeat(prompt_len - 1));
                repetition_penalties.extend_from_slice(&[1.0].repeat(prompt_len - 1));
                for _ in 0..prompt_len - 1 {
                    prompt_tokens.extend(Vec::new());
                    output_tokens.extend(Vec::new());
                }
            }
            // tracing::info!("from_sampling_metadata 0:{:?}", start.elapsed());
            for seq_id in seq_ids {
                match sampling_metadata.seq_data.get(seq_id) {
                    Some(seq_data) => {
                        let seq_data = seq_data.borrow();
                        prompt_tokens.push(Vec::from(seq_data.get_prompt_token_ids()));
                        output_tokens.push(Vec::from(seq_data.get_output_token_ids()));
                    }
                    None => {
                        //
                    }
                }
            }
            //tracing::info!("from_sampling_metadata 1:{:?}", start.elapsed());
            temperatures.extend_from_slice(&[temperature].repeat(seq_ids.len()));
            top_ps.extend_from_slice(&[top_p].repeat(seq_ids.len()));
            top_ks.extend_from_slice(&[top_k as u32].repeat(seq_ids.len()));
            min_ps.extend_from_slice(&[min_p].repeat(seq_ids.len()));
            presence_penalties.extend_from_slice(&[p].repeat(seq_ids.len()));
            frequency_penalties.extend_from_slice(&[f].repeat(seq_ids.len()));
            repetition_penalties.extend_from_slice(&[r].repeat(seq_ids.len()));
            //tracing::info!("from_sampling_metadata 2:{:?}", start.elapsed());
        }

        let s = SamplingTensors::from_list(
            temperatures,
            top_ps,
            top_ks,
            min_ps,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            prompt_tokens,
            output_tokens,
            vocab_size,
            device,
            dtype,
            tensor_creator,
        )?;
        //tracing::info!("from_sampling_metadata 3:{:?}", start.elapsed());
        Ok((s, do_penalties, do_top_p_top_k, do_min_p))
    }

    #[allow(clippy::too_many_arguments)]
    fn from_list<F: TensorCreator>(
        temperatures: Vec<f32>,
        top_ps: Vec<f32>,
        top_ks: Vec<u32>,
        min_ps: Vec<f32>,
        presence_penalties: Vec<f32>,
        frequency_penalties: Vec<f32>,
        repetition_penalties: Vec<f32>,
        prompt_tokens: Vec<Vec<u32>>,
        output_tokens: Vec<Vec<u32>>,
        vocab_size: usize,
        device: &Device,
        dtype: DType,
        tensor_creator: &mut F,
    ) -> candle::Result<Self> {
        let _start = std::time::Instant::now();
        let prompt_max_len = prompt_tokens.iter().map(|x| x.len()).max().unwrap();
        let mut prompt_padded_tokens = Vec::new();
        for mut tokens in prompt_tokens {
            tokens.extend_from_slice(&[vocab_size as u32].repeat(prompt_max_len - tokens.len()));
            prompt_padded_tokens.push(tokens);
        }
        let output_max_len = output_tokens.iter().map(|x| x.len()).max().unwrap();

        let mut output_padded_tokens = Vec::new();
        for mut tokens in output_tokens {
            tokens.extend_from_slice(&[vocab_size as u32].repeat(output_max_len - tokens.len()));

            if !tokens.is_empty() {
                output_padded_tokens.push(tokens);
            }
        }

        let cpu_device = Device::Cpu;
        let cpu_temperatures = Tensor::new(temperatures, &cpu_device)?.to_dtype(dtype)?;
        let temperatures_t = tensor_creator.like(&cpu_temperatures, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_temperatures, &temperatures_t)?;

        let cpu_top_ps = Tensor::new(top_ps, &cpu_device)?.to_dtype(dtype)?;
        let top_ps_t = tensor_creator.like(&cpu_top_ps, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_top_ps, &top_ps_t)?;

        let cpu_min_ps = Tensor::new(min_ps, &cpu_device)?.to_dtype(dtype)?;
        let min_ps_t = tensor_creator.like(&cpu_min_ps, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_min_ps, &min_ps_t)?;

        let cpu_presence_penalties =
            Tensor::new(presence_penalties, &cpu_device)?.to_dtype(dtype)?;
        let presence_penalties_t = tensor_creator.like(&cpu_presence_penalties, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_presence_penalties, &presence_penalties_t)?;

        let cpu_frequency_penalties =
            Tensor::new(frequency_penalties, &cpu_device)?.to_dtype(dtype)?;
        let frequency_penalties_t = tensor_creator.like(&cpu_frequency_penalties, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_frequency_penalties, &frequency_penalties_t)?;

        let cpu_repetition_penalties =
            Tensor::new(repetition_penalties, &cpu_device)?.to_dtype(dtype)?;
        let repetition_penalties_t = tensor_creator.like(&cpu_repetition_penalties, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_repetition_penalties, &repetition_penalties_t)?;

        let cpu_top_ks = Tensor::new(top_ks, &cpu_device)?.to_dtype(DType::U32)?;
        let top_ks_t = tensor_creator.like(&cpu_top_ks, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_top_ks, &top_ks_t)?;

        let cpu_prompt_tensor =
            Tensor::new(prompt_padded_tokens, &cpu_device)?.to_dtype(DType::I64)?;
        let prompt_tensor = tensor_creator.like(&cpu_prompt_tensor, device)?;
        tops::unsafe_tensor_htod_copy(&cpu_prompt_tensor, &prompt_tensor)?;

        let output_tensor = if output_padded_tokens.is_empty() {
            let empty: Vec<i64> = Vec::new();
            Tensor::from_vec(empty, (1, 0), device)?
        } else {
            let cpu_output_tensor =
                Tensor::new(output_padded_tokens, &cpu_device)?.to_dtype(DType::I64)?;
            let output_tensor = tensor_creator.like(&cpu_output_tensor, device)?;
            tops::unsafe_tensor_htod_copy(&cpu_output_tensor, &output_tensor)?;
            output_tensor
        };

        Ok(Self {
            temperatures: temperatures_t,
            top_ps: top_ps_t,
            top_ks: top_ks_t,
            min_ps: min_ps_t,
            presence_penalties: presence_penalties_t,
            frequency_penalties: frequency_penalties_t,
            repetition_penalties: repetition_penalties_t,
            prompt_tokens: prompt_tensor,
            output_tokens: output_tensor,
        })
    }
}
