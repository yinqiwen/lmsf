use std::{
    collections::{BTreeMap, HashMap},
    ops::Mul,
};

use candle_core::{DType, Device, IndexOp, Tensor, D};
// use candle_ext::F;
use clap::ArgAction;
// use common::cuda_ext;

use crate::{
    common::{
        sampling_params::SamplingParams,
        sampling_params::SamplingType,
        sequence::{
            PromptLogprobs, SampleLogprobs, SamplerOutput, SequenceData, SequenceGroupOutput,
            SequenceOutput, SequenceRef,
        },
    },
    model_executor::ops::tensor::masked_fill_neg_inf,
    model_executor::sampling_metadata::{SamplingMetadata, SamplingTensors},
    tensor::{cuda_div, cuda_gt_, cuda_scatter_add, cuda_sub_, cuda_tensor_ones, TensorArena},
};

fn get_bin_counts_and_mask(
    cache: &mut TensorArena,
    mut tokens: Tensor,
    vocab_size: usize,
    num_seqs: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    //let bin_counts = Tensor::zeros((num_seqs, vocab_size + 1), DType::I64, tokens.device())?;
    let bin_counts = cache.get(DType::I64, (num_seqs, vocab_size + 1), true)?;

    if tokens.elem_count() > 0 {
        let scatter_src = cache.get(tokens.dtype(), tokens.shape(), false)?;
        cuda_tensor_ones(&scatter_src)?;
        cuda_scatter_add(&bin_counts, &tokens, &scatter_src, 1)?;
    }

    let bin_counts = bin_counts.i((.., ..vocab_size))?;
    //let mask = cache.get(DType::U8, bin_counts.shape(), false)?;
    let mask = cuda_gt_(&bin_counts, 0_i64, cache)?;
    //let mask = bin_counts.gt(0_i64)?;
    Ok((bin_counts, mask))
}

fn apply_penalties(
    cache: &mut TensorArena,
    logits: Tensor,
    prompt_tokens_tensor: Tensor,
    output_tokens_tensor: Tensor,
    presence_penalties: Tensor,
    frequency_penalties: Tensor,
    repetition_penalties: Tensor,
) -> candle_core::Result<Tensor> {
    let start = std::time::Instant::now();
    let (num_seqs, vocab_size) = logits.shape().dims2()?;
    let (_, prompt_mask) =
        get_bin_counts_and_mask(cache, prompt_tokens_tensor, vocab_size, num_seqs)?;
    let (output_bin_counts, output_mask) =
        get_bin_counts_and_mask(cache, output_tokens_tensor, vocab_size, num_seqs)?;

    let repetition_penalties = repetition_penalties.unsqueeze(D::Minus1)?;
    // tracing::info!("Cost before cuda_repeat_ {:?}", start.elapsed());
    let repetition_penalties = tops::cuda_repeat_(
        &repetition_penalties,
        (1, vocab_size),
        cache,
        std::ptr::null_mut(),
    )?;
    // tracing::info!("Cost after cuda_repeat_ {:?}", start.elapsed());
    // py: repetition_penalties[~(prompt_mask | output_mask)] = 1.0
    let repetition_penalties_cond = prompt_mask.where_cond(&prompt_mask, &output_mask)?;
    let repetition_penalties_tmp = Tensor::ones_like(&repetition_penalties)?;
    let repetition_penalties =
        repetition_penalties_cond.where_cond(&repetition_penalties, &repetition_penalties_tmp)?;

    // tracing::info!("Cost after where_cond1 {:?}", start.elapsed());
    //py: logits = torch.where(logits > 0, logits / repetition_penalties,logits * repetition_penalties)
    //let cond = logits.gt(0_i64)?;
    let cond = cuda_gt_(&logits, 0_i64, cache)?;
    //let true_logits = logits.div(&repetition_penalties)?;
    let true_logits = cuda_div(&logits, &repetition_penalties, cache)?;
    //let false_logits = logits.mul(&repetition_penalties)?;
    let false_logits =
        tops::cuda_tensor_mul_(&logits, &&repetition_penalties, logits.dtype(), cache)?;
    let logits = cond.where_cond(&true_logits, &false_logits)?;
    // tracing::info!("Cost after where_cond2 {:?}", start.elapsed());

    // logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    let frequency_penalties = frequency_penalties.unsqueeze(1)?;
    // let output_bin_counts = output_bin_counts
    //     .to_dtype(DType::F32)?
    //     .to_dtype(frequency_penalties.dtype())?;
    // let logits = logits.sub(&(frequency_penalties.broadcast_mul(&output_bin_counts)?))?;
    // let logits = logits.sub(&tops::cuda_tensor_broadcast_mul_(
    //     &frequency_penalties,
    //     &output_bin_counts,
    //     logits.dtype(),
    //     cache,
    // )?)?;

    cuda_sub_(
        &logits,
        &tops::cuda_tensor_broadcast_mul_(
            &frequency_penalties,
            &output_bin_counts,
            logits.dtype(),
            cache,
        )?,
    )?;
    // cuda_sub_(
    //     &logits,
    //     &(frequency_penalties.broadcast_mul(&output_bin_counts)?),
    // )?;

    // logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    // let output_mask = output_mask.to_dtype(presence_penalties.dtype())?;
    let presence_penalties = presence_penalties.unsqueeze(1)?;
    // let logits = logits.sub(&(presence_penalties.broadcast_mul(&output_mask)?))?;
    // let logits = logits.sub(&tops::cuda_tensor_broadcast_mul_(
    //     &presence_penalties,
    //     &output_mask,
    //     logits.dtype(),
    //     cache,
    // )?)?;
    cuda_sub_(
        &logits,
        &tops::cuda_tensor_broadcast_mul_(
            &presence_penalties,
            &output_mask,
            logits.dtype(),
            cache,
        )?,
    )?;

    // cuda_sub_(&logits, &(presence_penalties.broadcast_mul(&output_mask)?))?;

    // tracing::info!("apply_penalties all cost {:?}", start.elapsed());
    Ok(logits)
}

fn apply_top_p_top_k(
    arena: &mut TensorArena,
    logits: Tensor,
    p: Tensor,
    k: Tensor,
) -> candle_core::Result<Tensor> {
    let dim = logits.shape().dims().len() - 1;
    let start = std::time::Instant::now();

    let (logits_sort, logits_idx) =
        tops::cuda_sort_(logits, dim, true, arena, std::ptr::null_mut())?;
    // tracing::info!(
    //     "After cuda_sort_ elapsed:{:?}/{:?}",
    //     start.elapsed(),
    //     logits_sort.dtype()
    // );
    let probs_sort = tops::cuda_softmax_(&logits_sort, D::Minus1, arena, std::ptr::null_mut())?;
    // tracing::info!("After cuda_softmax_ elapsed:{:?}", start.elapsed());
    let probs_sum = tops::cuda_cumsum_(&probs_sort, 1, arena, std::ptr::null_mut())?;
    // tracing::info!("After cuda_cumsum_ elapsed:{:?}", start.elapsed());
    cuda_sub_(&probs_sum, &probs_sort)?;

    // tracing::info!("After cuda_sub_ elapsed:{:?}", start.elapsed());

    let p = p.unsqueeze(1)?;
    let top_p_mask = probs_sum.broadcast_gt(&p)?;

    let logits_idx_end = *(logits_idx.shape().dims().last().unwrap());
    // let top_k_mask = Tensor::arange(0_u32, logits_idx_end as u32, logits_idx.device())?;
    let top_k_mask = tops::cuda_arange_(
        0_u32,
        logits_idx_end as u32,
        logits_idx.device(),
        arena,
        std::ptr::null_mut(),
    )?;
    // tracing::info!(
    //     "After arange elapsed:{}:{:?}",
    //     logits_idx_end,
    //     start.elapsed()
    // );
    let new_shape = (logits_idx.shape().dims()[0], top_k_mask.shape().dims()[0]);
    let top_k_mask = top_k_mask.expand(new_shape)?;
    // tracing::info!("After expand elapsed:{:?}", start.elapsed());
    let k = k.unsqueeze(1)?;
    let top_k_mask = top_k_mask.broadcast_ge(&k)?;
    // tracing::info!("After broadcast_ge elapsed:{:?}", start.elapsed());

    // mask = (top_p_mask | top_k_mask)
    let mask = top_p_mask.where_cond(&top_p_mask, &top_k_mask)?;
    // let logits_sort = masked_fill_neg_inf(&logits_sort, &mask)?;
    tops::cuda_masked_fill_neg_inf_(&logits_sort, &mask)?;

    // tracing::info!("After where elapsed:{:?}", start.elapsed());
    let src = tops::cuda_arange_(
        0_u32,
        logits_idx_end as u32,
        logits_idx.device(),
        arena,
        std::ptr::null_mut(),
    )?;
    let src = src.expand(logits_idx.shape())?.contiguous()?;
    // let src = Tensor::arange(0_u32, logits_idx_end as u32, logits_idx.device())?
    //     .expand(logits_idx.shape())?
    //     .contiguous()?;

    //let logits_idx_inv = Tensor::zeros_like(&logits_idx)?;
    let logits_idx_inv = arena.get(logits_idx.dtype(), logits_idx.shape(), true)?;

    tops::cuda_scatter(
        &logits_idx_inv,
        &logits_idx,
        &src,
        D::Minus1,
        std::ptr::null_mut(),
    )?;
    // tracing::info!("After cuda_scatter elapsed:{:?}", start.elapsed());
    let logits = logits_sort.gather(&logits_idx_inv, D::Minus1)?;
    // tracing::info!("After gather elapsed:{:?}", start.elapsed());
    Ok(logits)
}

// fn masked_fill(
//     on_false: &Tensor,
//     mask: &Tensor,
//     on_true: half::f16,
// ) -> candle_core::Result<Tensor> {
//     let shape = mask.shape();
//     let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
//     let m = mask.where_cond(&on_true, on_false)?;
//     Ok(m)
// }

fn apply_min_p(
    arena: &mut TensorArena,
    logits: Tensor,
    min_p: Tensor,
) -> candle_core::Result<Tensor> {
    // let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let probs = tops::cuda_softmax_(&logits, D::Minus1, arena, std::ptr::null_mut())?;
    let top_probs = probs.max_keepdim(D::Minus1)?;
    let scaled_min_p = min_p.unsqueeze(1)?.matmul(&top_probs)?;
    let tokens_to_remove = probs.broadcast_lt(&scaled_min_p)?;
    let logits = masked_fill_neg_inf(&logits, &tokens_to_remove)?;
    Ok(logits)

    // """
    // Adapted from
    // https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    // """
    // probs = torch.softmax(logits, dim=-1)
    // top_probs, _ = probs.max(dim=-1, keepdim=True)
    // scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
    // tokens_to_remove = probs < scaled_min_p
    // logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    // return logits
}
fn multinomial(mut probs: Tensor, num_samples: usize) -> candle_core::Result<Tensor> {
    // tracing::info!(
    //     "multinomial enter probs:{:?}, num_samples:{}",
    //     probs.to_string(),
    //     num_samples
    // );
    // let x = probs.to_vec2::<f32>()?;
    // for v0 in x {
    //     for (idx, v1) in v0.iter().enumerate() {
    //         if *v1 != 0.0 {
    //             tracing::info!("multinomial enter probs idx:{}, val:{}", idx, *v1);
    //         }
    //     }
    // }
    let start = std::time::Instant::now();
    if num_samples > 1 {
        let shape0 = probs.shape().dims()[0];
        let shape1 = probs.shape().dims()[1];
        let new_shape = (shape0, num_samples, shape1);
        probs = probs.unsqueeze(1)?.expand(new_shape)?.contiguous()?;
        probs = probs.reshape((probs.elem_count() / shape1, shape1))?;
    }
    let q = Tensor::rand_like(&probs, 0.0, 1.0)?;
    tops::cuda_tensor_exponential(&q, 1.0, std::ptr::null_mut())?;
    // let q = q.neg()?;
    // let q = q.exp()?;
    // tracing::info!("multinomial q {:?}", q.to_string(),);
    crate::tensor::cuda_div_(&probs, &q)?;
    // let probs = probs.div(&q)?;
    let x = probs.argmax(1)?;
    let x = x.reshape((x.elem_count() / num_samples, num_samples))?;
    // tracing::info!("multinomial exit {:?}", x.to_string(),);
    // Ok(x)
    // tracing::info!("multinomial exit {:?}", start.elapsed());
    Ok(x)
}

fn greedy_sample(
    selected_seq_groups: &Vec<&(Vec<u64>, SamplingParams)>,
    samples: &Tensor,
) -> candle_core::Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let samples = samples.to_vec2::<u32>()?;
    let mut sample_idx: u32 = 0;
    let mut results = Vec::new();
    for (seq_ids, _) in selected_seq_groups {
        let num_parent_seqs = seq_ids.len() as u32;
        let parent_ids: Vec<u32> = (0..num_parent_seqs).collect();
        assert!(
            num_parent_seqs == 1,
            "Greedy sampling should have only one seq."
        );
        //let next_token_ids = samples.i(0)?;
        let next_token_ids = &samples[sample_idx as usize];
        //results.append((next_token_ids, parent_ids))
        results.push((next_token_ids.clone(), parent_ids));
        sample_idx += num_parent_seqs;
    }
    Ok(results)
}

fn random_sample(
    selected_seq_groups: &Vec<&(Vec<u64>, SamplingParams)>,
    is_prompts: &Vec<bool>,
    random_samples: &Tensor,
) -> candle_core::Result<Vec<(Vec<u32>, Vec<u32>)>> {
    // force sync from gpu to cpu
    let cpu_device = Device::Cpu;
    let random_samples = random_samples.to_device(&cpu_device)?;

    let mut sample_idx: usize = 0;
    let mut results = Vec::new();
    for ((seq_ids, sampling_params), is_prompt) in selected_seq_groups.iter().zip(is_prompts.iter())
    {
        let num_parent_seqs = seq_ids.len();
        let best_of = sampling_params.best_of.unwrap();
        let (parent_ids, next_token_ids) = if *is_prompt {
            let parent_ids = [0_u32].repeat(best_of);
            let next_token_ids = random_samples
                .i((sample_idx, ..best_of))?
                .to_vec1::<u32>()?;
            (parent_ids, next_token_ids)
        } else {
            let parent_ids: Vec<u32> = (0..num_parent_seqs as u32).collect();
            let next_token_ids = random_samples
                .i((sample_idx..sample_idx + num_parent_seqs, 0))?
                .to_vec1::<u32>()?;
            (parent_ids, next_token_ids)
        };
        results.push((next_token_ids.clone(), parent_ids));
        sample_idx += num_parent_seqs;
    }
    Ok(results)
}

fn beam_search_sample(
    selected_seq_groups: &Vec<&(Vec<u64>, SamplingParams)>,
    is_prompts: &Vec<bool>,
    seq_data: &HashMap<u64, SequenceRef>,
    logprobs: &Tensor,
) -> candle_core::Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let mut sample_idx: usize = 0;
    let mut results = Vec::new();
    for ((seq_ids, sampling_params), is_prompt) in selected_seq_groups.iter().zip(is_prompts.iter())
    {
        let num_parent_seqs = seq_ids.len();
        let beam_width = sampling_params.best_of.unwrap();
        //seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
        let seq_group_logprobs = logprobs.i(sample_idx..sample_idx + num_parent_seqs)?;
        let (next_token_ids, parent_ids) = if *is_prompt {
            assert!(
                num_parent_seqs == 1,
                "Prompt input should have only one seq."
            );
            let parent_ids = [0_u32].repeat(2 * beam_width);
            let (_, next_token_ids) = tops::cuda_topk(
                &seq_group_logprobs.i(0)?,
                2 * beam_width,
                1,
                std::ptr::null_mut(),
            )?;
            // next_token_ids = next_token_ids.tolist()
            let next_token_ids = next_token_ids.to_vec1::<u32>()?;
            (next_token_ids, parent_ids)
        } else {
            let cumulative_logprobs = seq_ids
                .iter()
                .map(|x| seq_data.get(x).unwrap().borrow().get_cumulative_logprob())
                .collect::<Vec<_>>();
            let cumulative_logprobs =
                Tensor::new(cumulative_logprobs, seq_group_logprobs.device())?;
            let seq_group_logprobs = (seq_group_logprobs + cumulative_logprobs.unsqueeze(1))?;
            let (_, topk_ids) = tops::cuda_topk(
                &seq_group_logprobs.flatten_all()?,
                2 * beam_width,
                1,
                std::ptr::null_mut(),
            )?;
            let topk_ids = topk_ids.to_vec1::<u32>()?;
            let vocab_size = *seq_group_logprobs.dims().last().unwrap();
            let parent_ids = topk_ids
                .iter()
                .map(|x| x / vocab_size as u32)
                .collect::<Vec<_>>();
            let next_token_ids = topk_ids
                .iter()
                .map(|x| x % vocab_size as u32)
                .collect::<Vec<_>>();
            (next_token_ids, parent_ids)
        };
        results.push((next_token_ids, parent_ids));
        sample_idx += num_parent_seqs;
    }
    Ok(results)
}

fn sample(
    probs: Tensor,
    logprobs: &Tensor,
    sampling_metadata: &SamplingMetadata,
) -> candle_core::Result<Vec<(Vec<u32>, Vec<u32>)>> {
    // categorized_seq_group_ids = {t: [] for t in SamplingType}
    // categorized_sample_indices = sampling_metadata.categorized_sample_indices
    // for i, seq_group in enumerate(sampling_metadata.seq_groups):
    //     _, sampling_params = seq_group
    //     sampling_type = sampling_params.sampling_type
    //     categorized_seq_group_ids[sampling_type].append(i)

    // sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    // sample_metadata = {}

    let start = std::time::Instant::now();

    let mut categorized_seq_group_ids: Vec<Vec<usize>> = Vec::new();
    for i in 0..SamplingType::Beam as usize {
        categorized_seq_group_ids.push(Vec::new());
    }
    for (i, (_, sampling_params)) in sampling_metadata.seq_groups.iter().enumerate() {
        let idx = sampling_params.sampling_type() as usize;
        categorized_seq_group_ids[idx].push(i);
    }

    let mut sample_metadata = HashMap::new();

    let mut greedy_samples: Option<Tensor> = None;
    let mut multinomial_samples: Option<Tensor> = None;
    let mut beam_search_logprobs: Option<Tensor> = None;

    for i in 0..SamplingType::Beam as usize {
        let sample_type = SamplingType::from_int(i).unwrap();
        if let Some(sample_indices) = sampling_metadata
            .categorized_sample_indices
            .get(&sample_type)
        {
            let num_tokens = sample_indices.dims()[0];
            if num_tokens == 0 {
                continue;
            }
            let seq_group_ids = &categorized_seq_group_ids[i];
            let seq_groups: Vec<_> = seq_group_ids
                .iter()
                .map(|x| &sampling_metadata.seq_groups[*x])
                .collect();
            let is_prompts: Vec<bool> = seq_group_ids
                .iter()
                .map(|x| *x < sampling_metadata.num_prompts)
                .collect();
            sample_metadata.insert(
                i,
                (
                    seq_group_ids,
                    seq_groups.clone(),
                    is_prompts.clone(),
                    sample_indices,
                ),
            );

            match sample_type {
                SamplingType::Greedy => {
                    let x = logprobs
                        .index_select(&sample_indices, 0)?
                        .argmax(D::Minus1)?;
                    greedy_samples = Some(x);
                }
                SamplingType::Random => {
                    let mut max_best_of = 1;
                    for (seq_group, is_prompt) in seq_groups.iter().zip(is_prompts.iter()) {
                        if *is_prompt {
                            max_best_of = std::cmp::max(max_best_of, seq_group.1.best_of.unwrap());
                        }
                    }
                    let x = multinomial(probs.index_select(sample_indices, 0)?, max_best_of)?;
                    multinomial_samples = Some(x);
                }
                SamplingType::Beam => {
                    beam_search_logprobs = Some(logprobs.index_select(sample_indices, 0)?);
                }
                _ => {
                    candle_core::bail!("not supported sample type:{:?}", sample_type);
                }
            }
        }
    }
    // tracing::info!("sampler 0 {:?}", start.elapsed());

    // sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    let mut sample_results_dict: BTreeMap<usize, (Vec<u32>, Vec<u32>)> = BTreeMap::new();

    for i in 0..SamplingType::Beam as usize {
        if let Some((seq_group_ids, seq_groups, is_prompts, sample_indices)) =
            sample_metadata.get(&i)
        {
            let sample_type = SamplingType::from_int(i).unwrap();
            let sample_results = match sample_type {
                SamplingType::Greedy => {
                    greedy_sample(seq_groups, greedy_samples.as_ref().unwrap())?
                }
                SamplingType::Random => random_sample(
                    seq_groups,
                    is_prompts,
                    multinomial_samples.as_ref().unwrap(),
                )?,
                SamplingType::Beam => beam_search_sample(
                    seq_groups,
                    is_prompts,
                    &sampling_metadata.seq_data,
                    beam_search_logprobs.as_ref().unwrap(),
                )?,
                _ => {
                    candle_core::bail!("not supported sample type:{:?}", sample_type);
                }
            };
            // tracing::info!("sample_results:{:?}", sample_results);
            for (key, val) in seq_group_ids.iter().zip(sample_results.into_iter()) {
                sample_results_dict.insert(*key, val);
            }
        }
    }
    // tracing::info!("sampler 1 {:?}", start.elapsed());

    let mut result = Vec::new();
    for (i, v) in sample_results_dict {
        result.push(v);
    }
    Ok(result)
}

fn get_logprobs(
    logprobs: Tensor,
    sampling_metadata: &SamplingMetadata,
    sample_results: &Vec<(Vec<u32>, Vec<u32>)>,
) -> candle_core::Result<(Vec<Option<PromptLogprobs>>, Vec<SampleLogprobs>)> {
    let mut batched_logprobs_query_seq_indices: Vec<u32> = Vec::new();
    let mut batched_logprobs_query_token_indices: Vec<u32> = Vec::new();
    let mut largest_num_logprobs = 0_u32;
    let mut sample_idx = 0_usize;
    for (i, (seq_group, sample_result)) in sampling_metadata
        .seq_groups
        .iter()
        .zip(sample_results.iter())
        .enumerate()
    {
        let (seq_ids, sampling_params) = seq_group;
        let (next_token_ids, parent_ids) = sample_result;
        let num_parent_seqs = seq_ids.len();
        if i < sampling_metadata.num_prompts && sampling_params.prompt_logprobs.is_some() {
            largest_num_logprobs = std::cmp::max(
                sampling_params.prompt_logprobs.unwrap(),
                largest_num_logprobs,
            );
            let prompt_len = sampling_metadata.prompt_lens[i];
            batched_logprobs_query_seq_indices.extend(
                (sample_idx as u32..sample_idx as u32 + prompt_len as u32 - 1).collect::<Vec<_>>(),
            );
            {
                let seq_borrow = sampling_metadata
                    .seq_data
                    .get(&seq_ids[0])
                    .unwrap()
                    .borrow();
                let prompt_tokens = seq_borrow.get_prompt_token_ids();
                batched_logprobs_query_token_indices.extend_from_slice(&prompt_tokens[1..]);
            }
            sample_idx += prompt_len - 1
        }
        batched_logprobs_query_seq_indices.extend(
            parent_ids
                .iter()
                .map(|x| *x + sample_idx as u32)
                .collect::<Vec<_>>(),
        );
        batched_logprobs_query_token_indices.extend_from_slice(next_token_ids);
        if let Some(logprobs) = sampling_params.logprobs {
            largest_num_logprobs = std::cmp::max(largest_num_logprobs, logprobs);
        }
        sample_idx += num_parent_seqs;
    }
    assert!(sample_idx == logprobs.dims()[0]);

    let index_length = batched_logprobs_query_seq_indices.len();
    let batched_logprobs_query_seq_indices_t =
        Tensor::new(batched_logprobs_query_seq_indices, logprobs.device())?;
    let batched_logprobs_query_token_indices_t =
        Tensor::new(batched_logprobs_query_token_indices, logprobs.device())?;

    let selected_rows = logprobs.index_select(&batched_logprobs_query_seq_indices_t, 0)?;
    let batched_logprobs_query_result = selected_rows.gather(
        &batched_logprobs_query_token_indices_t.reshape((index_length, 1))?,
        1,
    )?;
    let batched_logprobs_query_result = batched_logprobs_query_result.squeeze(D::Minus1)?;

    // tracing::info!("logprobs:{:?}", logprobs.shape());

    let cpu = Device::Cpu;
    let top_logprobs_token_ids = if largest_num_logprobs > 0 {
        let dim = *logprobs.dims().last().unwrap();
        let (val, indices) = tops::cuda_topk(
            &logprobs,
            largest_num_logprobs as usize,
            dim,
            std::ptr::null_mut(),
        )?;
        Some((val.to_device(&cpu)?, indices.to_device(&cpu)?))
    } else {
        None
    };
    let batched_logprobs_query_result = batched_logprobs_query_result.to_device(&cpu)?;
    // Gather results
    let mut result_prompt_logprobs: Vec<Option<PromptLogprobs>> = Vec::new();
    let mut result_sample_logprobs: Vec<SampleLogprobs> = Vec::new();
    let mut sample_idx = 0_usize;
    let mut query_result_idx = 0_usize;

    for (i, (seq_group, sample_result)) in sampling_metadata
        .seq_groups
        .iter()
        .zip(sample_results.iter())
        .enumerate()
    {
        let (seq_ids, sampling_params) = seq_group;
        let (next_token_ids, parent_ids) = sample_result;

        if i < sampling_metadata.num_prompts && sampling_params.prompt_logprobs.is_some() {
            let num_logprobs = sampling_params.prompt_logprobs.unwrap();
            let prompt_len = sampling_metadata.prompt_lens[i];
            let seq_ref = sampling_metadata
                .seq_data
                .get(&seq_ids[0])
                .unwrap()
                .borrow();
            let prompt_tokens = seq_ref.get_prompt_token_ids();
            let mut group_prompt_logprobs: PromptLogprobs = vec![None];
            for token_id in &prompt_tokens[1..] {
                let mut prompt_logprobs_dict: HashMap<u32, f32> = HashMap::new();
                let logprobs: f32 = batched_logprobs_query_result
                    .i(query_result_idx)?
                    .to_scalar::<f32>()?;
                prompt_logprobs_dict.insert(*token_id, logprobs);
                if num_logprobs > 0 {
                    if let Some((top_logprobs, top_token_ids)) = &top_logprobs_token_ids {
                        let top_token_ids_slice = top_token_ids
                            .i((sample_idx, ..num_logprobs as usize))?
                            .to_vec1::<u32>()?;
                        let top_logprobs_slice = top_logprobs
                            .i((sample_idx, ..num_logprobs as usize))?
                            .to_vec1::<f32>()?;
                        for (id, logprobs) in top_token_ids_slice
                            .into_iter()
                            .zip(top_logprobs_slice.into_iter())
                        {
                            prompt_logprobs_dict.insert(id, logprobs);
                        }
                    }
                }
                group_prompt_logprobs.push(Some(prompt_logprobs_dict));
                sample_idx += 1;
                query_result_idx += 1;
            }
            result_prompt_logprobs.push(Some(group_prompt_logprobs))
        } else {
            result_prompt_logprobs.push(None);
        }
        // tracing::info!("result_prompt_logprobs:{:?}", result_prompt_logprobs);
        let num_logprobs = if let Some(v) = sampling_params.logprobs {
            v
        } else {
            0
        };
        let mut group_sample_logprobs: SampleLogprobs = Vec::new();
        for (next_token_id, parent_id) in next_token_ids.iter().zip(parent_ids.iter()) {
            let mut sample_logprobs_dict: HashMap<u32, f32> = HashMap::new();
            let logprobs: f32 = batched_logprobs_query_result
                .i(query_result_idx)?
                .to_scalar::<f32>()?;
            sample_logprobs_dict.insert(*next_token_id, logprobs);
            query_result_idx += 1;
            if num_logprobs > 0 {
                if let Some((top_logprobs, top_token_ids)) = &top_logprobs_token_ids {
                    let top_token_ids_slice = top_token_ids
                        .i((sample_idx + (*parent_id as usize), ..num_logprobs as usize))?
                        .to_vec1::<u32>()?;
                    let top_logprobs_slice = top_logprobs
                        .i((sample_idx + (*parent_id as usize), ..num_logprobs as usize))?
                        .to_vec1::<f32>()?;
                    for (id, logprobs) in top_token_ids_slice
                        .into_iter()
                        .zip(top_logprobs_slice.into_iter())
                    {
                        sample_logprobs_dict.insert(id, logprobs);
                    }
                }
            }
            group_sample_logprobs.push(sample_logprobs_dict);
        }
        result_sample_logprobs.push(group_sample_logprobs);
        sample_idx += seq_ids.len();
    }
    Ok((result_prompt_logprobs, result_sample_logprobs))
}

fn build_sampler_output(
    sample_results: Vec<(Vec<u32>, Vec<u32>)>,
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: Vec<Option<PromptLogprobs>>,
    sample_logprobs: Vec<SampleLogprobs>,
) -> candle_core::Result<SamplerOutput> {
    let mut sampler_output: SamplerOutput = Vec::new();
    for (seq_group, sample_result, group_prompt_logprobs, group_sample_logprobs) in
        sampling_metadata
            .seq_groups
            .into_iter()
            .zip(sample_results.into_iter())
            .zip(prompt_logprobs.into_iter())
            .zip(sample_logprobs.into_iter())
            .map(|(((x, y), z), w)| (x, y, z, w))
    {
        let (seq_ids, _) = seq_group;
        let (next_token_ids, parent_ids) = sample_result;
        let mut seq_outputs = Vec::new();

        for (parent_id, next_token_id, logprobs) in parent_ids
            .into_iter()
            .zip(next_token_ids.into_iter())
            .zip(group_sample_logprobs.into_iter())
            .map(|((x, y), z)| (x, y, z))
        {
            seq_outputs.push(SequenceOutput::new(
                seq_ids[parent_id as usize],
                next_token_id,
                logprobs,
            ));
        }
        sampler_output.push(SequenceGroupOutput::new(seq_outputs, group_prompt_logprobs));
    }
    Ok(sampler_output)
}

pub struct Sampler {
    // vocab_size: usize,
    // base_tensor_buffer: SamplingTensors,
    // cache: TensorCache,
    arena: TensorArena,
}

impl Sampler {
    pub fn new(
        max_batch: usize,
        max_model_len: usize,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Self> {
        // let base_tensor_buffer =
        //     SamplingTensors::allocate_base_buffer(max_batch, max_model_len, dtype, device)?;
        // let mut cache = TensorCache::new();
        // cache.fill(DType::I64, 1024 * 32, device)?;
        // cache.fill(DType::U8, 1024 * 32, device)?;
        // cache.fill(DType::F16, 1024 * 32, device)?;
        Ok(Self {
            // base_tensor_buffer,
            // cache,
            arena: TensorArena::new(device),
        })
    }

    pub fn forward(
        &mut self,
        mut logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> candle_core::Result<Option<SamplerOutput>> {
        // tracing::info!("sample logits:{}", logits.to_string());
        // self.cache.reset();
        self.arena.reset();
        // self.arena.print_stat();
        // let cuda_dev = match logits.device() {
        //     Device::Cuda(cuda) => cuda.clone(),
        //     _ => {
        //         candle_core::bail!("")
        //     }
        // };
        // cuda_dev.synchronize();

        let (_, vocab_size) = logits.shape().dims2()?;
        if !sampling_metadata.perform_sampling {
            return Ok(None);
        }

        let start = std::time::Instant::now();
        let (sampling_tensors, do_penalties, do_top_p_top_k, do_min_p) =
            SamplingTensors::from_sampling_metadata(
                &sampling_metadata,
                vocab_size,
                logits.device(),
                logits.dtype(),
                &mut self.arena,
            )?;
        // tracing::info!("sample 0 cost {:?}/{}", start.elapsed(), do_penalties);

        if do_penalties {
            logits = apply_penalties(
                &mut self.arena,
                logits,
                sampling_tensors.prompt_tokens,
                sampling_tensors.output_tokens,
                sampling_tensors.presence_penalties,
                sampling_tensors.frequency_penalties,
                sampling_tensors.repetition_penalties,
            )?;
        }

        // tracing::info!("sample 1 cost {:?}", start.elapsed());
        // # Apply temperature scaling.
        // # Use in-place division to avoid creating a new tensor.
        let temperatures = sampling_tensors.temperatures.unsqueeze(1)?;
        let mut logits = logits.broadcast_div(&temperatures)?;
        // cuda_dev.synchronize();
        // tracing::info!("sample 1.5 cost {:?}, {}", start.elapsed(), do_min_p);
        if do_top_p_top_k {
            //apply_top_p_top_k();
            logits = apply_top_p_top_k(
                &mut self.arena,
                logits,
                sampling_tensors.top_ps,
                sampling_tensors.top_ks,
            )?;
        }
        if do_min_p {
            logits = apply_min_p(&mut self.arena, logits, sampling_tensors.min_ps)?;
        }

        // cuda_dev.synchronize();
        // tracing::info!("sample 2 cost {:?}", start.elapsed());

        let logits = logits.to_dtype(DType::F32)?;
        //let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let probs = tops::cuda_softmax_(&logits, D::Minus1, &mut self.arena, std::ptr::null_mut())?;
        // cuda_dev.synchronize();
        // tracing::info!("sample 3 cost {:?}", start.elapsed());
        //let logprobs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let logprobs =
            tops::cuda_log_softmax_(&logits, D::Minus1, &mut self.arena, std::ptr::null_mut())?;

        // cuda_dev.synchronize();
        // tracing::info!("sample 4 cost {:?}", start.elapsed());
        // let x = probs.to_vec2::<f32>()?;
        // for v0 in x {
        //     for (idx, v1) in v0.iter().enumerate() {
        //         if *v1 != 0.0 {
        //             tracing::info!("probs idx:{}, val:{}", idx, *v1);
        //         }
        //     }
        // }
        // let x = logprobs.to_vec2::<f32>()?;
        // for v0 in x {
        //     for (idx, v1) in v0.iter().enumerate() {
        //         if *v1 != f32::NEG_INFINITY {
        //             tracing::info!("logprobs idx:{}, val:{}", idx, *v1);
        //         }
        //     }
        // }

        // # Sample the next tokens.
        // sample_results = _sample(probs, logprobs, sampling_metadata)
        let sample_results = sample(probs, &logprobs, &sampling_metadata)?;
        // tracing::info!("sample result{:?}", sample_results);
        // cuda_dev.synchronize();
        // tracing::info!("sample 5 cost {:?}", start.elapsed());
        // tracing::info!("sample:{:?}", sample_results);
        let (prompt_logprobs, sample_logprobs) =
            get_logprobs(logprobs, &sampling_metadata, &sample_results)?;
        // cuda_dev.synchronize();
        // tracing::info!("sample 6 cost {:?}", start.elapsed());

        let result = build_sampler_output(
            sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
        )?;
        // tracing::info!("sample forward cost {:?}", start.elapsed());
        Ok(Some(result))
    }
}
