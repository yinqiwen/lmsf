use std::{sync::Arc, time::Instant};

use super::sequence::{PromptLogprobs, SampleLogprobs, SequenceGroupRef};

#[derive(Debug, Clone)]
pub struct CompletionOutput {
    pub index: usize,
    pub text: String,
    pub latest_token: String,
    pub token_ids: Vec<u32>,
    cumulative_logprob: f32,
    logprobs: Option<SampleLogprobs>,
    pub finish_reason: Option<&'static str>,
}

impl CompletionOutput {
    pub fn new(
        index: usize,
        text: String,
        latest_token: String,
        token_ids: Vec<u32>,
        cumulative_logprob: f32,
        logprobs: Option<SampleLogprobs>,
        finish_reason: Option<&'static str>,
    ) -> Self {
        Self {
            index,
            text,
            latest_token,
            token_ids,
            cumulative_logprob,
            logprobs,
            finish_reason,
        }
    }
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    pub fn get_finish_reason(&self) -> Option<String> {
        match self.finish_reason {
            Some(s) => Some(String::from(s)),
            None => None,
        }
    }
}
#[derive(Debug)]
pub struct RequestOutput {
    pub request_id: u64,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_logprobs: Option<Arc<PromptLogprobs>>,
    pub outputs: Vec<CompletionOutput>,
    pub(crate) finished: bool,
}

impl RequestOutput {
    pub fn from(seq_group_ref: &SequenceGroupRef) -> Self {
        // # Get the top-n sequences.
        let mut seq_group = seq_group_ref.borrow_mut();
        let n = seq_group.sampling_params.n;
        let mut seqs = seq_group.get_seqs(None);

        if seq_group.sampling_params.use_beam_search {
            seqs.sort_by(|x, y| {
                let x_score = x.borrow().get_beam_search_score(
                    seq_group.sampling_params.length_penalty,
                    None,
                    None,
                );
                let y_score = y.borrow().get_beam_search_score(
                    seq_group.sampling_params.length_penalty,
                    None,
                    None,
                );
                x_score.total_cmp(&y_score).reverse()
            });
        } else {
            seqs.sort_by(|x, y| {
                let x_score = x.borrow().get_cumulative_logprob();
                let y_score = y.borrow().get_cumulative_logprob();
                x_score.total_cmp(&y_score).reverse()
            });
        }

        let top_n_seqs = &seqs[0..n];

        // # Create the outputs.
        let mut outputs = Vec::new();
        for (idx, seq) in top_n_seqs.iter().enumerate() {
            let seq = seq.borrow();
            let logprobs = if seq_group.sampling_params.logprobs.is_none() {
                // # NOTE: We need to take care of this case because the sequence
                // # always has the logprobs of the sampled tokens even if the
                // # logprobs are not requested.
                None
            } else {
                Some(seq.output_logprobs.clone())
            };

            let finshed_reason = seq.get_state().get_finished_reason();
            let latest_token_idx = seq.gen_texts.len() - 1;
            let output = CompletionOutput::new(
                idx,
                seq.output_text.clone(),
                seq.gen_texts[latest_token_idx].clone(),
                Vec::from(seq.get_output_token_ids()),
                seq.get_cumulative_logprob(),
                logprobs,
                finshed_reason,
            );
            outputs.push(output);
        }

        // # Every sequence in the sequence group should have the same prompt.
        let prompt = seq_group.prompt();
        let prompt_token_ids = seq_group.prompt_token_ids();
        let prompt_logprobs = seq_group.prompt_logprobs.clone();
        let finished = seq_group.is_finished();
        if finished {
            seq_group.set_finished_time(Instant::now())
        }
        Self {
            request_id: seq_group.request_id,
            prompt,
            prompt_token_ids,
            prompt_logprobs,
            outputs,
            finished,
        }
    }
}
