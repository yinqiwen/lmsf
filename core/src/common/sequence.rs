use candle_nn::seq;
use num::{FromPrimitive, ToPrimitive};
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{collections::HashMap, sync::atomic::AtomicU32};

use anyhow::anyhow;

use super::{block::LogicalTokenBlock, sampling_params::SamplingParams};

#[derive(Clone, PartialEq, FromPrimitive, ToPrimitive, Debug)]
pub enum SequenceState {
    Waiting = 0,
    Running,
    Swapped,
    FinishedStopeed,
    FinishedLengthCapped,
    FinishedAborted,
    FinishedIgnored,
}

impl SequenceState {
    pub fn is_finished(&self) -> bool {
        matches!(
            *self,
            SequenceState::FinishedAborted
                | SequenceState::FinishedIgnored
                | SequenceState::FinishedLengthCapped
                | SequenceState::FinishedStopeed
        )
    }
    pub fn get_finished_reason(&self) -> Option<&'static str> {
        match *self {
            SequenceState::FinishedStopeed => Some("stop"),
            SequenceState::FinishedLengthCapped => Some("length"),
            SequenceState::FinishedAborted => Some("abort"),
            SequenceState::FinishedIgnored => Some("length"),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct SequenceData {
    prompt_token_ids: Vec<u32>,
    output_token_ids: Vec<u32>,
    cumulative_logprob: f32,
}

impl SequenceData {
    pub fn new(prompt_token_ids: &[u32]) -> Self {
        Self {
            prompt_token_ids: Vec::from(prompt_token_ids),
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.0,
        }
    }

    pub fn append_token_id(&mut self, token_id: u32, logprob: f32) {
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
    }

    pub fn get_len(&self) -> usize {
        self.output_token_ids.len() + self.prompt_token_ids.len()
    }
    pub fn get_prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }
    pub fn get_output_len(&self) -> usize {
        self.output_token_ids.len()
    }
    pub fn get_token_ids(&self) -> Vec<u32> {
        let mut ids = self.prompt_token_ids.clone();
        ids.extend_from_slice(&self.output_token_ids);
        ids
    }
    pub fn get_prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }
    pub fn get_output_token_ids(&self) -> &[u32] {
        &self.output_token_ids
    }
    pub fn get_last_token_id(&self) -> u32 {
        if self.output_token_ids.is_empty() {
            return self.prompt_token_ids[self.prompt_token_ids.len() - 1];
        }

        self.output_token_ids[self.output_token_ids.len() - 1]
    }
}

pub struct Sequence {
    pub(crate) seq_id: u64,
    prompt: String,
    block_size: usize,
    data: SequenceData,
    pub(crate) prefix_offset: usize,
    pub(crate) read_offset: usize,
    pub(crate) gen_texts: Vec<String>,
    pub(crate) output_text: String,
    // pub(crate) latest_output_token: String,
    // pub(crate) tokens: Option<Vec<String>>,
    pub(crate) logical_token_blocks: Vec<LogicalTokenBlock>,
    pub(crate) output_logprobs: Vec<HashMap<u32, f32>>,
    state: AtomicU32,
}
pub type SequenceRef = Rc<RefCell<Sequence>>;

impl Sequence {
    pub fn new_ref(self) -> SequenceRef {
        Rc::new(RefCell::new(self))
    }
    pub fn new(seq_id: u64, prompt: &str, prompt_token_ids: &[u32], block_size: usize) -> Self {
        let mut seq = Self {
            seq_id,
            prompt: String::from(prompt),
            block_size,
            data: SequenceData::new(prompt_token_ids),
            output_text: String::new(),
            // latest_output_token: String::new(),
            gen_texts: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            // tokens: None,
            logical_token_blocks: Vec::new(),
            output_logprobs: Vec::new(),
            state: AtomicU32::new(SequenceState::Waiting.to_u32().unwrap()),
        };
        seq.append_tokens_to_blocks(prompt_token_ids);
        seq
    }

    pub fn update_state(&self, state: SequenceState) {
        let v = state.to_u32().unwrap();
        self.state.store(v, std::sync::atomic::Ordering::SeqCst);
    }

    fn append_logical_block(&mut self) {
        let block = LogicalTokenBlock::new(self.logical_token_blocks.len() as u32, self.block_size);
        self.logical_token_blocks.push(block);
    }

    fn append_tokens_to_blocks(&mut self, token_ids: &[u32]) {
        let mut cursor = 0;
        while cursor < token_ids.len() {
            if self.logical_token_blocks.is_empty() {
                self.append_logical_block()
            }
            let last_block = self.logical_token_blocks.last_mut().unwrap();
            let last_block = if last_block.is_full() {
                self.append_logical_block();
                self.logical_token_blocks.last_mut().unwrap()
            } else {
                last_block
            };

            let num_empty_slots = last_block.get_num_empty_slots();
            let slice = if cursor + num_empty_slots > token_ids.len() {
                &token_ids[cursor..]
            } else {
                &token_ids[cursor..cursor + num_empty_slots]
            };
            last_block.append_tokens(slice);
            cursor += num_empty_slots;
        }
    }

    pub fn append_token_id(&mut self, token_id: u32, logprobs: HashMap<u32, f32>) {
        self.append_tokens_to_blocks(&[token_id]);
        let logprob = *(logprobs.get(&token_id).unwrap());
        self.output_logprobs.push(logprobs);
        self.data.append_token_id(token_id, logprob);
    }

    pub fn get_len(&self) -> usize {
        self.data.get_len()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.data.get_prompt_len()
    }

    pub fn get_output_len(&self) -> usize {
        self.data.get_output_len()
    }

    pub fn get_token_ids(&self) -> Vec<u32> {
        self.data.get_token_ids()
    }

    pub fn get_last_token_id(&self) -> u32 {
        self.data.get_last_token_id()
    }

    pub fn get_output_token_ids(&self) -> &[u32] {
        &self.data.output_token_ids
    }

    pub fn get_cumulative_logprob(&self) -> f32 {
        self.data.cumulative_logprob
    }

    pub fn get_prompt_token_ids(&self) -> &[u32] {
        self.data.get_prompt_token_ids()
    }

    pub fn get_state(&self) -> SequenceState {
        SequenceState::from_u32(self.state.load(std::sync::atomic::Ordering::SeqCst)).unwrap()
    }
    pub fn is_finished(&self) -> bool {
        let state =
            SequenceState::from_u32(self.state.load(std::sync::atomic::Ordering::SeqCst)).unwrap();
        state.is_finished()
    }

    pub fn get_beam_search_score(
        &self,
        length_penalty: f32,
        seq_len: Option<usize>,
        eos_token_id: Option<u32>,
    ) -> f32 {
        let seq_len = if let Some(n) = seq_len {
            n
        } else {
            let mut seq_len = self.get_len();
            if let Some(eos_token_id) = eos_token_id {
                if eos_token_id == self.get_last_token_id() {
                    seq_len -= 1;
                }
            }
            seq_len
        };
        self.get_cumulative_logprob() / (seq_len as f32).powf(length_penalty)
    }

    pub fn fork(&self, new_seq_id: u64) -> Self {
        let v = Self {
            seq_id: new_seq_id,
            prompt: self.prompt.clone(),
            block_size: self.block_size,
            data: self.data.clone(),
            output_text: self.output_text.clone(),
            // latest_output_token: self.latest_output_token.clone(),
            gen_texts: self.gen_texts.clone(),
            prefix_offset: self.prefix_offset,
            read_offset: self.read_offset,
            // tokens: self.tokens.clone(),
            logical_token_blocks: self.logical_token_blocks.clone(),
            output_logprobs: self.output_logprobs.clone(),
            state: AtomicU32::new(self.state.load(std::sync::atomic::Ordering::SeqCst)),
        };
        v
    }
}

pub struct SequenceGroup {
    pub(crate) request_id: u64,
    seqs: HashMap<u64, SequenceRef>,
    pub(crate) sampling_params: SamplingParams,
    pub(crate) arrival_time: Duration,
    pub(crate) prompt_logprobs: Option<Arc<PromptLogprobs>>,
}

pub type SequenceGroupRef = Rc<RefCell<SequenceGroup>>;

impl SequenceGroup {
    pub fn newRef(self) -> SequenceGroupRef {
        Rc::new(RefCell::new(self))
    }
    pub fn new(
        request_id: u64,
        seqs: Vec<Sequence>,
        sampling_params: SamplingParams,
        arrival_time: Duration,
    ) -> Self {
        let mut seq_map: HashMap<_, _> = HashMap::new();
        for seq in seqs {
            seq_map.insert(seq.seq_id, Rc::new(RefCell::new(seq)));
        }
        Self {
            request_id,
            seqs: seq_map,
            sampling_params,
            arrival_time,
            prompt_logprobs: None,
        }
    }
    pub fn set_prompt_logprobs(&mut self, prompt_logprobs: PromptLogprobs) {
        self.prompt_logprobs = Some(Arc::new(prompt_logprobs));
    }
    pub fn prompt(&self) -> String {
        // # All sequences in the group should have the same prompt.
        // # We use the prompt of an arbitrary sequence.
        self.seqs.iter().next().unwrap().1.borrow().prompt.clone()
    }

    pub fn prompt_token_ids(&self) -> Vec<u32> {
        // # All sequences in the group should have the same prompt.
        // # We use the prompt of an arbitrary sequence.
        self.seqs
            .iter()
            .next()
            .unwrap()
            .1
            .borrow()
            .data
            .prompt_token_ids
            .clone()
    }
    pub fn get_max_num_running_seqs(&self) -> usize {
        // """The maximum number of sequences running in parallel in the remaining
        // lifetime of the request."""
        if self.sampling_params.use_beam_search {
            // # For beam search, maximally there will always be `best_of` beam
            // # candidates running in the future.
            self.sampling_params.best_of.unwrap()
        } else {
            if let Some(best_of) = self.sampling_params.best_of {
                if best_of > self.num_seqs(None) {
                    // # At prompt stage, the sequence group is not yet filled up
                    // # and only have one sequence running. However, in the
                    // # generation stage, we will have `best_of` sequences running.
                    return best_of;
                }
            }
            // # At sampling stages, return the number of actual sequences
            // # that are not finished yet.
            self.num_unfinished_seqs()
        }
    }

    pub fn get_seqs(&self, status: Option<SequenceState>) -> Vec<SequenceRef> {
        let mut seqs = Vec::new();
        if let Some(status) = status {
            for (_, v) in self.seqs.iter() {
                if v.borrow().state.load(std::sync::atomic::Ordering::SeqCst)
                    == status.to_u32().unwrap()
                {
                    seqs.push(v.clone());
                }
            }
        } else {
            for (_, v) in self.seqs.iter() {
                seqs.push(v.clone());
            }
        }
        seqs
    }
    pub fn get_unfinished_seqs(&self) -> Vec<SequenceRef> {
        let mut seqs = Vec::new();
        for (_, v) in self.seqs.iter() {
            if !v.borrow().is_finished() {
                seqs.push(v.clone());
            }
        }
        seqs
    }
    pub fn get_finished_seqs(&self) -> Vec<SequenceRef> {
        let mut seqs = Vec::new();
        for (_, v) in self.seqs.iter() {
            if v.borrow().is_finished() {
                seqs.push(v.clone());
            }
        }
        seqs
    }

    pub fn num_seqs(&self, status: Option<SequenceState>) -> usize {
        self.get_seqs(status).len()
    }

    pub fn num_unfinished_seqs(&self) -> usize {
        self.get_unfinished_seqs().len()
    }

    pub fn num_finished_seqs(&self) -> usize {
        self.get_finished_seqs().len()
    }

    pub fn find(&self, seq_id: u64) -> Option<SequenceRef> {
        self.seqs.get(&seq_id).cloned()
    }

    pub fn add(&mut self, seq: Sequence) -> anyhow::Result<()> {
        self.add_ref(seq.new_ref())
    }
    pub fn add_ref(&mut self, seq: SequenceRef) -> anyhow::Result<()> {
        let seq_id = seq.borrow().seq_id;
        if self.seqs.contains_key(&seq_id) {
            return Err(anyhow!("Sequence {} already exists.", seq_id));
        }
        self.seqs.insert(seq_id, seq);
        Ok(())
    }

    pub fn remove(&mut self, seq_id: u64) -> anyhow::Result<()> {
        if self.seqs.remove(&seq_id).is_none() {
            return Err(anyhow!("Sequence {} not found.", seq_id));
        }
        Ok(())
    }

    pub fn is_finished(&self) -> bool {
        for (_, v) in self.seqs.iter() {
            if v.borrow().is_finished() {
                return true;
            }
        }
        false
    }
}

pub struct SequenceGroupMetadata {
    request_id: u64,
    pub(crate) is_prompt: bool,
    pub(crate) seq_data: HashMap<u64, SequenceRef>,
    pub(crate) sampling_params: SamplingParams,
    pub(crate) block_tables: HashMap<u64, Vec<u32>>,
}

impl SequenceGroupMetadata {
    pub fn new(
        request_id: u64,
        is_prompt: bool,
        seq_data: HashMap<u64, SequenceRef>,
        sampling_params: SamplingParams,
        block_tables: HashMap<u64, Vec<u32>>,
    ) -> Self {
        Self {
            request_id,
            is_prompt,
            seq_data,
            sampling_params,
            block_tables,
        }
    }
}
#[derive(PartialEq, Debug)]
pub struct SequenceOutput {
    pub(crate) parent_seq_id: u64,
    pub(crate) output_token: u32,
    pub(crate) logprobs: HashMap<u32, f32>,
}

impl SequenceOutput {
    pub fn new(parent_seq_id: u64, output_token: u32, logprobs: HashMap<u32, f32>) -> Self {
        Self {
            parent_seq_id,
            output_token,
            logprobs,
        }
    }
}

pub type PromptLogprobs = Vec<Option<HashMap<u32, f32>>>;
pub type SampleLogprobs = Vec<HashMap<u32, f32>>;

#[derive(PartialEq, Debug)]
pub struct SequenceGroupOutput {
    pub(crate) samples: Vec<SequenceOutput>,
    pub(crate) prompt_logprobs: Option<PromptLogprobs>,
}

impl SequenceGroupOutput {
    pub fn new(samples: Vec<SequenceOutput>, prompt_logprobs: Option<PromptLogprobs>) -> Self {
        Self {
            samples,
            prompt_logprobs,
        }
    }
}

pub type SamplerOutput = Vec<SequenceGroupOutput>;
