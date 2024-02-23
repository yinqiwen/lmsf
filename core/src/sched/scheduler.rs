use std::cell::RefCell;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
    sync::Arc,
};

use crate::common::sequence::{SequenceGroupRef, SequenceRef};
use crate::{
    common::{
        config::{CacheConfig, SchedulerConfig},
        sequence::{Sequence, SequenceGroup, SequenceState},
    },
    sched::block_manager::AllocStatus,
};
use anyhow::{anyhow, Result};

use super::{
    block_manager::BlockSpaceManager,
    policy::{get_policy, Policy},
};

pub enum PreemptionMode {
    // """Preemption modes.

    // 1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    // and swap them back in when the sequences are resumed.
    // 2. Recomputation: Discard the blocks of the preempted sequences and
    // recompute them when the sequences are resumed, treating the sequences as
    // new prompts.
    // """
    Swap,
    Recompute,
}
pub struct SchedulerOutputs {
    pub(crate) scheduled_seq_groups: Vec<SequenceGroupRef>,
    pub(crate) prompt_run: bool,
    pub(crate) num_batched_tokens: usize,
    pub(crate) blocks_to_swap_in: HashMap<u32, u32>,
    pub(crate) blocks_to_swap_out: HashMap<u32, u32>,
    pub(crate) blocks_to_copy: HashMap<u32, Vec<u32>>,
    pub(crate) ignored_seq_groups: Vec<SequenceGroupRef>,
}
impl SchedulerOutputs {
    pub fn is_empty(&self) -> bool {
        // # NOTE: We do not consider the ignored sequence groups.
        self.scheduled_seq_groups.is_empty()
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_copy.is_empty()
            && self.blocks_to_swap_out.is_empty()
    }
}

pub struct Scheduler {
    pub(crate) scheduler_config: SchedulerConfig,
    cache_config: CacheConfig,
    block_manager: BlockSpaceManager,
    prompt_limit: usize,

    seqs: HashMap<u64, Sequence>,
    seq_group_queues: [VecDeque<SequenceGroupRef>; 3],
    policy: Box<dyn Policy>,
    // waiting: VecDeque<Arc<SequenceGroup>>,
    // running: VecDeque<Arc<SequenceGroup>>,
    // swapped: VecDeque<Arc<SequenceGroup>>,
}
impl Scheduler {
    const WAITING: usize = 0;
    const RUNNING: usize = 1;
    const SWAPPED: usize = 2;
    pub fn new(
        scheduler_config: &SchedulerConfig,
        cache_config: &CacheConfig,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> Self {
        let prompt_limit = std::cmp::min(
            scheduler_config.max_model_len,
            scheduler_config.max_num_batched_tokens,
        );
        let block_manager = BlockSpaceManager::new(
            cache_config.block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            0.01,
            cache_config.sliding_window,
        );

        let policy = get_policy("fcfs").unwrap();
        Self {
            scheduler_config: scheduler_config.clone(),
            cache_config: cache_config.clone(),
            block_manager,
            prompt_limit,
            policy,
            seqs: HashMap::new(),
            seq_group_queues: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
        }
    }

    pub fn add_seq_group(&mut self, seq_group: SequenceGroup) {
        // # Add sequence groups to the waiting queue.
        self.seq_group_queues[Scheduler::WAITING].push_back(Rc::new(RefCell::new(seq_group)));
    }
    pub fn free_seq(&mut self, seq_id: u64) {
        // tracing::info!("free_seq:{}", seq_id);
        self.block_manager.free(seq_id);
    }
    pub fn fork_seq(&mut self, parent_seq: &Sequence, child_seq: &Sequence) {
        self.block_manager.fork(parent_seq, child_seq);
    }
    pub fn free_finished_seq_groups(&mut self) {
        let new_vecs: VecDeque<_> = self.seq_group_queues[Scheduler::RUNNING]
            .clone()
            .into_iter()
            .filter(|seq_group| !seq_group.borrow().is_finished())
            .collect();
        self.seq_group_queues[Scheduler::RUNNING] = new_vecs;
    }

    fn abort_queue_seq_group<'a>(
        queue: &'a mut VecDeque<SequenceGroupRef>,
        mut request_ids: HashSet<u64>,
    ) -> Vec<SequenceRef> {
        let mut free_seqs = Vec::new();
        for seq_group in queue.iter().rev() {
            let seq_group = seq_group.borrow();
            // # Remove the sequence group from the state queue.
            if request_ids.contains(&seq_group.request_id) {
                for iter_seq in seq_group.get_seqs(None) {
                    let seq = iter_seq.borrow();
                    if seq.is_finished() {
                        continue;
                    }
                    seq.update_state(crate::common::sequence::SequenceState::FinishedAborted);
                    free_seqs.push(iter_seq.clone());
                }
                request_ids.remove(&seq_group.request_id);
            }
            if request_ids.is_empty() {
                break;
            }
        }
        free_seqs
    }
    pub fn abort_seq_group(&mut self, orig_request_ids: HashSet<u64>) {
        for idx in 0..=Scheduler::SWAPPED {
            //         # We need to reverse the list as we are removing elements
            //         # from it as we iterate over it. If we don't do it,
            //         # indices will get messed up and we will skip over elements.
            let seq_group_queue = &mut self.seq_group_queues[idx];
            let mut request_ids = orig_request_ids.clone();
            let mut free_seqs = Vec::new();
            let mut remove_seq_groups = Vec::new();
            for (idx, seq_group) in seq_group_queue.iter().enumerate().rev() {
                let seq_group = seq_group.borrow();
                if request_ids.contains(&seq_group.request_id) {
                    // # Remove the sequence group from the state queue.
                    remove_seq_groups.push(idx);
                    for seq in seq_group.get_seqs(None) {
                        let seq = seq.borrow();
                        if seq.is_finished() {
                            continue;
                        }
                        seq.update_state(SequenceState::FinishedAborted);
                        //self.free_seq(seq.seq_id);
                        free_seqs.push(seq.seq_id);
                    }
                    request_ids.remove(&seq_group.request_id);
                }
                if request_ids.is_empty() {
                    break;
                }
            }
            for idx in remove_seq_groups {
                seq_group_queue.remove(idx);
            }
            for seq_id in free_seqs {
                self.free_seq(seq_id);
            }
        }
    }

    fn allocate(&mut self, seq_group: &SequenceGroup) -> Result<()> {
        self.block_manager.allocate(seq_group)?;
        for seq in seq_group.get_seqs(None) {
            seq.borrow().update_state(SequenceState::Running);
        }
        Ok(())
    }

    fn append_slot(
        &mut self,
        seq_group: &SequenceGroup,
        mut blocks_to_copy: &mut HashMap<u32, Vec<u32>>,
    ) -> Result<()> {
        for seq in seq_group.get_seqs(Some(SequenceState::Running)) {
            if let Some((src_block, dst_block)) = self.block_manager.append_slot(&seq.borrow())? {
                if let Some(dst_blocks) = blocks_to_copy.get_mut(&src_block) {
                    dst_blocks.push(dst_block);
                } else {
                    blocks_to_copy.insert(src_block, vec![dst_block]);
                }
            }
        }
        Ok(())
    }

    pub fn has_unfinished_seqs(&self) -> bool {
        self.seq_group_queues
            .iter()
            .map(|seq_group| !seq_group.is_empty())
            .fold(false, |acc, mk| acc || mk)
    }
    pub fn get_num_unfinished_seq_groups(&self) -> usize {
        self.seq_group_queues
            .iter()
            .map(|seq_group| seq_group.len())
            .sum()
    }

    fn preempt(
        &mut self,
        seq_group: SequenceGroupRef,
        blocks_to_swap_out: &mut HashMap<u32, u32>,
        mut preemption_mode: Option<PreemptionMode>,
    ) -> Result<()> {
        // # If preemption mode is not specified, we determine the mode as follows:
        // # We use recomputation by default since it incurs lower overhead than
        // # swapping. However, when the sequence group has multiple sequences
        // # (e.g., beam search), recomputation is not currently supported. In
        // # such a case, we use swapping instead.
        // # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        // # As swapped sequences are prioritized over waiting sequences,
        // # sequence groups with multiple sequences are implicitly prioritized
        // # over sequence groups with a single sequence.
        // # TODO(woosuk): Support recomputation for sequence groups with multiple
        // # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode.is_none() {
            if seq_group.borrow().get_max_num_running_seqs() == 1 {
                preemption_mode = Some(PreemptionMode::Recompute);
            } else {
                preemption_mode = Some(PreemptionMode::Swap);
            }
        }
        match preemption_mode {
            Some(PreemptionMode::Recompute) => {
                self.preempt_by_recompute(seq_group);
            }
            Some(PreemptionMode::Swap) => {
                self.preempt_by_swap(seq_group, blocks_to_swap_out)?;
            }
            None => {
                return Err(anyhow!("Invalid preemption mode."));
            }
        }
        Ok(())
    }

    fn preempt_by_recompute(&mut self, seq_group: SequenceGroupRef) {
        tracing::info!("preempt_by_recompute");
        for seq in seq_group.borrow().get_seqs(Some(SequenceState::Running)) {
            let seq = seq.borrow();
            seq.update_state(SequenceState::Waiting);
            self.block_manager.free(seq.seq_id);
        }
        // # NOTE: For FCFS, we insert the preempted sequence group to the front
        // # of the waiting queue.
        self.seq_group_queues[Scheduler::WAITING].insert(0, seq_group);
    }

    fn preempt_by_swap(
        &mut self,
        seq_group: SequenceGroupRef,
        blocks_to_swap_out: &mut HashMap<u32, u32>,
    ) -> Result<()> {
        self.swap_out(seq_group.clone(), blocks_to_swap_out)?;
        self.seq_group_queues[Scheduler::SWAPPED].push_back(seq_group);
        Ok(())
    }

    fn swap_in(
        &mut self,
        seq_group: SequenceGroupRef,
        blocks_to_swap_in: &mut HashMap<u32, u32>,
    ) -> Result<()> {
        let mapping = self.block_manager.swap_in(&seq_group.borrow())?;
        for (k, v) in mapping {
            blocks_to_swap_in.insert(k, v);
        }
        for seq in seq_group.borrow().get_seqs(Some(SequenceState::Swapped)) {
            let seq = seq.borrow();
            seq.update_state(SequenceState::Running);
        }
        Ok(())
    }
    fn swap_out(
        &mut self,
        seq_group: SequenceGroupRef,
        blocks_to_swap_out: &mut HashMap<u32, u32>,
    ) -> Result<()> {
        if !self.block_manager.can_swap_out(&seq_group.borrow()) {
            // # FIXME(woosuk): Abort the sequence group instead of aborting the
            // # entire engine.
            return Err(anyhow!("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error."));
        }
        let mapping = self.block_manager.swap_out(&seq_group.borrow())?;
        for (k, v) in mapping {
            blocks_to_swap_out.insert(k, v);
        }
        for seq in seq_group.borrow().get_seqs(Some(SequenceState::Running)) {
            let seq = seq.borrow();
            seq.update_state(SequenceState::Swapped);
        }
        Ok(())
    }

    pub fn get_block_table(&self, seq_id: u64) -> Vec<u32> {
        self.block_manager.get_block_table(seq_id)
    }

    pub fn schedule(&mut self) -> Result<SchedulerOutputs> {
        // # Blocks that need to be swaped or copied before model execution.
        let mut blocks_to_swap_in = HashMap::new();
        let mut blocks_to_swap_out = HashMap::new();
        let mut blocks_to_copy = HashMap::new();

        // # Join waiting sequences if possible.
        if self.seq_group_queues[Scheduler::SWAPPED].is_empty() {
            let mut ignored_seq_groups = Vec::new();
            let mut scheduled = Vec::new();
            // # The total number of sequences on the fly, including the
            // # requests in the generation phase.
            let mut num_curr_seqs: usize = self.seq_group_queues[Scheduler::RUNNING]
                .iter()
                .map(|x| x.borrow().get_max_num_running_seqs())
                .sum();
            let mut seq_lens = Vec::new();
            // # Optimization: We do not sort the waiting queue since the preempted
            // # sequence groups are added to the front and the new sequence groups
            // # are added to the back.
            loop {
                if self.seq_group_queues[Scheduler::WAITING].is_empty() {
                    break;
                }
                let seq_group = self.seq_group_queues[Scheduler::WAITING][0].clone();
                let waiting_seqs = seq_group.borrow().get_seqs(Some(SequenceState::Waiting));
                assert!(
                    waiting_seqs.len() == 1,
                    "Waiting sequence group should have only one prompt sequence."
                );

                let num_prompt_tokens = waiting_seqs[0].borrow().get_len();
                if num_prompt_tokens > self.prompt_limit {
                    tracing::warn!(
                        "Input prompt ({} tokens) is too long and exceeds limit of {}",
                        num_prompt_tokens,
                        self.prompt_limit
                    );
                    for seq in waiting_seqs {
                        let seq = seq.borrow();
                        seq.update_state(SequenceState::FinishedIgnored);
                    }
                    ignored_seq_groups.push(seq_group.clone());
                    self.seq_group_queues[Scheduler::WAITING].pop_front();
                    continue;
                }
                // # If the sequence group cannot be allocated, stop.

                let can_allocate = self.block_manager.can_allocate(&seq_group.borrow());
                match can_allocate {
                    AllocStatus::LATER => {
                        tracing::warn!("Input prompt ({} tokens) is too long and exceeds the capacity of block_manager", num_prompt_tokens);
                        for seq in seq_group.borrow().get_seqs(None) {
                            seq.borrow().update_state(SequenceState::FinishedIgnored);
                        }
                        ignored_seq_groups.push(seq_group.clone());
                        self.seq_group_queues[Scheduler::WAITING].pop_front();
                        break;
                    }
                    AllocStatus::NEVER => {
                        continue;
                    }
                    _ => {
                        //
                    }
                }

                // # If the number of batched tokens exceeds the limit, stop.
                // # exceed the maximum number of sequences.
                let mut new_seq_lens = seq_lens.clone();
                new_seq_lens.push(num_prompt_tokens);
                let num_batched_tokens = new_seq_lens.len() * new_seq_lens.iter().max().unwrap();
                if num_batched_tokens > self.scheduler_config.max_num_batched_tokens {
                    break;
                }
                // # The total number of sequences in the RUNNING state should not
                // # exceed the maximum number of sequences.
                let num_new_seqs = seq_group.borrow().get_max_num_running_seqs();
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs {
                    break;
                }
                let num_paddings: usize = num_batched_tokens - new_seq_lens.iter().sum::<usize>();
                if num_paddings > self.scheduler_config.max_paddings {
                    break;
                }

                seq_lens = new_seq_lens;
                let seq_group = self.seq_group_queues[Scheduler::WAITING]
                    .pop_front()
                    .unwrap();
                self.allocate(&seq_group.borrow())?;
                self.seq_group_queues[Scheduler::RUNNING].push_back(seq_group.clone());
                num_curr_seqs += num_new_seqs;
                scheduled.push(seq_group);
            }
            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                let num_batched_tokens = if seq_lens.is_empty() {
                    0
                } else {
                    seq_lens.len() * seq_lens.iter().max().unwrap()
                };
                return Ok(SchedulerOutputs {
                    scheduled_seq_groups: scheduled,
                    prompt_run: true,
                    num_batched_tokens,
                    blocks_to_swap_in,
                    blocks_to_swap_out,
                    blocks_to_copy,
                    ignored_seq_groups,
                });
            }
        }

        // # NOTE(woosuk): Preemption happens only when there is no available slot
        // # to keep all the sequence groups in the RUNNING state.
        // # In this case, the policy is responsible for deciding which sequence
        // # groups to preempt.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        self.policy
            .sort_by_priority(now, &mut self.seq_group_queues[Scheduler::RUNNING]);

        let mut preempted = Vec::new();
        let mut running = VecDeque::new();
        loop {
            if self.seq_group_queues[Scheduler::RUNNING].is_empty() {
                break;
            }
            let seq_group = self.seq_group_queues[Scheduler::RUNNING]
                .pop_front()
                .unwrap();
            let mut append_slot = true;
            while !self.block_manager.can_append_slot(&seq_group.borrow()) {
                if !self.seq_group_queues[Scheduler::RUNNING].is_empty() {
                    // # Preempt the lowest-priority sequence groups.
                    let victim_seq_group = self.seq_group_queues[Scheduler::RUNNING]
                        .pop_back()
                        .unwrap();
                    self.preempt(victim_seq_group.clone(), &mut blocks_to_swap_out, None)?;
                    preempted.push(victim_seq_group);
                } else {
                    // # No other sequence groups can be preempted.
                    // # Preempt the current sequence group.
                    self.preempt(seq_group.clone(), &mut blocks_to_swap_out, None)?;
                    preempted.push(seq_group.clone());
                    append_slot = false;
                    break;
                }
            }
            // # Append new slots to the sequence group.
            if append_slot {
                self.append_slot(&seq_group.borrow(), &mut blocks_to_copy)?;
                running.push_back(seq_group);
            }
        }
        self.seq_group_queues[Scheduler::RUNNING] = running;

        // # Swap in the sequence groups in the SWAPPED state if possible.
        self.policy
            .sort_by_priority(now, &mut self.seq_group_queues[Scheduler::SWAPPED]);
        if preempted.is_empty() {
            let mut num_curr_seqs: usize = self.seq_group_queues[Scheduler::RUNNING]
                .iter()
                .map(|x| x.borrow().get_max_num_running_seqs())
                .sum();
            loop {
                if self.seq_group_queues[Scheduler::SWAPPED].is_empty() {
                    break;
                }
                let seq_group = self.seq_group_queues[Scheduler::SWAPPED][0].clone();
                //  # If the sequence group cannot be swapped in, stop.
                if !self.block_manager.can_swap_in(&seq_group.borrow()) {
                    break;
                }
                // # The total number of sequences in the RUNNING state should not
                //  # exceed the maximum number of sequences.
                let num_new_seqs = seq_group.borrow().get_max_num_running_seqs();
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs {
                    break;
                }
                let seq_group = self.seq_group_queues[Scheduler::SWAPPED]
                    .pop_front()
                    .unwrap();
                self.swap_in(seq_group.clone(), &mut blocks_to_swap_in)?;
                self.append_slot(&seq_group.borrow(), &mut blocks_to_copy)?;
                num_curr_seqs += num_new_seqs;
                self.seq_group_queues[Scheduler::RUNNING].push_back(seq_group);
            }
        }
        // # Each sequence in the generation phase only takes one token slot.
        // # Therefore, the number of batched tokens is equal to the number of
        // # sequences in the RUNNING state.
        let num_batched_tokens = self.seq_group_queues[Scheduler::RUNNING]
            .iter()
            .map(|x| x.borrow().num_seqs(Some(SequenceState::Running)))
            .sum();

        let scheduler_outputs = SchedulerOutputs {
            scheduled_seq_groups: Vec::from(self.seq_group_queues[Scheduler::RUNNING].clone()),
            prompt_run: false,
            num_batched_tokens,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            ignored_seq_groups: Vec::new(),
        };

        Ok(scheduler_outputs)
    }
}
