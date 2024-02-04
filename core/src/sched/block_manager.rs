use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::common::block::DeviceType;
use crate::common::block::PhysicalTokenBlock;
use crate::common::sequence::Sequence;
use crate::common::sequence::SequenceGroup;
use crate::common::sequence::SequenceState;
use anyhow::anyhow;
use anyhow::Result;
pub struct BlockAllocator {
    device: DeviceType,
    block_size: usize,
    num_blocks: usize,
    free_blocks: VecDeque<Arc<PhysicalTokenBlock>>,
}

impl BlockAllocator {
    pub fn new(device: DeviceType, block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = VecDeque::new();
        for i in 0..num_blocks {
            let block = PhysicalTokenBlock::new(device.clone(), i as u32, block_size);
            free_blocks.push_back(Arc::new(block));
        }
        Self {
            device,
            block_size,
            num_blocks,
            free_blocks,
        }
    }

    pub fn allocate(&mut self) -> Result<Arc<PhysicalTokenBlock>> {
        let block = self
            .free_blocks
            .pop_back()
            .ok_or(anyhow!("Out of memory! No free blocks are available."))?;
        block.inc_ref_count();
        Ok(block)
    }

    pub fn free(&mut self, block: &Arc<PhysicalTokenBlock>) -> Result<()> {
        if block.ref_count() == 0 {
            return Err(anyhow!("Double free! {:?} is already freed.", block));
        }
        block.dec_ref_count();
        if block.ref_count() == 0 {
            self.free_blocks.push_back(block.clone());
        }
        Ok(())
    }
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }
}

pub enum AllocStatus {
    OK,
    LATER,
    NEVER,
}

type BlockTable = Vec<Arc<PhysicalTokenBlock>>;
pub struct BlockSpaceManager {
    block_size: usize,
    num_total_gpu_blocks: usize,
    num_total_cpu_blocks: usize,
    block_sliding_window: Option<usize>,
    watermark_blocks: usize,
    gpu_allocator: BlockAllocator,
    cpu_allocator: BlockAllocator,
    block_tables: HashMap<u64, Vec<Arc<PhysicalTokenBlock>>>,
}

impl BlockSpaceManager {
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        watermark: f64,
        sliding_window: Option<usize>,
    ) -> Self {
        let watermark_blocks = (watermark * num_gpu_blocks as f64).round() as usize;
        let gpu_allocator = BlockAllocator::new(DeviceType::GPU, block_size, num_gpu_blocks);
        let cpu_allocator = BlockAllocator::new(DeviceType::CPU, block_size, num_cpu_blocks);

        Self {
            block_size,
            num_total_gpu_blocks: num_gpu_blocks,
            num_total_cpu_blocks: num_cpu_blocks,
            block_sliding_window: sliding_window,
            watermark_blocks,
            gpu_allocator,
            cpu_allocator,
            block_tables: HashMap::new(),
        }
    }

    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> AllocStatus {
        //     # FIXME(woosuk): Here we assume that all sequences in the group share
        //     # the same prompt. This may not be true for preempted sequences.
        let seq = seq_group.get_seqs(Some(SequenceState::Waiting))[0].clone();
        let mut num_required_blocks = seq.borrow().logical_token_blocks.len();
        if let Some(block_sliding_window) = self.block_sliding_window {
            num_required_blocks = std::cmp::min(num_required_blocks, block_sliding_window);
        }
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();
        //     # Use watermark to avoid frequent cache eviction.

        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks {
            return AllocStatus::NEVER;
        }
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks {
            return AllocStatus::OK;
        }
        return AllocStatus::LATER;
    }
    pub fn allocate(&mut self, seq_group: &SequenceGroup) -> Result<()> {
        // # NOTE: Here we assume that all sequences in the group have the same
        // # prompt.
        let seq = seq_group.get_seqs(Some(SequenceState::Waiting))[0].clone();
        let mut block_table: Vec<Arc<PhysicalTokenBlock>> = Vec::new();
        for logical_idx in 0..seq.borrow().logical_token_blocks.len() {
            let block = if let Some(block_sliding_window) = self.block_sliding_window {
                if logical_idx >= block_sliding_window {
                    //block_table.get_mut(index)
                    block_table[logical_idx % block_sliding_window].clone()
                } else {
                    self.gpu_allocator.allocate()?
                }
            } else {
                self.gpu_allocator.allocate()?
            };
            block.set_ref_count(seq_group.num_seqs(None));
            block_table.push(block);
        }

        // # Assign the block table for each sequence.
        for seq in seq_group.get_seqs(Some(SequenceState::Waiting)) {
            self.block_tables
                .insert(seq.borrow().seq_id, block_table.clone());
        }
        tracing::info!("###After Allocate :{:?}", self.block_tables);
        Ok(())
    }

    pub fn can_append_slot(&self, seq_group: &SequenceGroup) -> bool {
        // # Simple heuristic: If there is at least one free block
        // # for each sequence, we can append.
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();
        let num_seqs = seq_group.num_seqs(Some(SequenceState::Running));
        num_seqs <= num_free_gpu_blocks
    }

    pub fn append_slot(&mut self, seq: &Sequence) -> Result<Option<(u32, u32)>> {
        // """Allocate a physical slot for a new token."""
        let logical_blocks = &seq.logical_token_blocks;
        if let Some(block_table) = self.block_tables.get_mut(&seq.seq_id) {
            if block_table.len() < logical_blocks.len() {
                let block_sliding_window =
                    if let Some(block_sliding_window) = self.block_sliding_window {
                        block_sliding_window
                    } else {
                        0
                    };
                if block_sliding_window > 0 && block_table.len() >= block_sliding_window {
                    //# re-use a block
                    let reuse_block = block_table[block_table.len() % block_sliding_window].clone();
                    block_table.push(reuse_block);
                } else {
                    //         # The sequence has a new logical block.
                    //         # Allocate a new physical block.
                    let block = self.gpu_allocator.allocate()?;
                    block_table.push(block.clone());
                    return Ok(None);
                }
            }
            // # We want to append the token to the last physical block.
            let last_block = block_table.last().unwrap().clone();

            if last_block.ref_count() == 1 {
                // # Not shared with other sequences. Appendable.
                Ok(None)
            } else {
                //     # The last block is shared with other sequences.
                //     # Copy on Write: Allocate a new block and copy the tokens.
                let new_block = self.gpu_allocator.allocate()?;
                block_table.push(new_block.clone());
                let _ = self.gpu_allocator.free(&last_block);
                Ok(Some((last_block.block_number, new_block.block_number)))
            }
        } else {
            tracing::error!("block_tables:{:?}", self.block_tables);
            Err(anyhow!("seq_id:{} not exist in block table", seq.seq_id))
        }
    }

    pub fn fork(&mut self, parent_seq: &Sequence, child_seq: &Sequence) {
        // # NOTE: fork does not allocate a new physical block.
        // # Thus, it is always safe from OOM.
        let src_block_table =
            if let Some(src_block_table) = self.block_tables.get(&parent_seq.seq_id) {
                src_block_table.clone()
            } else {
                return;
            };
        self.block_tables
            .insert(child_seq.seq_id, src_block_table.clone());
        for block in src_block_table {
            block.inc_ref_count();
        }
    }
    fn get_physical_blocks(&self, seq_group: &SequenceGroup) -> Vec<Arc<PhysicalTokenBlock>> {
        // # NOTE: Here, we assume that the physical blocks are only shared by
        // # the sequences in the same group.
        let mut blocks = HashMap::new();
        for seq in seq_group.get_seqs(None) {
            if seq.borrow().is_finished() {
                continue;
            }
            let seq_blocks = self.block_tables.get(&seq.borrow().seq_id).unwrap();
            for seq in seq_blocks {
                blocks.insert(seq.block_number, seq.clone());
            }
        }
        blocks.values().cloned().collect()
    }

    pub fn can_swap_in(&self, seq_group: &SequenceGroup) -> bool {
        let blocks = self.get_physical_blocks(seq_group);
        let num_swapped_seqs = seq_group.num_seqs(Some(SequenceState::Swapped));
        let num_free_blocks = self.gpu_allocator.get_num_free_blocks();
        // # NOTE: Conservatively, we assume that every sequence will allocate
        // # at least one free block right after the swap-in.
        // # NOTE: This should match the logic in can_append_slot().
        let num_required_blocks = blocks.len() + num_swapped_seqs;
        num_free_blocks - num_required_blocks >= self.watermark_blocks
    }

    pub fn swap_in(&mut self, seq_group: &SequenceGroup) -> Result<HashMap<u32, u32>> {
        // # CPU block -> GPU block.
        let mut mapping: HashMap<u32, Arc<PhysicalTokenBlock>> = HashMap::new();
        for seq in seq_group.get_seqs(Some(SequenceState::Swapped)) {
            let mut new_block_table = Vec::new();
            let block_table = self.block_tables.get(&seq.borrow().seq_id).unwrap();
            for cpu_block in block_table {
                let gpu_block = if let Some(gpu_block) = mapping.get(&cpu_block.block_number) {
                    gpu_block.inc_ref_count();
                    gpu_block.clone()
                } else {
                    let gpu_block = self.gpu_allocator.allocate()?;
                    mapping.insert(cpu_block.block_number, gpu_block.clone());
                    gpu_block
                };
                new_block_table.push(gpu_block.clone());
                // # Free the CPU block swapped in to GPU.
                let _ = self.cpu_allocator.free(cpu_block);
            }
            self.block_tables
                .insert(seq.borrow().seq_id, new_block_table);
        }
        let mut block_number_mapping: HashMap<u32, u32> = HashMap::new();
        for (cpu_block_number, gpu_block) in mapping.iter() {
            block_number_mapping.insert(*cpu_block_number, gpu_block.block_number);
        }
        Ok(block_number_mapping)
    }
    pub fn can_swap_out(&self, seq_group: &SequenceGroup) -> bool {
        let blocks = self.get_physical_blocks(seq_group);
        blocks.len() <= self.cpu_allocator.get_num_free_blocks()
    }
    pub fn swap_out(&mut self, seq_group: &SequenceGroup) -> Result<HashMap<u32, u32>> {
        // # GPU block -> CPU block.
        let mut mapping: HashMap<u32, Arc<PhysicalTokenBlock>> = HashMap::new();
        for seq in seq_group.get_seqs(Some(SequenceState::Running)) {
            let mut new_block_table = Vec::new();
            let block_table = self.block_tables.get(&seq.borrow().seq_id).unwrap();
            for gpu_block in block_table {
                let cpu_block = if let Some(cpu_block) = mapping.get(&gpu_block.block_number) {
                    cpu_block.inc_ref_count();
                    cpu_block.clone()
                } else {
                    let cpu_block = self.cpu_allocator.allocate()?;
                    mapping.insert(gpu_block.block_number, cpu_block.clone());
                    cpu_block
                };
                new_block_table.push(cpu_block.clone());
                // # Free the GPU block swapped out to CPU.
                let _ = self.gpu_allocator.free(gpu_block);
            }
            self.block_tables
                .insert(seq.borrow().seq_id, new_block_table);
        }
        let mut block_number_mapping: HashMap<u32, u32> = HashMap::new();
        for (gpu_block_number, cpu_block) in mapping.iter() {
            block_number_mapping.insert(*gpu_block_number, cpu_block.block_number);
        }
        Ok(block_number_mapping)
    }
    fn free_block_table(&mut self, block_table: &BlockTable) {
        for block in block_table {
            if block.is_gpu() {
                let _ = self.gpu_allocator.free(block);
            } else {
                let _ = self.cpu_allocator.free(block);
            }
        }
    }
    pub fn free(&mut self, seq_id: u64) {
        if let Some(block_table) = self.block_tables.remove(&seq_id) {
            self.free_block_table(&block_table);
        }
        tracing::info!(
            "###After free:{} block_tables:{:?}",
            seq_id,
            self.block_tables
        );
    }

    pub fn reset(&mut self) {
        let block_tables = self.block_tables.clone();
        for block_table in block_tables.values() {
            self.free_block_table(block_table);
        }
        drop(block_tables);
        self.block_tables.clear();
    }

    pub fn get_block_table(&self, seq_id: u64) -> Vec<u32> {
        let block_table = self.block_tables.get(&seq_id).unwrap();
        // let block_numbers = Vec::new();
        block_table.iter().map(|v| v.block_number).collect()
    }
    pub fn get_num_free_gpu_blocks(&self) -> usize {
        self.gpu_allocator.get_num_free_blocks()
    }

    pub fn get_num_free_cpu_blocks(&self) -> usize {
        self.cpu_allocator.get_num_free_blocks()
    }
}
