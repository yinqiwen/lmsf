use std::sync::atomic::Ordering;
#[derive(Clone)]
pub struct LogicalTokenBlock {
    block_number: u32,
    block_size: usize,
    token_ids: Vec<u32>,
}

impl LogicalTokenBlock {
    pub fn new(block_number: u32, block_size: usize) -> Self {
        Self {
            block_number,
            block_size,
            token_ids: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
    pub fn is_full(&self) -> bool {
        self.token_ids.len() == self.block_size
    }
    pub fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.token_ids.len()
    }
    pub fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub fn get_last_token_id(&self) -> u32 {
        self.token_ids[self.token_ids.len() - 1]
    }
    pub fn append_tokens(&mut self, token_ids: &[u32]) {
        self.token_ids.extend_from_slice(token_ids);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    CPU,
    GPU,
}

#[derive(Debug)]
pub struct PhysicalTokenBlock {
    pub(crate) block_number: u32,
    block_size: usize,
    ref_count: std::sync::atomic::AtomicUsize,
    device: DeviceType,
}

impl PhysicalTokenBlock {
    pub fn new(device: DeviceType, block_number: u32, block_size: usize) -> Self {
        Self {
            device,
            block_number,
            block_size,
            ref_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    pub fn is_gpu(&self) -> bool {
        self.device == DeviceType::GPU
    }
    pub fn set_ref_count(&self, n: usize) {
        self.ref_count.store(n, Ordering::SeqCst);
    }
    pub fn inc_ref_count(&self) {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
    }
    pub fn dec_ref_count(&self) {
        self.ref_count.fetch_sub(1, Ordering::SeqCst);
    }
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }
}
