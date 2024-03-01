use anyhow::{anyhow, Result};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::common::sequence::{SequenceGroup, SequenceGroupRef};
pub trait Policy: Send + Sync {
    fn get_priority(&self, now: Instant, seq_group: &SequenceGroup) -> usize;

    fn sort_by_priority(&self, now: Instant, seq_groups: &mut VecDeque<SequenceGroupRef>) {
        seq_groups.make_contiguous().sort_by(|x, y| {
            self.get_priority(now, &x.borrow())
                .cmp(&self.get_priority(now, &y.borrow()))
                .reverse()
        });
    }
}

pub(crate) struct FCFS {}

impl Policy for FCFS {
    fn get_priority(&self, now: Instant, seq_group: &SequenceGroup) -> usize {
        //now.saturating_sub(seq_group.arrival_time).as_secs() as usize
        now.duration_since(seq_group.arrival_time).as_micros() as usize
    }
}

pub fn get_policy(name: &str) -> Result<Box<dyn Policy>> {
    match name {
        "fcfs" => Ok(Box::new(FCFS {})),
        _ => Err(anyhow!("unsupoorted policy:{}", name)),
    }
}
