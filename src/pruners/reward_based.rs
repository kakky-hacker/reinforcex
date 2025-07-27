use super::base_pruner::BasePruner;
use crate::memory::TransitionBuffer;
use std::sync::Arc;

pub struct RewardBasedPruner {
    transition_buffer: Arc<TransitionBuffer>,
}

impl RewardBasedPruner {
    pub fn new(transition_buffer: Arc<TransitionBuffer>) -> Self {
        RewardBasedPruner { transition_buffer }
    }
}

impl BasePruner for RewardBasedPruner {
    fn step(&mut self) {}

    fn prune(&self, agent_id: ulid::Ulid) -> bool {
        false
    }
}
