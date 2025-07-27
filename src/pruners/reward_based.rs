use super::base_pruner::BasePruner;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::mann_whitney_u::mann_whitney_u;

use crate::memory::Experience;
use std::collections::HashMap;
use ulid::Ulid;

pub struct RewardBasedPruner {
    p: f64,
    horizon: usize,
    rewards_by_agent: HashMap<Ulid, BoundedVecDeque<f64>>,
}

impl RewardBasedPruner {
    pub fn new(p: f64, horizon: usize) -> Self {
        RewardBasedPruner {
            p,
            horizon,
            rewards_by_agent: HashMap::new(),
        }
    }
}

impl BasePruner for RewardBasedPruner {
    fn step(&mut self, experience: &Experience) {
        self.rewards_by_agent
            .entry(experience.agent_id)
            .or_insert_with(|| BoundedVecDeque::new(self.horizon))
            .push_back(experience.reward_for_this_state);
    }

    fn prune(&self, agent_id: &Ulid) -> bool {
        if let Some(target_agent_rewards) = self.rewards_by_agent.get(agent_id) {
            //TODO
            return mann_whitney_u(&target_agent_rewards.to_vec(), &[], self.p);
        }
        false
    }
}
