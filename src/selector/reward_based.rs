use super::base_selector::BaseSelector;
use crate::memory::Experience;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::mann_whitney_u::mann_whitney_u;
use rayon::range;
use std::collections::HashMap;
use std::sync::Mutex;
use ulid::Ulid;

pub struct RewardBasedSelector {
    z_threshold: f64,
    horizon: usize,
    rewards_by_agent: HashMap<Ulid, Mutex<BoundedVecDeque<f64>>>,
}

// z_threshold is the threshold for a one-sided test (5% p-value corresponds to -1.96 in z_threshold).
impl RewardBasedSelector {
    pub fn new(horizon: usize, z_threshold: f64) -> Self {
        assert!(z_threshold < 0.0);
        RewardBasedSelector {
            z_threshold,
            horizon,
            rewards_by_agent: HashMap::new(),
        }
    }
}

impl BaseSelector for RewardBasedSelector {
    fn observe(&mut self, experience: &Experience) {
        self.rewards_by_agent
            .entry(experience.agent_id)
            .or_insert_with(|| Mutex::new(BoundedVecDeque::new(self.horizon)))
            .lock()
            .unwrap()
            .push_back(experience.reward_for_this_state);
    }

    fn prune(&self, agent_id: &Ulid) -> bool {
        if let Some(target_agent_rewards) = self.rewards_by_agent.get(agent_id) {
            for other_agent_rewards in self.rewards_by_agent.values() {
                if mann_whitney_u(
                    &target_agent_rewards.lock().unwrap().to_vec(),
                    &other_agent_rewards.lock().unwrap().to_vec(),
                    self.z_threshold,
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn select_next_parents(&self, agent_id: &Ulid) -> Vec<&Ulid> {
        vec![]
    }
}
