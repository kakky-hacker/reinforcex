use super::base_selector::BaseSelector;
use crate::memory::Experience;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::mann_whitney_u::mann_whitney_u;
use std::collections::HashMap;
use std::sync::Mutex;
use ulid::Ulid;

pub struct RewardBasedSelector {
    z_threshold: Mutex<f64>,
    horizon: usize,
    select_start_size: usize,
    rewards_by_agent: Mutex<HashMap<Ulid, BoundedVecDeque<f64>>>,
}

// z_threshold is the threshold for a one-sided test (5% p-value corresponds to -1.96 in z_threshold).
impl RewardBasedSelector {
    pub fn new(z_threshold: f64, horizon: usize, select_start_size: usize) -> Self {
        assert!(z_threshold < 0.0);
        RewardBasedSelector {
            z_threshold: Mutex::new(z_threshold),
            horizon,
            select_start_size,
            rewards_by_agent: Mutex::new(HashMap::new()),
        }
    }
}

impl BaseSelector for RewardBasedSelector {
    fn observe(&self, experience: &Experience) {
        self.rewards_by_agent
            .lock()
            .unwrap()
            .entry(experience.agent_id)
            .or_insert_with(|| BoundedVecDeque::new(self.horizon))
            .push_back(experience.reward_for_this_state);
    }

    fn delete(&self, agent_id: &Ulid) {
        self.rewards_by_agent.lock().unwrap().remove(agent_id);
    }

    fn find_pareto_dominant(&self, agent_id: &Ulid) -> Vec<Ulid> {
        let rewards_by_agent = {
            let guard = self.rewards_by_agent.lock().unwrap();
            guard
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<Ulid, BoundedVecDeque<f64>>>()
        };
        let mut dominants: Vec<Ulid> = vec![];
        if let Some(target_agent_rewards) = rewards_by_agent.get(&agent_id) {
            if target_agent_rewards.len() < self.select_start_size {
                return dominants;
            }
            for (other_agent_id, other_agent_rewards) in rewards_by_agent.iter() {
                if other_agent_id == agent_id || other_agent_rewards.len() < self.select_start_size
                {
                    continue;
                }
                if mann_whitney_u(
                    &target_agent_rewards.to_vec(),
                    &other_agent_rewards.to_vec(),
                    *self.z_threshold.lock().unwrap(),
                ) {
                    dominants.push(other_agent_id.clone());
                }
            }
            if dominants.len() > 0 {
                *self.z_threshold.lock().unwrap() *= 1.01
            }
        }
        dominants
    }
}
