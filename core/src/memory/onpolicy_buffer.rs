use super::experience::Experience;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::prob_distributions::BaseDistribution;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::Tensor;
use ulid::Ulid;

pub struct OnPolicyBuffer {
    experiences_by_episode: HashMap<Ulid, BoundedVecDeque<Arc<Experience>>>,
}

impl OnPolicyBuffer {
    pub fn new() -> Self {
        Self {
            experiences_by_episode: HashMap::new(),
        }
    }

    pub fn append(
        &mut self,
        agent_id: Ulid,
        episode_id: Ulid,
        state: Tensor,
        action: Option<Tensor>,
        action_distrib: Option<Box<dyn BaseDistribution>>,
        reward: f64,
        is_episode_terminal: bool,
    ) -> Arc<Experience> {
        let experience = Arc::new(Experience::new(
            agent_id,
            episode_id,
            state,
            action,
            action_distrib,
            reward,
            is_episode_terminal,
            Mutex::new(None),
            Mutex::new(None),
        ));

        self.experiences_by_episode
            .entry(episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        experience
    }

    pub fn flush(&mut self) -> Vec<Vec<Arc<Experience>>> {
        self.experiences_by_episode
            .drain()
            .map(|(_k, v)| v.to_vec())
            .collect()
    }
}
