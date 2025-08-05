use super::replay_buffer::{Experience, ReplayBuffer};
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::Tensor;
use ulid::Ulid;

pub struct OnPolicyBuffer {
    experiences_by_episode: HashMap<Ulid, BoundedVecDeque<Arc<Experience>>>,
    replay_buffer: Option<Arc<ReplayBuffer>>,
}

impl OnPolicyBuffer {
    pub fn new(replay_buffer: Option<Arc<ReplayBuffer>>) -> Self {
        Self {
            experiences_by_episode: HashMap::new(),
            replay_buffer,
        }
    }

    pub fn append(
        &mut self,
        agent_id: Ulid,
        episode_id: Ulid,
        state: Tensor,
        action: Option<Tensor>,
        reward: f64,
        is_episode_terminal: bool,
        gamma: f64,
    ) -> Arc<Experience> {
        let experience;

        if self.replay_buffer.is_some() {
            experience = self.replay_buffer.as_ref().unwrap().append(
                agent_id,
                episode_id,
                state,
                action,
                reward,
                is_episode_terminal,
                gamma,
            );
        } else {
            experience = Arc::new(Experience {
                agent_id,
                episode_id,
                state,
                action,
                reward,
                n_step_discounted_reward: Mutex::new(None),
                n_step_after_experience: Mutex::new(None),
                is_episode_terminal,
            });
        }

        self.experiences_by_episode
            .entry(episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        experience
    }

    pub fn clear(&mut self) {
        self.experiences_by_episode = HashMap::new();
    }
}
