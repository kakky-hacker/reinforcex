use crate::prob_distributions::BaseDistribution;
use std::sync::{Arc, Mutex};
use tch::Tensor;
use ulid::Ulid;

pub struct Experience {
    pub agent_id: Ulid,
    pub episode_id: Ulid,
    pub state: Tensor,
    pub action: Option<Tensor>,
    pub action_distrib: Option<Box<dyn BaseDistribution>>,
    pub reward: f64,
    pub is_episode_terminal: bool,
    pub n_step_discounted_reward: Mutex<Option<f64>>,
    pub n_step_after_experience: Mutex<Option<Arc<Experience>>>,
}

// Tensor does not implement Sync due to raw pointer, so we promise safety manually
unsafe impl Sync for Experience {}

impl Experience {
    pub fn new(
        agent_id: Ulid,
        episode_id: Ulid,
        state: Tensor,
        action: Option<Tensor>,
        action_distrib: Option<Box<dyn BaseDistribution>>,
        reward: f64,
        is_episode_terminal: bool,
        n_step_discounted_reward: Mutex<Option<f64>>,
        n_step_after_experience: Mutex<Option<Arc<Experience>>>,
    ) -> Self {
        Self {
            agent_id,
            episode_id,
            state,
            action,
            action_distrib,
            reward,
            is_episode_terminal,
            n_step_discounted_reward,
            n_step_after_experience,
        }
    }
}
