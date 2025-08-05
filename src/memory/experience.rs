use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::Tensor;
use ulid::Ulid;

pub struct Experience {
    pub agent_id: Ulid,
    pub episode_id: Ulid,
    pub state: Tensor,
    pub action: Option<Tensor>,
    pub reward_for_this_state: f64,
    pub is_episode_terminal: bool,
    pub n_step_discounted_reward: Mutex<Option<f64>>,
    pub n_step_after_experience: Mutex<Option<Arc<Experience>>>,
}

// Tensor does not implement Sync due to raw pointer, so we promise safety manually
unsafe impl Sync for Experience {}
