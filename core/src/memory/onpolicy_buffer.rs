use super::experience::Experience;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use std::{collections::HashMap, sync::Arc};
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

    pub fn append(&mut self, experience: Arc<Experience>) {
        self.experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience);
    }

    pub fn flush(&mut self) -> Vec<Vec<Arc<Experience>>> {
        self.experiences_by_episode
            .drain()
            .map(|(_k, v)| v.to_vec())
            .collect()
    }
}
