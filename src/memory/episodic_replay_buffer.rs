use super::experience::Experience;
use crate::memory::experience;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::prob_distributions::BaseDistribution;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::Tensor;
use ulid::Ulid;

pub struct EpisodicReplayBuffer {
    experiences_by_episode: HashMap<Ulid, BoundedVecDeque<Arc<Experience>>>,
    score_for_episode: HashMap<Ulid, f64>,
}

impl EpisodicReplayBuffer {
    pub fn new() -> Self {
        Self {
            experiences_by_episode: HashMap::new(),
            score_for_episode: HashMap::new(),
        }
    }

    pub fn append(&mut self, experience: Arc<Experience>) {
        self.experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        if experience.is_episode_terminal {
            self.score_for_episode.insert(
                experience.episode_id,
                self.experiences_by_episode
                    .get(&experience.episode_id)
                    .unwrap()
                    .to_vec()
                    .iter()
                    .map(|e| e.reward)
                    .sum::<f64>(),
            );
        }
    }

    pub fn get_best_episode(&self) -> Vec<Arc<Experience>> {
        if let Some((key, score)) = self
            .score_for_episode
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if *score > 120.0 {
                return self.experiences_by_episode.get(&key).unwrap().to_vec();
            }
        }
        vec![]
    }

    pub fn remove(&mut self, episode_id: Ulid) {
        self.experiences_by_episode.remove(&episode_id);
    }
}
