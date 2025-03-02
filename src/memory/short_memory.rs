use crate::memory::{Experience, ReplayBuffer};
use std::rc::Rc;
use tch::Tensor;

pub struct ShortMemory {
    replay_buffer: ReplayBuffer,
}

impl ShortMemory {
    pub fn new(capacity: usize, n_steps: usize) -> Self {
        Self {
            replay_buffer: ReplayBuffer::new(capacity, n_steps),
        }
    }

    pub fn append(
        &mut self,
        state: Tensor,
        action: Option<Tensor>,
        reward: f64,
        is_episode_terminal: bool,
        gamma: f64,
    ) {
        self.replay_buffer
            .append(state, action, reward, is_episode_terminal, gamma);
    }

    pub fn sample(&mut self) -> Vec<&Rc<Experience>> {
        let experiences: Vec<&Rc<Experience>> =
            self.replay_buffer.sample(self.replay_buffer.len(), true);
        experiences
    }

    pub fn len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn clear(&mut self) {
        self.replay_buffer.clear();
    }
}
