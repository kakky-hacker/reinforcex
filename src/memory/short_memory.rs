use crate::memory::{Experience, ReplayBuffer};
use candle_core::Tensor;
use std::rc::Rc;

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_append_and_len() {
        let mut short_memory = ShortMemory::new(10, 1);
        assert_eq!(short_memory.len(), 0);

        let state = Tensor::from_slice(&[0.0, 1.0], &[2], &Device::Cpu).unwrap();
        short_memory.append(state.clone(), None, 1.0, false, 0.99);
        short_memory.append(state.clone(), None, 1.0, false, 0.99);
        assert_eq!(short_memory.len(), 1);
    }

    #[test]
    fn test_sample_and_clear() {
        let mut short_memory = ShortMemory::new(10, 1);
        let state = Tensor::from_slice(&[0.0, 1.0], &[2], &Device::Cpu).unwrap();
        short_memory.append(state.clone(), None, 1.0, false, 0.99);
        short_memory.append(state.clone(), None, 0.5, false, 0.99);
        short_memory.append(state.clone(), None, 0.5, false, 0.99);

        let samples = short_memory.sample();
        assert_eq!(samples.len(), 2);
        short_memory.clear();
        assert_eq!(short_memory.len(), 0);
    }
}
