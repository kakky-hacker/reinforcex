use std::rc::Rc;
use tch::Tensor;
use crate::misc::random_access_queue::RandomAccessQueue;
use crate::misc::bounded_vec_deque::BoundedVecDeque;

pub struct ReplayBuffer {
    memory: RandomAccessQueue<Rc<Experience>>,
    last_n_experiences: BoundedVecDeque<Rc<Experience>>,
}

pub struct Experience {
    state: Tensor,
    action: Option<Tensor>,
    reward: f64,
    prev_n_experiences: Vec<Rc<Experience>>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, n_steps: usize) -> Self {
        assert!(capacity > 0);
        assert!(n_steps > 0);
        ReplayBuffer {
            memory: RandomAccessQueue::new(capacity),
            last_n_experiences: BoundedVecDeque::new(n_steps),
        }
    }

    pub fn append(
        &mut self,
        state: Tensor,
        action: Option<Tensor>,
        reward: f64,
        is_state_terminal: bool,
    ) {
        let prev_n_experiences: Vec<Rc<Experience>> = self.last_n_experiences.clone().into_iter().collect();

        let experience: Rc<Experience> = Rc::new(Experience {
            state,
            action,
            reward,
            prev_n_experiences,
        });

        self.last_n_experiences.push_back(Rc::clone(&experience));
        self.memory.append(experience);

        if is_state_terminal {
            self.last_n_experiences.empty();
        }
    }

    pub fn sample(&self, num_experiences: usize) -> Vec<&Rc<Experience>> {
        self.memory.sample(num_experiences)
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_replay_buffer_new() {
        let buffer = ReplayBuffer::new(100, 5);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_replay_buffer_append_and_len() {
        let mut buffer = ReplayBuffer::new(100, 5);
        let state = Tensor::from_slice(&[1.0]);
        buffer.append(state, None, 1.0, false);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = ReplayBuffer::new(100, 5);
        for i in 0..10 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, false);
        }
        let samples = buffer.sample(3);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_replay_buffer_terminal_state() {
        let mut buffer = ReplayBuffer::new(100, 5);
        for i in 0..5 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, i == 4);
        }
        assert_eq!(buffer.last_n_experiences.clone().len(), 0);
    }
}