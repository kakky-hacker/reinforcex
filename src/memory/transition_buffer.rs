use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum;
use crate::misc::random_access_queue::RandomAccessQueue;
use crate::prob_distributions::BaseDistribution;
use std::cell::RefCell;
use std::rc::Rc;
use tch::{NoGradGuard, Tensor};

pub struct TransitionBuffer {
    memory: RandomAccessQueue<Rc<Experience>>,
    last_n_experiences: BoundedVecDeque<Rc<Experience>>,
}

pub struct Experience {
    pub state: Tensor,
    pub action: Option<Tensor>,
    pub reward_for_this_state: f64,
    pub n_step_discounted_reward: RefCell<Option<f64>>,
    pub n_step_after_experience: RefCell<Option<Rc<Experience>>>,
    pub is_episode_terminal: bool,
}

impl TransitionBuffer {
    pub fn new(capacity: usize, n_steps: usize) -> Self {
        assert!(capacity > 0);
        assert!(n_steps > 0);
        TransitionBuffer {
            memory: RandomAccessQueue::new(capacity),
            last_n_experiences: BoundedVecDeque::new(n_steps),
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
        let experience = Rc::new(Experience {
            state,
            action,
            reward_for_this_state: reward,
            n_step_discounted_reward: RefCell::new(None),
            n_step_after_experience: RefCell::new(None),
            is_episode_terminal,
        });

        if !self.last_n_experiences.is_empty() {
            let mut rewards: Vec<f64> = self
                .last_n_experiences
                .clone()
                .into_iter()
                .skip(1)
                .map(|e| e.reward_for_this_state)
                .collect();
            rewards.push(reward);
            let n_step_discounted_reward = cumsum::cumsum_rev(&rewards, gamma)[0];
            *self
                .last_n_experiences
                .front_mut()
                .n_step_discounted_reward
                .borrow_mut() = Some(n_step_discounted_reward);
            *self
                .last_n_experiences
                .front_mut()
                .n_step_after_experience
                .borrow_mut() = Some(Rc::clone(&experience));
        }

        let n_step_before_experience = self.last_n_experiences.push_back(Rc::clone(&experience));
        if n_step_before_experience.is_some() {
            self.memory.append(n_step_before_experience.unwrap());
        }

        if is_episode_terminal {
            let mut rewards: Vec<f64> = self
                .last_n_experiences
                .clone()
                .into_iter()
                .skip(1)
                .map(|e| e.reward_for_this_state)
                .collect();
            rewards.push(0.0); // For terminal state
            let q_values = cumsum::cumsum_rev(&rewards, gamma);
            for (experience, &n_step_discounted_reward) in self
                .last_n_experiences
                .clone()
                .into_iter()
                .zip(q_values.iter())
            {
                *experience.n_step_discounted_reward.borrow_mut() = Some(n_step_discounted_reward);
                *experience.n_step_after_experience.borrow_mut() = Some(Rc::clone(&experience));
                // TODO:append last n-1 experiences to memory
            }
            self.last_n_experiences.empty();
        }
    }

    pub fn sample(&self, num_experiences: usize, replacement: bool) -> Vec<&Rc<Experience>> {
        if replacement {
            self.memory.sample_with_replacement(num_experiences)
        } else {
            self.memory.sample_without_replacement(num_experiences)
        }
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }

    pub fn clear(&mut self) {
        self.memory.clear();
        self.last_n_experiences.empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_replay_buffer_new() {
        let buffer = TransitionBuffer::new(100, 5);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_replay_buffer_append_and_len() {
        let mut buffer = TransitionBuffer::new(100, 1);
        let state = Tensor::from_slice(&[1.0]);
        buffer.append(state.shallow_clone(), None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 0);
        buffer.append(state.shallow_clone(), None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 1);
        buffer.append(state.shallow_clone(), None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = TransitionBuffer::new(100, 5);
        for i in 0..10 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, false, 1.0);
        }
        let samples = buffer.sample(3, false);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_replay_buffer_terminal_state() {
        let mut buffer = TransitionBuffer::new(100, 5);
        for i in 0..5 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, i == 4, 1.0);
        }
        assert_eq!(buffer.last_n_experiences.clone().len(), 0);
    }

    #[test]
    fn test_q_value_and_next_experience_update() {
        let mut buffer = TransitionBuffer::new(100, 2);
        let state1 = Tensor::from_slice(&[0.0]);
        let state2 = Tensor::from_slice(&[1.0]);
        let state3 = Tensor::from_slice(&[2.0]);
        let state4 = Tensor::from_slice(&[3.0]);
        let state5 = Tensor::from_slice(&[4.0]);
        let state6 = Tensor::from_slice(&[5.0]);
        let state7 = Tensor::from_slice(&[6.0]);
        let state8 = Tensor::from_slice(&[7.0]);
        let state9 = Tensor::from_slice(&[8.0]);

        buffer.append(state1, None, 0.0, false, 0.9);
        buffer.append(state2, None, 2.0, false, 0.9);
        buffer.append(state3, None, 3.0, true, 0.9);
        buffer.append(state4, None, 0.0, false, 0.9);
        buffer.append(state5, None, 0.0, false, 0.9);
        buffer.append(state6, None, 0.0, false, 0.9);
        buffer.append(state7, None, 0.0, false, 0.9);
        buffer.append(state8, None, 0.0, false, 0.9);
        buffer.append(state9, None, 5.0, true, 0.9);

        for experience in buffer.sample(1000, true) {
            let n_step_discounted_reward = *experience.n_step_discounted_reward.borrow();
            let n_step_after_experience = experience.n_step_after_experience.borrow();
            let expected_q_value;
            if experience.state.double_value(&[]) == 0.0 {
                expected_q_value = 2.0 + 0.9 * 3.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if experience.state.double_value(&[]) == 1.0 {
                expected_q_value = 3.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_none());
            } else if experience.state.double_value(&[]) == 3.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if experience.state.double_value(&[]) == 4.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if experience.state.double_value(&[]) == 5.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if experience.state.double_value(&[]) == 6.0 {
                expected_q_value = 0.9 * 5.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if experience.state.double_value(&[]) == 7.0 {
                expected_q_value = 5.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_none());
            } else {
                panic!("Unexpected state")
            }
        }
    }
}
