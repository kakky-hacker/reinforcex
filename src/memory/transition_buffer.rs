use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum;
use crate::misc::random_access_queue::RandomAccessQueue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::Tensor;

#[derive(Clone)]
pub struct TransitionBuffer {
    memory: Arc<Mutex<Memory>>,
    n_steps: usize,
}

pub struct Memory {
    experiences: RandomAccessQueue<Arc<Experience>>,
    last_n_experiences: HashMap<usize, BoundedVecDeque<Arc<Experience>>>,
}

pub struct Experience {
    pub state: Tensor,
    pub action: Option<Tensor>,
    pub reward_for_this_state: f64,
    pub n_step_discounted_reward: Mutex<Option<f64>>,
    pub n_step_after_experience: Mutex<Option<Arc<Experience>>>,
}

// Tensor does not implement Sync due to raw pointer, so we promise safety manually
unsafe impl Sync for Experience {}

impl TransitionBuffer {
    pub fn new(capacity: usize, n_steps: usize) -> Self {
        assert!(capacity > 0);
        assert!(n_steps > 0);
        Self {
            memory: Arc::new(Mutex::new(Memory {
                experiences: RandomAccessQueue::new(capacity),
                last_n_experiences: HashMap::new(),
            })),
            n_steps,
        }
    }

    pub fn append(
        &self,
        episode_id: usize,
        state: Tensor,
        action: Option<Tensor>,
        reward: f64,
        is_episode_terminal: bool,
        gamma: f64,
    ) {
        let mut memory = self.memory.lock().unwrap();
        let deque = memory
            .last_n_experiences
            .entry(episode_id)
            .or_insert_with(|| BoundedVecDeque::new(self.n_steps));

        let experience = Arc::new(Experience {
            state,
            action,
            reward_for_this_state: reward,
            n_step_discounted_reward: Mutex::new(None),
            n_step_after_experience: Mutex::new(None),
        });

        if !deque.is_empty() {
            let mut rewards: Vec<f64> = deque
                .clone()
                .into_iter()
                .skip(1)
                .map(|e| e.reward_for_this_state)
                .collect();
            rewards.push(reward);
            let n_step_discounted_reward = cumsum::cumsum_rev(&rewards, gamma)[0];
            *deque.front_mut().n_step_discounted_reward.lock().unwrap() =
                Some(n_step_discounted_reward);
            *deque.front_mut().n_step_after_experience.lock().unwrap() = Some(experience.clone());
        }

        let n_step_before = deque.push_back(experience.clone());
        if let Some(exp) = n_step_before {
            memory.experiences.append(exp);
        }

        if is_episode_terminal {
            if let Some(mut deque) = memory.last_n_experiences.remove(&episode_id) {
                let mut rewards: Vec<f64> = deque
                    .clone()
                    .into_iter()
                    .skip(1)
                    .map(|e| e.reward_for_this_state)
                    .collect();
                rewards.push(0.0);
                let q_values = cumsum::cumsum_rev(&rewards, gamma);
                for (exp, &q) in deque.clone().into_iter().zip(q_values.iter()) {
                    *exp.n_step_discounted_reward.lock().unwrap() = Some(q);
                    *exp.n_step_after_experience.lock().unwrap() = None;
                    memory.experiences.append(exp);
                }
            }
        }
    }

    pub fn sample(&self, num_experiences: usize, replacement: bool) -> Vec<Arc<Experience>> {
        let memory = self.memory.lock().unwrap();
        if replacement {
            memory
                .experiences
                .sample_with_replacement(num_experiences)
                .into_iter()
                .cloned()
                .collect()
        } else {
            memory
                .experiences
                .sample_without_replacement(num_experiences)
                .into_iter()
                .cloned()
                .collect()
        }
    }

    pub fn len(&self) -> usize {
        self.memory.lock().unwrap().experiences.len()
    }

    pub fn clear(&self) {
        let mut memory = self.memory.lock().unwrap();
        memory.experiences.clear();
        memory.last_n_experiences.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::{sync::Arc, thread::sleep, time::Duration};
    use tch::Tensor;

    #[test]
    fn test_replay_buffer_new() {
        let buffer = TransitionBuffer::new(100, 5);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_replay_buffer_append_and_len() {
        let buffer = TransitionBuffer::new(100, 1);
        let state = Tensor::from_slice(&[1.0]);
        buffer.append(0, state.shallow_clone(), None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 0);
        buffer.append(0, state.shallow_clone(), None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 1);
        buffer.append(0, state.shallow_clone(), None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let buffer = TransitionBuffer::new(100, 5);
        for i in 0..10 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(0, state, None, i as f64, false, 1.0);
        }
        let samples = buffer.sample(3, false);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_replay_buffer_terminal_state() {
        let buffer = TransitionBuffer::new(100, 5);
        for i in 0..5 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(0, state, None, i as f64, i == 4, 1.0);
        }
        let memory = buffer.memory.lock().unwrap();
        assert_eq!(
            memory
                .last_n_experiences
                .get(&0)
                .map(|v| v.len())
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn test_q_value_and_next_experience_update() {
        let buffer = TransitionBuffer::new(100, 2);
        let state1 = Tensor::from_slice(&[0.0]);
        let state2 = Tensor::from_slice(&[1.0]);
        let state3 = Tensor::from_slice(&[2.0]);
        let state4 = Tensor::from_slice(&[3.0]);
        let state5 = Tensor::from_slice(&[4.0]);
        let state6 = Tensor::from_slice(&[5.0]);
        let state7 = Tensor::from_slice(&[6.0]);
        let state8 = Tensor::from_slice(&[7.0]);
        let state9 = Tensor::from_slice(&[8.0]);

        buffer.append(0, state1, None, 0.0, false, 0.9);
        buffer.append(0, state2, None, 2.0, false, 0.9);
        buffer.append(0, state3, None, 3.0, true, 0.9);
        buffer.append(1, state4, None, 0.0, false, 0.9);
        buffer.append(1, state5, None, 0.0, false, 0.9);
        buffer.append(1, state6, None, 0.0, false, 0.9);
        buffer.append(1, state7, None, 0.0, false, 0.9);
        buffer.append(1, state8, None, 0.0, false, 0.9);
        buffer.append(1, state9, None, 5.0, true, 0.9);

        for experience in buffer.sample(9, false) {
            let n_step_discounted_reward = *experience.n_step_discounted_reward.lock().unwrap();
            let n_step_after_experience = experience.n_step_after_experience.lock().unwrap();
            let expected_q_value;
            let state_val = experience.state.double_value(&[]);
            if state_val == 0.0 {
                expected_q_value = 2.0 + 0.9 * 3.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if state_val == 1.0 {
                expected_q_value = 3.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_none());
            } else if state_val == 2.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_none());
            } else if state_val == 3.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if state_val == 4.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if state_val == 5.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if state_val == 6.0 {
                expected_q_value = 0.9 * 5.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_some());
            } else if state_val == 7.0 {
                expected_q_value = 5.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_none());
            } else if state_val == 8.0 {
                expected_q_value = 0.0;
                assert!((n_step_discounted_reward.unwrap() - expected_q_value).abs() < 1e-6);
                assert!(n_step_after_experience.is_none());
            } else {
                panic!("Unexpected state")
            }
        }
    }

    #[test]
    fn test_concurrent_append_and_sample_with_threads() {
        let buffer = Arc::new(TransitionBuffer::new(200, 3));
        let n_threads = 10;

        (0..n_threads).into_par_iter().for_each(|i| {
            for j in 1..100 {
                let state = Tensor::from_slice(&[i as f64, j as f64]);
                buffer.append(i, state, None, 1.0, false, 0.99);
                sleep(Duration::from_millis(1));

                if j % 10 == 0 {
                    let samples = buffer.sample(3, true);
                    assert!(samples.len() == 3);
                }
            }
        });

        assert!(buffer.len() > 0);
        let samples = buffer.sample(5, true);
        assert_eq!(samples.len(), 5);
    }
}
