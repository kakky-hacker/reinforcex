use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum;
use crate::misc::random_access_queue::RandomAccessQueue;
use std::sync::{Arc, Mutex};
use tch::Tensor;

#[derive(Clone)]
pub struct SharedTransitionBuffer {
    inner: Arc<Mutex<TransitionBuffer>>,
}

pub struct TransitionBuffer {
    memory: RandomAccessQueue<Arc<Experience>>,
    last_n_experiences: BoundedVecDeque<Arc<Experience>>,
}

pub struct Experience {
    pub state: Tensor,
    pub action: Option<Tensor>,
    pub reward_for_this_state: f64,
    pub n_step_discounted_reward: Mutex<Option<f64>>,
    pub n_step_after_experience: Mutex<Option<Arc<Experience>>>,
}

impl SharedTransitionBuffer {
    pub fn new(capacity: usize, n_steps: usize) -> Self {
        assert!(capacity > 0);
        assert!(n_steps > 0);
        Self {
            inner: Arc::new(Mutex::new(TransitionBuffer {
                memory: RandomAccessQueue::new(capacity),
                last_n_experiences: BoundedVecDeque::new(n_steps),
            })),
        }
    }

    pub fn append(
        &self,
        state: Tensor,
        action: Option<Tensor>,
        reward: f64,
        is_episode_terminal: bool,
        gamma: f64,
    ) {
        let mut buffer = self.inner.lock().unwrap();

        let experience = Arc::new(Experience {
            state,
            action,
            reward_for_this_state: reward,
            n_step_discounted_reward: Mutex::new(None),
            n_step_after_experience: Mutex::new(None),
        });

        if !buffer.last_n_experiences.is_empty() {
            let mut rewards: Vec<f64> = buffer
                .last_n_experiences
                .clone()
                .into_iter()
                .skip(1)
                .map(|e| e.reward_for_this_state)
                .collect();
            rewards.push(reward);
            let n_step_discounted_reward = cumsum::cumsum_rev(&rewards, gamma)[0];
            *buffer
                .last_n_experiences
                .front_mut()
                .n_step_discounted_reward
                .lock()
                .unwrap() = Some(n_step_discounted_reward);
            *buffer
                .last_n_experiences
                .front_mut()
                .n_step_after_experience
                .lock()
                .unwrap() = Some(experience.clone());
        }

        let n_step_before = buffer.last_n_experiences.push_back(experience.clone());
        if let Some(exp) = n_step_before {
            buffer.memory.append(exp);
        }

        if is_episode_terminal {
            let mut rewards: Vec<f64> = buffer
                .last_n_experiences
                .clone()
                .into_iter()
                .skip(1)
                .map(|e| e.reward_for_this_state)
                .collect();
            rewards.push(0.0);
            let q_values = cumsum::cumsum_rev(&rewards, gamma);
            for (exp, &q) in buffer
                .last_n_experiences
                .clone()
                .into_iter()
                .zip(q_values.iter())
            {
                *exp.n_step_discounted_reward.lock().unwrap() = Some(q);
                *exp.n_step_after_experience.lock().unwrap() = Some(exp.clone());
            }
            buffer.last_n_experiences.empty();
        }
    }

    pub fn sample(&self, num_experiences: usize, replacement: bool) -> Vec<Arc<Experience>> {
        let buffer = self.inner.lock().unwrap();
        if replacement {
            buffer
                .memory
                .sample_with_replacement(num_experiences)
                .into_iter()
                .cloned()
                .collect()
        } else {
            buffer
                .memory
                .sample_without_replacement(num_experiences)
                .into_iter()
                .cloned()
                .collect()
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().memory.len()
    }

    pub fn clear(&self) {
        let mut buffer = self.inner.lock().unwrap();
        buffer.memory.clear();
        buffer.last_n_experiences.empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tch::Tensor;
    use tokio::task;
    use tokio::task::LocalSet;

    #[test]
    fn test_replay_buffer_new() {
        let buffer = SharedTransitionBuffer::new(100, 5);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_replay_buffer_append_and_len() {
        let buffer = SharedTransitionBuffer::new(100, 1);
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
        let buffer = SharedTransitionBuffer::new(100, 5);
        for i in 0..10 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, false, 1.0);
        }
        let samples = buffer.sample(3, false);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_replay_buffer_terminal_state() {
        let buffer = SharedTransitionBuffer::new(100, 5);
        for i in 0..5 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, i == 4, 1.0);
        }
        let inner = buffer.inner.lock().unwrap();
        assert_eq!(inner.last_n_experiences.clone().len(), 0);
    }

    #[test]
    fn test_q_value_and_next_experience_update() {
        let buffer = SharedTransitionBuffer::new(100, 2);
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
            } else {
                panic!("Unexpected state")
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_concurrent_appends() {
        let local = LocalSet::new();

        local
            .run_until(async {
                let buffer = SharedTransitionBuffer::new(200, 3);

                let tasks: Vec<_> = (0..10)
                    .map(|i| {
                        let buffer = buffer.clone();
                        task::spawn_local(async move {
                            for j in 0..10 {
                                let state = tch::Tensor::from_slice(&[i as f64, j as f64]);
                                buffer.append(state, None, 1.0, false, 0.99);
                                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                            }
                        })
                    })
                    .collect();

                for t in tasks {
                    t.await.unwrap();
                }

                assert!(buffer.len() > 0);
                let samples = buffer.sample(5, false);
                assert_eq!(samples.len(), 5);
            })
            .await;
    }
}
