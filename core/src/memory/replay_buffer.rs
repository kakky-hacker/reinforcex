use super::experience::Experience;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum;
use crate::misc::random_access_queue::RandomAccessQueue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ulid::Ulid;

#[derive(Clone)]
pub struct ReplayBuffer {
    experiences: Arc<Mutex<RandomAccessQueue<Arc<Experience>>>>,
    last_n_experiences_by_episode: Arc<Mutex<HashMap<Ulid, BoundedVecDeque<Arc<Experience>>>>>,
    n_steps: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, n_steps: usize) -> Self {
        assert!(capacity > 0);
        assert!(n_steps > 0);
        Self {
            experiences: Arc::new(Mutex::new(RandomAccessQueue::new(capacity))),
            last_n_experiences_by_episode: Arc::new(Mutex::new(HashMap::new())),
            n_steps,
        }
    }

    pub fn append(&self, experience: Arc<Experience>, gamma: f64) {
        let mut last_n_experiences_by_episode = self.last_n_experiences_by_episode.lock().unwrap();

        let last_n_experiences = last_n_experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(self.n_steps));

        if let Some(exp) = last_n_experiences.push_back(experience.clone()) {
            *exp.n_step_discounted_reward.lock().unwrap() = Some(
                cumsum::cumsum_rev(
                    last_n_experiences
                        .clone_deque()
                        .into_iter()
                        .map(|e| e.reward)
                        .collect::<Vec<f64>>()
                        .as_ref(),
                    &vec![gamma; last_n_experiences.len()],
                )[0],
            );
            *exp.n_step_horizon.lock().unwrap() = Some(last_n_experiences.len());
            *exp.n_step_after_experience.lock().unwrap() = Some(experience.clone());
            self.experiences.lock().unwrap().append(exp);
        }

        if experience.is_episode_end {
            if let Some(last_n_experiences) =
                last_n_experiences_by_episode.remove(&experience.episode_id)
            {
                let mut rewards = last_n_experiences
                    .clone_deque()
                    .into_iter()
                    .skip(1)
                    .map(|e| e.reward)
                    .collect::<Vec<f64>>();
                rewards.push(0.0);
                let remaining_steps = last_n_experiences.len().saturating_sub(1);
                for (index, (exp, &q)) in last_n_experiences
                    .clone_deque()
                    .into_iter()
                    .zip(cumsum::cumsum_rev(&rewards, &vec![gamma; rewards.len()]).iter())
                    .enumerate()
                {
                    if exp.is_episode_end {
                        continue;
                    }
                    *exp.n_step_discounted_reward.lock().unwrap() = Some(q);
                    *exp.n_step_horizon.lock().unwrap() = Some(remaining_steps - index);
                    *exp.n_step_after_experience.lock().unwrap() = Some(experience.clone());
                    self.experiences.lock().unwrap().append(exp);
                }
            }
        }
    }

    pub fn sample(&self, num_experiences: usize, replacement: bool) -> Vec<Arc<Experience>> {
        if replacement {
            self.experiences
                .lock()
                .unwrap()
                .sample_with_replacement(num_experiences)
                .into_iter()
                .cloned()
                .collect()
        } else {
            self.experiences
                .lock()
                .unwrap()
                .sample_without_replacement(num_experiences)
                .into_iter()
                .cloned()
                .collect()
        }
    }

    pub fn len(&self) -> usize {
        self.experiences.lock().unwrap().len()
    }

    pub fn clear(&self) {
        // Hold both locks in append's order so a clear is one atomic boundary.
        let mut pending = self.last_n_experiences_by_episode.lock().unwrap();
        self.experiences.lock().unwrap().clear();
        pending.clear();
    }

    /// Release unfinished n-step tails when an agent is dropped or abandoned.
    pub fn discard_episode(&self, episode_id: &Ulid) {
        self.last_n_experiences_by_episode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(episode_id);
    }

    pub fn get_n_steps(&self) -> usize {
        self.n_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::{sync::Arc, thread::sleep, time::Duration};
    use tch::Tensor;

    fn experience(
        episode_id: Ulid,
        state: Tensor,
        reward: f64,
        is_episode_terminal: bool,
    ) -> Arc<Experience> {
        Arc::new(Experience::new(
            episode_id,
            episode_id,
            state,
            None,
            None,
            reward,
            is_episode_terminal,
        ))
    }

    #[test]
    fn test_replay_buffer_new() {
        let buffer = ReplayBuffer::new(100, 5);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_replay_buffer_append_and_len() {
        let buffer = ReplayBuffer::new(100, 1);
        let state = Tensor::from_slice(&[1.0]);
        let episode_id = Ulid::new();
        buffer.append(
            experience(episode_id, state.shallow_clone(), 1.0, false),
            1.0,
        );
        assert_eq!(buffer.len(), 0);
        buffer.append(
            experience(episode_id, state.shallow_clone(), 1.0, false),
            1.0,
        );
        assert_eq!(buffer.len(), 1);
        buffer.append(
            experience(episode_id, state.shallow_clone(), 1.0, false),
            1.0,
        );
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_is_episode_terminal() {
        let buffer = ReplayBuffer::new(100, 1);
        let state = Tensor::from_slice(&[1.0]);
        let episode_id = Ulid::new();
        buffer.append(
            experience(episode_id, state.shallow_clone(), 1.0, true),
            1.0,
        );
    }

    #[test]
    fn test_replay_buffer_sample() {
        let buffer = ReplayBuffer::new(100, 5);
        let episode_id = Ulid::new();
        for i in 0..10 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(experience(episode_id, state, i as f64, false), 1.0);
        }
        let samples = buffer.sample(3, false);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_replay_buffer_terminal_state() {
        let buffer = ReplayBuffer::new(100, 5);
        let episode_id = Ulid::new();
        for i in 0..5 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(experience(episode_id, state, i as f64, i == 4), 1.0);
        }
        let last_n_experiences_by_episode = buffer.last_n_experiences_by_episode.lock().unwrap();
        assert_eq!(
            last_n_experiences_by_episode
                .get(&episode_id)
                .map(|v| v.len())
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn test_q_value_and_next_experience_update() {
        let buffer = ReplayBuffer::new(100, 2);
        let state1 = Tensor::from_slice(&[0.0]);
        let state2 = Tensor::from_slice(&[1.0]);
        let state3 = Tensor::from_slice(&[2.0]);
        let state4 = Tensor::from_slice(&[3.0]);
        let state5 = Tensor::from_slice(&[4.0]);
        let state6 = Tensor::from_slice(&[5.0]);
        let state7 = Tensor::from_slice(&[6.0]);
        let state8 = Tensor::from_slice(&[7.0]);
        let state9 = Tensor::from_slice(&[8.0]);

        let episode1_id = Ulid::new();
        let episode2_id = Ulid::new();

        buffer.append(experience(episode1_id, state1, 0.0, false), 0.9);
        buffer.append(experience(episode1_id, state2, 2.0, false), 0.9);
        buffer.append(experience(episode1_id, state3, 3.0, true), 0.9);
        buffer.append(experience(episode2_id, state4, 0.0, false), 0.9);
        buffer.append(experience(episode2_id, state5, 0.0, false), 0.9);
        buffer.append(experience(episode2_id, state6, 0.0, false), 0.9);
        buffer.append(experience(episode2_id, state7, 0.0, false), 0.9);
        buffer.append(experience(episode2_id, state8, 0.0, false), 0.9);
        buffer.append(experience(episode2_id, state9, 5.0, true), 0.9);

        for experience in buffer.sample(7, false) {
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
                assert!(n_step_after_experience.is_some());
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
                assert!(n_step_after_experience.is_some());
            } else {
                panic!("Unexpected state")
            }
        }
    }

    #[test]
    fn test_concurrent_append_and_sample_with_threads() {
        let buffer = Arc::new(ReplayBuffer::new(200, 3));
        let n_threads = 10;

        (0..n_threads).into_par_iter().for_each(|i| {
            let episode_id = Ulid::new();
            for j in 1..100 {
                let state = Tensor::from_slice(&[i as f64, j as f64]);
                buffer.append(experience(episode_id, state, 1.0, false), 0.99);
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

    #[test]
    fn test_truncation_flushes_short_horizons_without_marking_terminal() {
        let buffer = ReplayBuffer::new(100, 5);
        let episode_id = Ulid::new();
        for (step, reward) in [0.0, 2.0, 3.0].into_iter().enumerate() {
            let exp = Experience::new(
                episode_id,
                episode_id,
                Tensor::from_slice(&[step as f32]),
                if step < 2 {
                    Some(Tensor::from_slice(&[0i64]))
                } else {
                    None
                },
                None,
                reward,
                false,
            )
            .with_episode_end(step == 2);
            buffer.append(Arc::new(exp), 0.9);
        }
        assert_eq!(buffer.len(), 2);
        assert!(buffer
            .last_n_experiences_by_episode
            .lock()
            .unwrap()
            .is_empty());
        for exp in buffer.sample(2, false) {
            let step = exp.state.int64_value(&[0]) as usize;
            assert_eq!(*exp.n_step_horizon.lock().unwrap(), Some(2 - step));
            let expected = if step == 0 { 2.0 + 0.9 * 3.0 } else { 3.0 };
            assert!(
                (exp.n_step_discounted_reward.lock().unwrap().unwrap() - expected).abs() < 1e-9
            );
            let next = exp.n_step_after_experience.lock().unwrap();
            let next = next.as_ref().unwrap();
            assert!(next.is_episode_end);
            assert!(!next.is_episode_terminal);
            assert!(exp.action.is_some());
        }
    }

    #[test]
    fn test_discard_episode_releases_pending_tail() {
        let buffer = ReplayBuffer::new(100, 3);
        let episode_id = Ulid::new();
        let exp = experience(episode_id, Tensor::from_slice(&[1.0]), 0.0, false);
        let weak = Arc::downgrade(&exp);
        buffer.append(exp, 0.9);
        assert!(weak.upgrade().is_some());
        buffer.discard_episode(&episode_id);
        assert!(weak.upgrade().is_none());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_parallel_n_step_returns_never_mix_agents_or_episodes() {
        let buffer = Arc::new(ReplayBuffer::new(4000, 3));
        let gamma = 0.5;
        (0..6).into_par_iter().for_each(|agent_index| {
            let agent_id = Ulid::new();
            for episode_index in 0..40 {
                let episode_id = Ulid::new();
                let marker = (agent_index * 100 + episode_index + 1) as f64;
                for step in 0..=5 {
                    let exp = Experience::new(
                        agent_id,
                        episode_id,
                        Tensor::from_slice(&[marker, step as f64]),
                        if step < 5 {
                            Some(Tensor::from_slice(&[0i64]))
                        } else {
                            None
                        },
                        None,
                        if step == 0 { 0.0 } else { marker },
                        step == 5 && episode_index % 2 == 0,
                    )
                    .with_episode_end(step == 5);
                    buffer.append(Arc::new(exp), gamma);
                }
            }
        });
        assert_eq!(buffer.len(), 6 * 40 * 5);
        assert!(buffer
            .last_n_experiences_by_episode
            .lock()
            .unwrap()
            .is_empty());
        for exp in buffer.sample(buffer.len(), false) {
            let marker = exp.state.double_value(&[0]);
            let step = exp.state.double_value(&[1]) as usize;
            let horizon = 3.min(5 - step);
            let expected = marker * (0..horizon).map(|i| gamma.powi(i as i32)).sum::<f64>();
            assert_eq!(*exp.n_step_horizon.lock().unwrap(), Some(horizon));
            assert!(
                (exp.n_step_discounted_reward.lock().unwrap().unwrap() - expected).abs() < 1e-9
            );
            let next = exp.n_step_after_experience.lock().unwrap();
            let next = next.as_ref().unwrap();
            assert_eq!(next.agent_id, exp.agent_id);
            assert_eq!(next.episode_id, exp.episode_id);
            assert_eq!(next.state.double_value(&[0]), marker);
            assert_eq!(next.state.double_value(&[1]) as usize, step + horizon);
        }
    }
}
