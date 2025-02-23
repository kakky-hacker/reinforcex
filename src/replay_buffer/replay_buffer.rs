use std::rc::Rc;
use std::cell::RefCell;
use tch::Tensor;
use crate::misc::random_access_queue::RandomAccessQueue;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum;

pub struct ReplayBuffer {
    memory: RandomAccessQueue<Rc<Experience>>,
    last_n_experiences: BoundedVecDeque<Rc<Experience>>,
}

pub struct Experience {
    pub state: Tensor,
    pub action: Option<Tensor>,
    pub reward_for_this_state: f64,
    pub n_step_discounted_reward: RefCell<Option<f64>>,
    pub n_step_after_experience: RefCell<Option<Rc<Experience>>>,
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
        is_episode_terminal: bool,
        gamma: f64,
    ) {
        let experience = Rc::new(Experience {
            state,
            action,
            reward_for_this_state: reward,
            n_step_discounted_reward: RefCell::new(None),
            n_step_after_experience: RefCell::new(None),
        });
    
        let prev_n_experiences: Vec<Rc<Experience>> = self.last_n_experiences.clone().into_iter().collect();

        if !prev_n_experiences.is_empty() {
            let mut rewards: Vec<f64> = prev_n_experiences.iter().skip(1).map(|e| e.reward_for_this_state).collect();
            rewards.push(reward);
            let n_step_discounted_reward = cumsum::cumsum_rev(&rewards, gamma)[0];
            *prev_n_experiences[0].n_step_discounted_reward.borrow_mut() = Some(n_step_discounted_reward);
            *prev_n_experiences[0].n_step_after_experience.borrow_mut() = Some(Rc::clone(&experience));
        }

        self.last_n_experiences.push_back(Rc::clone(&experience));

        if is_episode_terminal {
            let prev_experiences: Vec<Rc<Experience>> = self.last_n_experiences.clone().into_iter().collect();
            let mut rewards: Vec<f64> = prev_experiences.iter().skip(1).map(|e| e.reward_for_this_state).collect();
            rewards.push(0.0);  // For terminal state
            let q_values = cumsum::cumsum_rev(&rewards, gamma);
            for (experience, &n_step_discounted_reward) in prev_experiences.iter().zip(q_values.iter()) {
                *experience.n_step_discounted_reward.borrow_mut() = Some(n_step_discounted_reward);
            }
            self.last_n_experiences.empty();
        } else {
            self.memory.append(experience);
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
        buffer.append(state, None, 1.0, false, 1.0);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = ReplayBuffer::new(100, 5);
        for i in 0..10 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, false, 1.0);
        }
        let samples = buffer.sample(3);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_replay_buffer_terminal_state() {
        let mut buffer = ReplayBuffer::new(100, 5);
        for i in 0..5 {
            let state = Tensor::from_slice(&[i as f64]);
            buffer.append(state, None, i as f64, i == 4, 1.0);
        }
        assert_eq!(buffer.last_n_experiences.clone().len(), 0);
    }

    #[test]
    fn test_q_value_and_next_experience_update() {
        let mut buffer = ReplayBuffer::new(100, 2);
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

        for experience in buffer.sample(200) {
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