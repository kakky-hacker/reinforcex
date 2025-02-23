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
    pub reward: f64,
    pub q_value: RefCell<Option<f64>>,
    pub next_experience: RefCell<Option<Rc<Experience>>>,
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
        gamma: f64,
    ) {
        let prev_n_experiences: Vec<Rc<Experience>> = self.last_n_experiences.clone().into_iter().collect();

        let experience = Rc::new(Experience {
            state,
            action,
            reward,
            q_value: RefCell::new(None),
            next_experience: RefCell::new(None),
            prev_n_experiences: prev_n_experiences.clone(),
        });
    
        if !prev_n_experiences.is_empty() {
            let mut rewards: Vec<f64> = prev_n_experiences.iter().map(|e| e.reward).collect();
            rewards.push(reward);
            let q_value = cumsum::cumsum_rev(&rewards, gamma)[0];
            *prev_n_experiences[0].q_value.borrow_mut() = Some(q_value);
            *prev_n_experiences[prev_n_experiences.len() - 1].next_experience.borrow_mut() = Some(Rc::clone(&experience));
        }

        self.last_n_experiences.push_back(Rc::clone(&experience));
        self.memory.append(experience);

        if is_state_terminal {
            let prev_experiences: Vec<Rc<Experience>> = self.last_n_experiences.clone().into_iter().collect();
            let rewards: Vec<f64> = prev_experiences.iter().map(|e| e.reward).collect();
            let q_values = cumsum::cumsum_rev(&rewards, gamma);
            for (experience, &q_value) in prev_experiences.iter().zip(q_values.iter()) {
                *experience.q_value.borrow_mut() = Some(q_value);
            }
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

        buffer.append(state1, None, 1.0, false, 0.9);
        buffer.append(state2, None, 2.0, false, 0.9);
        buffer.append(state3, None, 3.0, true, 0.9);
        buffer.append(state4, None, 0.0, false, 0.9);
        buffer.append(state5, None, 0.0, false, 0.9);
        buffer.append(state6, None, 0.0, false, 0.9);
        buffer.append(state7, None, 0.0, false, 0.9);
        buffer.append(state8, None, 0.0, false, 0.9);
        buffer.append(state9, None, 5.0, true, 0.9);

        for experience in buffer.sample(200) {
            let q_value = *experience.q_value.borrow();
            let next_experience = experience.next_experience.borrow();
            let expected_q_value;
            if experience.state.double_value(&[]) == 0.0 {
                expected_q_value = 1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 1.0 {
                expected_q_value = 2.0 + 0.9 * 3.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 2.0 {
                expected_q_value = 3.0;
                assert!(next_experience.is_none());
            } else if experience.state.double_value(&[]) == 3.0 {
                expected_q_value = 0.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 4.0 {
                expected_q_value = 0.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 5.0 {
                expected_q_value = 0.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 6.0 {
                expected_q_value = 0.9 * 0.9 * 5.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 7.0 {
                expected_q_value = 0.9 * 5.0;
                assert!(next_experience.is_some());
            } else if experience.state.double_value(&[]) == 8.0 {
                expected_q_value = 5.0;
                assert!(next_experience.is_none());
            } else {
                expected_q_value = -1.0;
            }
            assert!((q_value.unwrap() - expected_q_value).abs() < 1e-6);
        }
    }
}