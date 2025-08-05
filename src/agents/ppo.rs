use super::base_agent::BaseAgent;
use crate::memory::ReplayBuffer;
use crate::misc::batch_states::batch_states;
use crate::models::BasePolicy;
use std::sync::Arc;
use tch::{nn, no_grad, Device, Kind, Tensor};
use ulid::Ulid;

pub struct PPO {
    agent_id: Ulid,
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    transition_buffer: Arc<ReplayBuffer>,
    gamma: f64,
    update_interval: usize,
    epoch: usize,
    minibatch_size: usize,
    clip_epsilon: f64,
    entropy_coef: f64,
    t: usize,
    current_episode_id: Ulid,
}

unsafe impl Send for PPO {}

impl PPO {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        transition_buffer: Arc<ReplayBuffer>,
        gamma: f64,
        update_interval: usize,
        epoch: usize,
        minibatch_size: usize,
        clip_epsilon: f64,
        entropy_coef: f64,
    ) -> Self {
        assert!(minibatch_size <= update_interval);
        PPO {
            agent_id: Ulid::new(),
            model,
            optimizer,
            transition_buffer,
            gamma,
            update_interval,
            epoch,
            minibatch_size,
            clip_epsilon,
            entropy_coef,
            t: 0,
            current_episode_id: Ulid::new(),
        }
    }

    fn _update(&mut self) {
        let n_iter = self.transition_buffer.len().div_ceil(self.minibatch_size);
        let n_data_per_epoch = n_iter * self.minibatch_size;
        let n_data = n_data_per_epoch * self.epoch;
        let experiences = (0..n_data.div_ceil(self.transition_buffer.len()))
            .map(|_| {
                self.transition_buffer
                    .sample(self.transition_buffer.len(), false)
            })
            .collect::<Vec<_>>()
            .concat();
        for i in 0..self.epoch {
            for j in 0..n_iter {
                let mut states: Vec<Tensor> = vec![];
                let mut n_step_after_states: Vec<Tensor> = vec![];
                let mut actions: Vec<Tensor> = vec![];
                let mut n_step_discounted_rewards: Vec<f64> = vec![];
                for experience in &experiences[i * n_iter + j * self.minibatch_size
                    ..(i + 1) * n_iter + (j + 1) * self.minibatch_size]
                {
                    let state = experience.state.shallow_clone();
                    let n_step_after_state = experience
                        .n_step_after_experience
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .state
                        .shallow_clone();
                    let action = experience.action.as_ref().unwrap().shallow_clone();
                    let n_step_discounted_reward = experience
                        .n_step_discounted_reward
                        .lock()
                        .unwrap()
                        .unwrap_or(experience.reward);
                    states.push(state);
                    n_step_after_states.push(n_step_after_state);
                    actions.push(action);
                    n_step_discounted_rewards.push(n_step_discounted_reward);
                }
                let _batch_states = batch_states(&states, self.model.device());
                let _batch_n_step_after_states =
                    batch_states(&n_step_after_states, self.model.device());
                let _batch_actions = batch_states(&actions, self.model.device());
                let _batch_probs = self
                    .model
                    .forward(&_batch_states)
                    .0
                    .prob(&_batch_actions)
                    .detach();
                let (action_distribs, values) = self.model.forward(&_batch_states);
                let td_errors = self._compute_td_error(
                    values.unwrap(),
                    &n_step_discounted_rewards,
                    &_batch_n_step_after_states,
                );
                let probs = action_distribs.prob(&_batch_actions);
                let ratio = probs / &_batch_probs;
                let clipped_ratio = ratio.clip(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon);
                // TODO: shouldn't detach td_errors?
                let policy_loss: Tensor = -1.0
                    * (clipped_ratio * td_errors.detach()).minimum(&(ratio * td_errors.detach()));
                let entropy = action_distribs.entropy();
                let loss = policy_loss.sum(Kind::Double)
                    + td_errors.square().mean(tch::Kind::Float)
                    - self.entropy_coef * entropy.mean(tch::Kind::Float);
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            }
        }
    }

    fn _compute_td_error(
        &self,
        values: Tensor,
        n_step_discounted_rewards: &Vec<f64>,
        n_step_after_states: &Tensor,
    ) -> Tensor {
        let (_, pred_values) = self.model.forward(&n_step_after_states);
        let n_step_discounted_rewards_tensor = Tensor::from_slice(&n_step_discounted_rewards);
        let gamma_n = self.gamma.powi(self.transition_buffer.get_n_steps() as i32);
        let td_error = n_step_discounted_rewards_tensor + pred_values.unwrap() * gamma_n - values;
        td_error
    }
}

impl BaseAgent for PPO {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
            let (action_distrib, _) = self.model.forward(&state);
            let action = action_distrib.most_probable().to_device(Device::Cpu);
            action
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;

        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let (action_distrib, _) = self.model.forward(&state);
        let action = action_distrib.sample().to_device(Device::Cpu);

        self.transition_buffer.append(
            self.agent_id,
            self.current_episode_id,
            state,
            Some(action.shallow_clone()),
            reward,
            false,
            self.gamma,
        );

        if self.t % self.update_interval == 0 {
            self._update();
            self.transition_buffer.clear(); // Clear buffer, because PPO is on-policy algotithm.
        }

        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        self.transition_buffer.append(
            self.agent_id,
            self.current_episode_id,
            state,
            None,
            reward,
            true,
            self.gamma,
        );
        self.current_episode_id = Ulid::new();
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        vec![]
    }

    fn get_agent_id(&self) -> &Ulid {
        &self.agent_id
    }

    fn save(&self, dirname: &str, ancestors: std::collections::HashSet<String>) {}

    fn load(&self, dirname: &str, ancestors: std::collections::HashSet<String>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FCSoftmaxPolicyWithValue;
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    #[test]
    fn test_ppo_new() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 2, 2, 64, 0.0);
        let transition_buffer = Arc::new(ReplayBuffer::new(1000, 3));

        let ppo = PPO::new(
            Box::new(model),
            optimizer,
            transition_buffer,
            0.99,
            100,
            8,
            16,
            0.1,
            1.0,
        );

        assert_eq!(ppo.update_interval, 100);
        assert_eq!(ppo.epoch, 8);
        assert_eq!(ppo.gamma, 0.99);
        assert_eq!(ppo.t, 0);
    }

    #[test]
    fn test_ppo_act_and_train() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 4, 2, 64, 0.0);
        let transition_buffer = Arc::new(ReplayBuffer::new(1000, 1));

        let mut ppo = PPO::new(
            Box::new(model),
            optimizer,
            transition_buffer,
            0.5,
            100,
            3,
            32,
            0.1,
            1.0,
        );

        let mut reward = 0.0;
        for i in 0..2000 {
            let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
            let action = ppo.act_and_train(&obs, reward);
            let action_value = i64::from(action.int64_value(&[]));
            if action_value == 2 {
                reward = 100.0;
            } else {
                reward = 0.0
            }
            assert!([0, 1, 2, 3].contains(&action_value));
            assert_eq!(ppo.t, i + 1);
        }
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
        ppo.stop_episode_and_train(&obs, 1.0);

        for _ in 0..1000 {
            let action = ppo.act(&obs);
            let action_value = i64::from(action.int64_value(&[]));
            assert_eq!(action_value, 2);
        }
    }
}
