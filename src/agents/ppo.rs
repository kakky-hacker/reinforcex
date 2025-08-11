use super::base_agent::BaseAgent;
use crate::memory::{Experience, OnPolicyBuffer};
use crate::misc::batch_states::batch_states;
use crate::misc::cumsum::cumsum_rev;
use crate::models::BasePolicy;
use futures::lock;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::de::value;
use std::sync::Arc;
use tch::{nn, no_grad, Device, Kind, Tensor};
use ulid::Ulid;

pub struct PPO {
    agent_id: Ulid,
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    buffer: OnPolicyBuffer,
    gamma: f64,
    lambda: f64,
    update_interval: usize,
    epoch: usize,
    minibatch_size: usize,
    clip_epsilon: f64,
    value_coef: f64,
    entropy_coef: f64,
    gae_std: bool,
    t: usize,
    current_episode_id: Ulid,
}

unsafe impl Send for PPO {}

impl PPO {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        buffer: OnPolicyBuffer,
        gamma: f64,
        lambda: f64,
        update_interval: usize,
        epoch: usize,
        minibatch_size: usize,
        clip_epsilon: f64,
        value_coef: f64,
        entropy_coef: f64,
        gae_std: bool,
    ) -> Self {
        assert!(minibatch_size <= update_interval);
        PPO {
            agent_id: Ulid::new(),
            model,
            optimizer,
            buffer,
            gamma,
            lambda,
            update_interval,
            epoch,
            minibatch_size,
            clip_epsilon,
            value_coef,
            entropy_coef,
            gae_std,
            t: 0,
            current_episode_id: Ulid::new(),
        }
    }

    fn _update(&mut self) {
        let experiences_per_episode: Vec<Vec<Arc<Experience>>> = self.buffer.flush();

        let total_transitions = experiences_per_episode
            .iter()
            .map(|v| v.len())
            .sum::<usize>()
            - experiences_per_episode.len();
        let n_iter = total_transitions.div_ceil(self.minibatch_size);
        let n_data_per_epoch = n_iter * self.minibatch_size;
        let n_data = n_data_per_epoch * self.epoch;

        // Create shuffled indice for minibatch.
        let mut rng = thread_rng();
        let mut batch_indice = (0..total_transitions).collect::<Vec<usize>>();
        let mut all_indice =
            Vec::with_capacity(total_transitions * n_data.div_ceil(total_transitions));
        for _ in 0..n_data.div_ceil(total_transitions) {
            batch_indice.shuffle(&mut rng);
            all_indice.extend(batch_indice.iter().cloned());
        }
        let all_indice = all_indice
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();

        // Create data.
        let skip_first = experiences_per_episode
            .iter()
            .flat_map(|v| v.iter().skip(1))
            .cloned()
            .collect::<Vec<Arc<Experience>>>();
        let skip_last = experiences_per_episode
            .iter()
            .flat_map(|v| v.iter().take(v.len().saturating_sub(1)))
            .cloned()
            .collect::<Vec<Arc<Experience>>>();
        let state = batch_states(
            &skip_last
                .iter()
                .map(|e| e.state.shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let next_state = batch_states(
            &skip_first
                .iter()
                .map(|e| e.state.shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let _action = batch_states(
            &skip_last
                .iter()
                .map(|e| e.action.as_ref().unwrap().shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let action = _action.view([total_transitions as i64, *_action.size().last().unwrap()]);
        let reward = Tensor::from_slice(&skip_first.iter().map(|e| e.reward).collect::<Vec<f64>>())
            .to_device(self.model.device());
        let discounted_reward = Tensor::from_slice(&cumsum_rev(
            &skip_first.iter().map(|e| e.reward).collect::<Vec<f64>>(),
            &skip_first
                .iter()
                .map(|e| {
                    if e.is_episode_terminal {
                        0.0 // For preventing td_error from passing through between different episodes.
                    } else {
                        self.gamma
                    }
                })
                .collect::<Vec<f64>>(),
        ))
        .to_device(self.model.device());
        let td_error_decay = &skip_first
            .iter()
            .map(|e| {
                if e.is_episode_terminal {
                    0.0 // For preventing td_error from passing through between different episodes.
                } else {
                    self.gamma * self.lambda
                }
            })
            .collect::<Vec<f64>>();

        let batch_log_prob = self
            .model
            .forward(&state)
            .0
            .log_prob(&action.detach())
            .detach();

        for i in 0..self.epoch {
            for j in 0..n_iter {
                let minibatch_indice = Tensor::from_slice(
                    &all_indice[i * n_iter + j * self.minibatch_size
                        ..(i + 1) * n_iter + (j + 1) * self.minibatch_size],
                );

                // Forward
                let (action_distrib, value) = self.model.forward(&state);
                let value = value.unwrap().flatten(0, 1);
                let next_value = self.model.forward(&next_state).1.unwrap().flatten(0, 1);

                // Compute ratio.
                let log_prob = action_distrib.log_prob(&action.detach());
                let ratio = (log_prob - &batch_log_prob)
                    .index_select(0, &minibatch_indice)
                    .exp();
                let clipped_ratio = ratio.clip(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon);

                // Compute GAE
                let td_error = &reward + next_value - &value;
                let mut gae = Tensor::from_slice(&cumsum_rev(
                    &(0..td_error.size()[0])
                        .map(|i| td_error.double_value(&[i]))
                        .collect::<Vec<f64>>(),
                    td_error_decay,
                ))
                .to_device(self.model.device())
                .detach();
                if self.gae_std {
                    gae = (&gae - (&gae).mean(Kind::Float)) / (&gae).std(true);
                }
                gae = gae.index_select(0, &minibatch_indice).detach();

                // Compute loss
                let policy_loss = -1.0
                    * (clipped_ratio * &gae)
                        .minimum(&(ratio * &gae))
                        .sum(Kind::Float);
                let value_loss = (&discounted_reward - value)
                    .index_select(0, &minibatch_indice)
                    .square()
                    .mean(tch::Kind::Float);
                let entropy_regularized = -1.0
                    * action_distrib
                        .entropy() // check shape
                        .index_select(0, &minibatch_indice)
                        .mean(tch::Kind::Float);
                let loss: Tensor = policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_regularized;

                // Backward
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            }
        }
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

        self.buffer.append(
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
        }

        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        self.buffer.append(
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
        let buffer = OnPolicyBuffer::new(None);

        let ppo = PPO::new(
            Box::new(model),
            optimizer,
            buffer,
            0.99,
            0.99,
            100,
            8,
            16,
            0.1,
            1.0,
            1.0,
            false,
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
        let buffer = OnPolicyBuffer::new(None);

        let mut ppo = PPO::new(
            Box::new(model),
            optimizer,
            buffer,
            0.5,
            0.99,
            100,
            3,
            32,
            0.1,
            1.0,
            1.0,
            false,
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
