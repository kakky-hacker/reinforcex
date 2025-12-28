use super::base_agent::BaseAgent;
use crate::memory::{EpisodicReplayBuffer, Experience, OnPolicyBuffer};
use crate::misc::batch_states::batch_states;
use crate::misc::cumsum::cumsum_rev;
use crate::models::BasePolicy;
use crate::prob_distributions::BaseDistribution;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::sync::Arc;
use tch::{nn, no_grad, Device, Kind, Tensor};
use ulid::Ulid;

const POLICY_LOG_PROB_RATIO_CLAMP_RANGE: f64 = 8.0;

pub struct PPOX {
    agent_id: Ulid,
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    local_buffer: OnPolicyBuffer,
    replay_buffer: EpisodicReplayBuffer,
    gamma: f64,
    lambda: f64,
    update_interval: usize,
    epoch: usize,
    minibatch_size: usize,
    policy_clip_epsilon: f64,
    value_clip_range: f64,
    value_coef: f64,
    entropy_coef: f64,
    gae_std: bool,
    t: usize,
    current_episode_id: Ulid,
}

unsafe impl Send for PPOX {}

impl PPOX {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        gamma: f64,
        lambda: f64,
        update_interval: usize,
        epoch: usize,
        minibatch_size: usize,
        policy_clip_epsilon: f64,
        value_clip_range: f64,
        value_coef: f64,
        entropy_coef: f64,
        gae_std: bool,
    ) -> Self {
        assert!(minibatch_size <= update_interval);
        PPOX {
            agent_id: Ulid::new(),
            model,
            optimizer,
            local_buffer: OnPolicyBuffer::new(),
            replay_buffer: EpisodicReplayBuffer::new(),
            gamma,
            lambda,
            update_interval,
            epoch,
            minibatch_size,
            policy_clip_epsilon,
            value_clip_range,
            value_coef,
            entropy_coef,
            gae_std,
            t: 0,
            current_episode_id: Ulid::new(),
        }
    }

    fn _update(&mut self) {
        let experiences_per_episode: Vec<Vec<Arc<Experience>>> = self.local_buffer.flush();

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
        let _skip_first = experiences_per_episode
            .iter()
            .flat_map(|v| v.iter().skip(1))
            .cloned()
            .collect::<Vec<Arc<Experience>>>();
        let _skip_last = experiences_per_episode
            .iter()
            .flat_map(|v| v.iter().take(v.len().saturating_sub(1)))
            .cloned()
            .collect::<Vec<Arc<Experience>>>();
        let state = batch_states(
            &_skip_last
                .iter()
                .map(|e| e.state.shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let next_state = batch_states(
            &_skip_first
                .iter()
                .map(|e| e.state.shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let _action = batch_states(
            &_skip_last
                .iter()
                .map(|e| e.action.as_ref().unwrap().shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let action = _action.view([total_transitions as i64, *_action.size().last().unwrap()]);
        let reward =
            Tensor::from_slice(&_skip_first.iter().map(|e| e.reward).collect::<Vec<f64>>())
                .to_device(self.model.device());

        let (old_action_distrib, old_value) = self.model.forward(&state);
        let old_value = old_value.unwrap().flatten(0, 1).detach();
        let old_log_prob = old_action_distrib.log_prob(&action.detach()).detach();

        let (_, old_next_value) = self.model.forward(&next_state);
        let old_next_value = old_next_value.unwrap().flatten(0, 1).detach();

        let non_terminal: Tensor = 1.0
            - Tensor::from_slice(
                &_skip_first
                    .iter()
                    .map(|e| if e.is_episode_terminal { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            )
            .to_device(self.model.device());

        let old_next_value = (old_next_value * non_terminal).detach();

        // Compute GAE
        let td_error = (reward + self.gamma * &old_next_value - &old_value).to_device(Device::Cpu);
        let _gae = Tensor::from_slice(&cumsum_rev(
            &(0..td_error.size()[0])
                .map(|i| td_error.double_value(&[i]))
                .collect::<Vec<f64>>(),
            &_skip_first
                .iter()
                .map(|e| {
                    if e.is_episode_terminal {
                        0.0 // For preventing td_error from passing through between different episodes.
                    } else {
                        self.gamma * self.lambda
                    }
                })
                .collect::<Vec<f64>>(),
        ))
        .to_device(self.model.device())
        .detach();
        let gae = if self.gae_std {
            (&_gae - (&_gae).mean(Kind::Float)) / ((&_gae).std(false) + 1e-8)
        } else {
            _gae
        };

        // Compute value target
        let value_target = &gae + &old_value;

        // Compute anchor policy
        let best_episodes = self.replay_buffer.get_best_episode();
        let mut anchor_states: Option<Tensor> = None;
        let mut anchor_action_distrib: Option<Box<dyn BaseDistribution>> = None;
        if best_episodes.len() > 2 {
            let (_anchor_states, mut anchor_action_distribs): (
                Vec<Tensor>,
                Vec<Box<dyn BaseDistribution>>,
            ) = best_episodes
                .iter()
                .filter_map(|e| {
                    e.action_distrib
                        .as_ref()
                        .map(|d| (e.state.shallow_clone(), d.copy()))
                })
                .collect::<Vec<(Tensor, Box<dyn BaseDistribution>)>>()
                .into_iter()
                .unzip();
            anchor_states = Some(batch_states(&_anchor_states, self.model.device()));
            let mut _anchor_action_distrib = anchor_action_distribs.remove(0);
            _anchor_action_distrib.concat(anchor_action_distribs);
            _anchor_action_distrib.detach();
            anchor_action_distrib = Some(_anchor_action_distrib);
        }

        for i in 0..self.epoch {
            for j in 0..n_iter {
                let minibatch_indice = Tensor::from_slice(
                    &all_indice[i * n_data_per_epoch + j * self.minibatch_size
                        ..i * n_data_per_epoch + (j + 1) * self.minibatch_size],
                )
                .to_device(self.model.device());

                // Forward
                let (action_distrib, value) = self.model.forward(&state);
                let value = value
                    .unwrap()
                    .flatten(0, 1)
                    .index_select(0, &minibatch_indice);

                // Compute kl from anchor
                let mut kl_from_anchor = Tensor::from_slice(&[0.0]);
                if (&anchor_states).is_some() {
                    let (action_distrib_for_anchor, _) =
                        self.model.forward(&(&anchor_states).as_ref().unwrap());
                    kl_from_anchor = anchor_action_distrib
                        .as_ref()
                        .unwrap()
                        .kl(&action_distrib_for_anchor)
                        .sum(Kind::Float);
                }

                // Compute policy ratio.
                let log_prob = action_distrib.log_prob(&action.detach());
                let policy_ratio = (log_prob - &old_log_prob)
                    .index_select(0, &minibatch_indice)
                    .clamp(
                        -POLICY_LOG_PROB_RATIO_CLAMP_RANGE,
                        POLICY_LOG_PROB_RATIO_CLAMP_RANGE,
                    )
                    .exp();
                let clipped_policy_ratio = policy_ratio.clamp(
                    1.0 - self.policy_clip_epsilon,
                    1.0 + self.policy_clip_epsilon,
                );

                // Compute value ratio.
                let _old_value = old_value.index_select(0, &minibatch_indice).detach();
                let clipped_value = &_old_value
                    + (&value - &_old_value).clamp(-self.value_clip_range, self.value_clip_range);

                let minibatch_gae = gae.index_select(0, &minibatch_indice).detach();

                // Compute loss
                let _value_target = (&value_target).index_select(0, &minibatch_indice).detach();
                let policy_loss: Tensor = -1.0
                    * (policy_ratio * &minibatch_gae)
                        .minimum(&(clipped_policy_ratio * &minibatch_gae))
                        .mean(Kind::Float);
                let value_loss = (&_value_target - value)
                    .square()
                    .maximum(&(&_value_target - clipped_value).square())
                    .mean(Kind::Float);
                let entropy_regularized = action_distrib
                    .entropy()
                    .index_select(0, &minibatch_indice)
                    .mean(Kind::Float);

                // Check Nan
                assert!(policy_loss.isnan().any().int64_value(&[]) == 0);
                assert!(value_loss.isnan().any().int64_value(&[]) == 0);
                assert!(entropy_regularized.isnan().any().int64_value(&[]) == 0);
                assert!(kl_from_anchor.isnan().any().int64_value(&[]) == 0);

                let loss: Tensor = policy_loss + self.value_coef * value_loss
                    - self.entropy_coef * entropy_regularized
                    + kl_from_anchor * 10;

                // Backward
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            }
        }
    }
}

impl BaseAgent for PPOX {
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
        let action_distrib = no_grad(|| {
            let (action_distrib, _) = self.model.forward(&state);
            action_distrib
        });
        let action = action_distrib.sample().detach().to_device(Device::Cpu);

        let experience = self.local_buffer.append(
            self.agent_id,
            self.current_episode_id,
            state,
            Some(action.shallow_clone()),
            Some(action_distrib),
            reward,
            false,
        );

        self.replay_buffer.append(experience);

        if self.t % self.update_interval == 0 {
            self._update();
        }

        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let experience = self.local_buffer.append(
            self.agent_id,
            self.current_episode_id,
            state,
            None,
            None,
            reward,
            true,
        );
        self.replay_buffer.append(experience);
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
