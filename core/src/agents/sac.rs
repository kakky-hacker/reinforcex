use super::base_agent::BaseAgent;
use crate::memory::{Experience, ReplayBuffer};
use crate::misc::batch_states::batch_states;
use crate::models::{BasePolicy, BaseQFunction};
use std::sync::Arc;
use tch::{nn, no_grad, Device, Kind, Tensor};
use ulid::Ulid;

const LOG_PROB_EPSILON: f64 = 1e-6;

struct SACBatch {
    states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    non_terminal: Tensor,
}

pub struct SAC {
    agent_id: Ulid,
    actor: Box<dyn BasePolicy>,
    actor_optimizer: nn::Optimizer,
    critic1: Box<dyn BaseQFunction>,
    critic1_optimizer: nn::Optimizer,
    critic2: Box<dyn BaseQFunction>,
    critic2_optimizer: nn::Optimizer,
    target_critic1: Box<dyn BaseQFunction>,
    target_critic2: Box<dyn BaseQFunction>,
    transition_buffer: Arc<ReplayBuffer>,
    replay_start_size: usize,
    batch_size: usize,
    update_interval: usize,
    gamma: f64,
    tau: f64,
    alpha: f64,
    squash_action: bool,
    t: usize,
    current_episode_id: Ulid,
    latest_actor_loss: Option<f64>,
    latest_critic1_loss: Option<f64>,
    latest_critic2_loss: Option<f64>,
    n_updates: usize,
}

unsafe impl Send for SAC {}

impl SAC {
    pub fn new(
        actor: Box<dyn BasePolicy>,
        actor_optimizer: nn::Optimizer,
        critic1: Box<dyn BaseQFunction>,
        critic1_optimizer: nn::Optimizer,
        critic2: Box<dyn BaseQFunction>,
        critic2_optimizer: nn::Optimizer,
        transition_buffer: Arc<ReplayBuffer>,
        replay_start_size: usize,
        batch_size: usize,
        update_interval: usize,
        gamma: f64,
        tau: f64,
        alpha: f64,
        squash_action: bool,
    ) -> Self {
        assert!(replay_start_size > 0);
        assert!(batch_size > 0);
        assert!(update_interval > 0);
        assert!((0.0..=1.0).contains(&gamma));
        assert!((0.0..=1.0).contains(&tau));
        assert!(alpha >= 0.0);
        assert_eq!(actor.device(), critic1.device());
        assert_eq!(actor.device(), critic2.device());

        let target_critic1 = critic1.clone();
        let target_critic2 = critic2.clone();

        SAC {
            agent_id: Ulid::new(),
            actor,
            actor_optimizer,
            critic1,
            critic1_optimizer,
            critic2,
            critic2_optimizer,
            target_critic1,
            target_critic2,
            transition_buffer,
            replay_start_size,
            batch_size,
            update_interval,
            gamma,
            tau,
            alpha,
            squash_action,
            t: 0,
            current_episode_id: Ulid::new(),
            latest_actor_loss: None,
            latest_critic1_loss: None,
            latest_critic2_loss: None,
            n_updates: 0,
        }
    }

    fn _update(&mut self) {
        if self.transition_buffer.len() < self.replay_start_size.max(self.batch_size) {
            return;
        }

        let batch = self._sample_batch();

        let gamma_n = self.gamma.powi(self.transition_buffer.get_n_steps() as i32);
        let target_q = no_grad(|| {
            let (next_actions, next_log_prob) =
                self._sample_action_and_log_prob(&batch.next_states);
            let next_inputs = self._critic_input(&batch.next_states, &next_actions);
            let next_q1 = self.target_critic1.forward(&next_inputs).view([-1]);
            let next_q2 = self.target_critic2.forward(&next_inputs).view([-1]);
            let next_q = next_q1.minimum(&next_q2) - self.alpha * next_log_prob;
            &batch.rewards + gamma_n * &batch.non_terminal * next_q
        });

        let critic_inputs = self._critic_input(&batch.states, &batch.actions);
        let pred_q1 = self.critic1.forward(&critic_inputs).view([-1]);
        let critic1_loss: Tensor = (&pred_q1 - &target_q).square().mean(Kind::Float) * 0.5;
        assert!(critic1_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_critic1_loss = Some(critic1_loss.double_value(&[]));
        self.critic1_optimizer.zero_grad();
        critic1_loss.backward();
        self.critic1_optimizer.step();

        let pred_q2 = self.critic2.forward(&critic_inputs).view([-1]);
        let critic2_loss: Tensor = (&pred_q2 - &target_q).square().mean(Kind::Float) * 0.5;
        assert!(critic2_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_critic2_loss = Some(critic2_loss.double_value(&[]));
        self.critic2_optimizer.zero_grad();
        critic2_loss.backward();
        self.critic2_optimizer.step();

        let (actions, log_prob) = self._sample_action_and_log_prob(&batch.states);
        let actor_inputs = self._critic_input(&batch.states, &actions);
        let q1 = self.critic1.forward(&actor_inputs).view([-1]);
        let q2 = self.critic2.forward(&actor_inputs).view([-1]);
        let actor_loss = (self.alpha * log_prob - q1.minimum(&q2)).mean(Kind::Float);
        assert!(actor_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_actor_loss = Some(actor_loss.double_value(&[]));
        self.actor_optimizer.zero_grad();
        actor_loss.backward();
        self.actor_optimizer.step();

        self.target_critic1
            .soft_update_from(self.critic1.as_ref(), self.tau);
        self.target_critic2
            .soft_update_from(self.critic2.as_ref(), self.tau);
        self.n_updates += 1;
    }

    fn _sample_batch(&self) -> SACBatch {
        let experiences = self.transition_buffer.sample(self.batch_size, true);
        let mut states: Vec<Tensor> = Vec::with_capacity(self.batch_size);
        let mut actions: Vec<Tensor> = Vec::with_capacity(self.batch_size);
        let mut rewards: Vec<f64> = Vec::with_capacity(self.batch_size);
        let mut next_states: Vec<Tensor> = Vec::with_capacity(self.batch_size);
        let mut non_terminal: Vec<f64> = Vec::with_capacity(self.batch_size);

        for experience in experiences {
            let n_step_after_experience = Self::_n_step_after_experience(&experience);

            states.push(experience.state.shallow_clone());
            actions.push(
                experience
                    .action
                    .as_ref()
                    .expect("sampled replay experience must have an action")
                    .shallow_clone(),
            );
            rewards.push(
                experience
                    .n_step_discounted_reward
                    .lock()
                    .unwrap()
                    .unwrap_or(experience.reward),
            );
            next_states.push(n_step_after_experience.state.shallow_clone());
            non_terminal.push(if n_step_after_experience.is_episode_terminal {
                0.0
            } else {
                1.0
            });
        }

        let device = self.actor.device();
        let batch_size = self.batch_size as i64;
        let states = batch_states(&states, device).view([batch_size, -1]);
        let actions = Tensor::stack(&actions, 0)
            .to_device(device)
            .view([batch_size, -1]);
        let rewards = Tensor::from_slice(&rewards)
            .to_kind(Kind::Float)
            .to_device(device);
        let next_states = batch_states(&next_states, device).view([batch_size, -1]);
        let non_terminal = Tensor::from_slice(&non_terminal)
            .to_kind(Kind::Float)
            .to_device(device);

        SACBatch {
            states,
            actions,
            rewards,
            next_states,
            non_terminal,
        }
    }

    fn _n_step_after_experience(experience: &Arc<Experience>) -> Arc<Experience> {
        experience
            .n_step_after_experience
            .lock()
            .unwrap()
            .as_ref()
            .expect("sampled replay experience must have a next experience")
            .clone()
    }

    fn _sample_action_and_log_prob(&self, states: &Tensor) -> (Tensor, Tensor) {
        let (action_distrib, _) = self.actor.forward(states);
        let (mean, var) = action_distrib.params();
        let var = var + LOG_PROB_EPSILON;
        let std = var.sqrt();
        let noise = Tensor::randn_like(mean);
        let raw_action = mean + std * noise;

        let diff = (&raw_action - mean).pow_tensor_scalar(2.0);
        let log_prob_each_dim: Tensor =
            -0.5 * ((2.0 * std::f64::consts::PI).ln() + var.log() + diff / &var);
        let mut log_prob = log_prob_each_dim.sum_dim_intlist([-1].as_ref(), false, Kind::Float);

        if self.squash_action {
            let action = raw_action.tanh();
            let correction = (Tensor::ones_like(&action) - action.square() + LOG_PROB_EPSILON)
                .log()
                .sum_dim_intlist([-1].as_ref(), false, Kind::Float);
            log_prob = log_prob - correction;
            (action, log_prob)
        } else {
            (raw_action, log_prob)
        }
    }

    fn _critic_input(&self, states: &Tensor, actions: &Tensor) -> Tensor {
        let batch_size = states.size()[0];
        Tensor::cat(
            &[
                states.view([batch_size, -1]),
                actions.view([batch_size, -1]),
            ],
            1,
        )
    }
}

impl BaseAgent for SAC {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
            let (action_distrib, _) = self.actor.forward(&state);
            let action = action_distrib.most_probable();
            let action = if self.squash_action {
                action.tanh()
            } else {
                action
            };
            action.to_device(Device::Cpu)
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;

        let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
        let (action, _) = no_grad(|| self._sample_action_and_log_prob(&state));
        let action = action.detach().to_device(Device::Cpu);

        self.transition_buffer.append(
            self.agent_id,
            self.current_episode_id,
            state,
            Some(action.shallow_clone()),
            None,
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
        let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
        self.transition_buffer.append(
            self.agent_id,
            self.current_episode_id,
            state,
            None,
            None,
            reward,
            true,
            self.gamma,
        );
        self.current_episode_id = Ulid::new();
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        let mut statistics = vec![
            ("t".to_string(), self.t as f64),
            ("n_updates".to_string(), self.n_updates as f64),
            ("temperature".to_string(), self.alpha),
            (
                "replay_buffer_len".to_string(),
                self.transition_buffer.len() as f64,
            ),
        ];

        if let Some(loss) = self.latest_actor_loss {
            statistics.push(("actor_loss".to_string(), loss));
        }
        if let Some(loss) = self.latest_critic1_loss {
            statistics.push(("critic1_loss".to_string(), loss));
        }
        if let Some(loss) = self.latest_critic2_loss {
            statistics.push(("critic2_loss".to_string(), loss));
        }

        statistics
    }

    fn get_agent_id(&self) -> &Ulid {
        &self.agent_id
    }

    fn save(&self) {}

    fn load(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FCGaussianPolicy, FCQNetwork};
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    fn build_sac() -> SAC {
        let device = Device::Cpu;
        let obs_size = 4;
        let action_size = 2;
        let hidden_layers = 1;
        let hidden_channels = 16;

        let actor_vs = nn::VarStore::new(device);
        let actor_optimizer = nn::Adam::default().build(&actor_vs, 1e-3).unwrap();
        let actor = FCGaussianPolicy::new(
            actor_vs,
            obs_size,
            action_size,
            hidden_layers,
            hidden_channels,
            None,
            None,
            false,
            "diagonal",
            1e-3,
        );

        let critic1_vs = nn::VarStore::new(device);
        let critic1_optimizer = nn::Adam::default().build(&critic1_vs, 1e-3).unwrap();
        let critic1 = FCQNetwork::new(
            critic1_vs,
            obs_size + action_size,
            1,
            hidden_layers,
            hidden_channels,
        );

        let critic2_vs = nn::VarStore::new(device);
        let critic2_optimizer = nn::Adam::default().build(&critic2_vs, 1e-3).unwrap();
        let critic2 = FCQNetwork::new(
            critic2_vs,
            obs_size + action_size,
            1,
            hidden_layers,
            hidden_channels,
        );

        SAC::new(
            Box::new(actor),
            actor_optimizer,
            Box::new(critic1),
            critic1_optimizer,
            Box::new(critic2),
            critic2_optimizer,
            Arc::new(ReplayBuffer::new(100, 1)),
            4,
            4,
            2,
            0.99,
            0.005,
            0.2,
            true,
        )
    }

    #[test]
    fn test_soft_actor_critic_new() {
        let sac = build_sac();

        assert_eq!(sac.replay_start_size, 4);
        assert_eq!(sac.batch_size, 4);
        assert_eq!(sac.update_interval, 2);
        assert_eq!(sac.gamma, 0.99);
        assert_eq!(sac.t, 0);
        assert!(sac.squash_action);
    }

    #[test]
    fn test_soft_actor_critic_act() {
        let sac = build_sac();
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);

        let action = sac.act(&obs);

        assert_eq!(action.size(), vec![1, 2]);
        assert!(action.min().double_value(&[]) >= -1.0);
        assert!(action.max().double_value(&[]) <= 1.0);
    }

    #[test]
    fn test_soft_actor_critic_act_and_train() {
        let mut sac = build_sac();
        let mut reward = 0.0;

        for i in 0..20 {
            let obs =
                Tensor::from_slice(&[i as f32, (i as f32) * 0.1, 1.0 - (i as f32) * 0.01, 0.5])
                    .to_kind(Kind::Float);
            let action = sac.act_and_train(&obs, reward);
            reward = -action.square().sum(Kind::Float).double_value(&[]);
            assert_eq!(action.size(), vec![1, 2]);
            assert_eq!(sac.t, i + 1);
        }

        let obs = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0]).to_kind(Kind::Float);
        sac.stop_episode_and_train(&obs, reward);

        assert!(sac.transition_buffer.len() > 0);
        assert!(sac.latest_actor_loss.is_some());
        assert!(sac.latest_critic1_loss.is_some());
        assert!(sac.latest_critic2_loss.is_some());
    }
}
