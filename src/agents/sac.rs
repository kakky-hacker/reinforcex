use super::base_agent::BaseAgent;
use crate::memory::TransitionBuffer;
use crate::misc::batch_states::batch_states;
use crate::models::{BasePolicy, BaseQFunction};
use std::sync::Arc;
use tch::{nn, no_grad, Device, Kind, Tensor};
use ulid::Ulid;

pub struct SAC {
    actor: Box<dyn BasePolicy>,
    critic1: Box<dyn BaseQFunction>,
    critic2: Box<dyn BaseQFunction>,
    target_critic1: Box<dyn BaseQFunction>,
    target_critic2: Box<dyn BaseQFunction>,
    actor_optimizer: nn::Optimizer,
    critic_optimizer: nn::Optimizer,
    transition_buffer: Arc<TransitionBuffer>,
    gamma: f64,
    tau: f64,
    alpha: f64,
    batch_size: usize,
    update_interval: usize,
    target_update_interval: usize,
    t: usize,
    current_episode_id: Ulid,
}

unsafe impl Send for SAC {}

impl SAC {
    pub fn new(
        actor: Box<dyn BasePolicy>,
        critic1: Box<dyn BaseQFunction>,
        critic2: Box<dyn BaseQFunction>,
        actor_optimizer: nn::Optimizer,
        critic_optimizer: nn::Optimizer,
        transition_buffer: Arc<TransitionBuffer>,
        gamma: f64,
        tau: f64,
        alpha: f64,
        batch_size: usize,
        update_interval: usize,
        target_update_interval: usize,
    ) -> Self {
        let target_critic1 = critic1.clone();
        let target_critic2 = critic2.clone();
        SAC {
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            actor_optimizer,
            critic_optimizer,
            transition_buffer,
            gamma,
            tau,
            alpha,
            batch_size,
            update_interval,
            target_update_interval,
            t: 0,
            current_episode_id: Ulid::new(),
        }
    }

    fn _update(&mut self) {
        if self.transition_buffer.len() < self.batch_size {
            return;
        }

        let experiences = self.transition_buffer.sample(self.batch_size, true);
        let mut states = vec![];
        let mut next_states = vec![];
        let mut actions = vec![];
        let mut rewards = vec![];

        for exp in experiences {
            states.push(exp.state.shallow_clone());
            next_states.push(
                exp.n_step_after_experience
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .state
                    .shallow_clone(),
            );
            actions.push(exp.action.as_ref().unwrap().shallow_clone());
            rewards.push(exp.reward_for_this_state);
        }

        let states_batch = batch_states(&states, self.actor.device());
        let next_states_batch = batch_states(&next_states, self.actor.device());
        let actions_batch = Tensor::stack(&actions, 0);

        let rewards_tensor = Tensor::from_slice(&rewards).to_device(states_batch.device());

        let (next_action_dist, _) = self.actor.forward(&next_states_batch);
        let next_actions = next_action_dist.sample();
        let log_probs = next_action_dist.log_prob(&next_actions);

        let q1_target = self
            .target_critic1
            .forward(&next_states_batch)
            .gather(1, &next_actions, false)
            .squeeze();
        let q2_target = self
            .target_critic2
            .forward(&next_states_batch)
            .gather(1, &next_actions, false)
            .squeeze();
        let q_target_min = q1_target.minimum(&q2_target);
        let q_target =
            rewards_tensor + self.gamma * (q_target_min - self.alpha * log_probs).detach();

        let q1 = self
            .critic1
            .forward(&states_batch)
            .gather(1, &actions_batch, false)
            .squeeze();
        let q2 = self
            .critic2
            .forward(&states_batch)
            .gather(1, &actions_batch, false)
            .squeeze();

        let critic_loss = (q1 - &q_target).square().mean(Kind::Float)
            + (q2 - &q_target).square().mean(Kind::Float);

        self.critic_optimizer.zero_grad();
        critic_loss.backward();
        self.critic_optimizer.step();

        let (action_dist, _) = self.actor.forward(&states_batch);
        let actions_sampled = action_dist.sample();
        let log_probs = action_dist.log_prob(&actions_sampled);

        let q1_pi = self
            .critic1
            .forward(&states_batch)
            .gather(1, &actions_sampled, false)
            .squeeze();
        let q2_pi = self
            .critic2
            .forward(&states_batch)
            .gather(1, &actions_sampled, false)
            .squeeze();
        let q_pi = q1_pi.minimum(&q2_pi);

        let actor_loss = (self.alpha * log_probs - q_pi).mean(Kind::Float);

        self.actor_optimizer.zero_grad();
        actor_loss.backward();
        self.actor_optimizer.step();
    }

    fn _sync_target_model(&mut self) {
        self.target_critic1 = self.critic1.clone();
        self.target_critic2 = self.critic2.clone();
    }
}

impl BaseAgent for SAC {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
            let (action_dist, _) = self.actor.forward(&state);
            action_dist.most_probable().to_device(Device::Cuda(0))
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;
        let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
        let (action_dist, _) = self.actor.forward(&state);
        let action = action_dist.sample().to_device(Device::Cuda(0));
        self.transition_buffer.append(
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
        if self.t % self.target_update_interval == 0 {
            self._sync_target_model();
        }

        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
        self.transition_buffer.append(
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

    fn save(&self, dirname: &str, ancestors: std::collections::HashSet<String>) {}

    fn load(&self, dirname: &str, ancestors: std::collections::HashSet<String>) {}
}
