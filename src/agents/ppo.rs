use super::base_agent::BaseAgent;
use crate::memory::TransitionBuffer;
use crate::misc::batch_states::batch_states;
use crate::models::BasePolicy;
use crate::prob_distributions::BaseDistribution;
use tch::{nn, no_grad, Device, Tensor};

pub struct PPO {
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    gamma: f64,
    update_interval: usize,
    transition_buffer: TransitionBuffer,
    epoch: usize,
    t: usize,
}

impl PPO {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        gamma: f64,
        update_interval: usize,
        n_steps: usize,
        epoch: usize,
    ) -> Self {
        PPO {
            model,
            optimizer,
            gamma,
            update_interval,
            transition_buffer: TransitionBuffer::new(100000000, n_steps),
            epoch,
            t: 0,
        }
    }

    fn _update(&mut self) {
        let experiences = self
            .transition_buffer
            .sample(self.transition_buffer.len(), false);
        let mut states: Vec<Tensor> = vec![];
        let mut n_step_after_states: Vec<Tensor> = vec![];
        let mut n_step_after_actions: Vec<Tensor> = vec![];
        let mut actions: Vec<Tensor> = vec![];
        let mut n_step_discounted_rewards: Vec<f64> = vec![];
        for experience in experiences {
            let state = experience.state.shallow_clone();
            let n_step_after_experience = experience
                .n_step_after_experience
                .borrow()
                .as_ref()
                .unwrap();
            let n_step_after_state = n_step_after_experience.state.shallow_clone();
            let action = experience.action.as_ref().unwrap().shallow_clone();
            let n_step_discounted_reward = experience
                .n_step_discounted_reward
                .borrow()
                .unwrap_or(experience.reward_for_this_state);
            states.push(state);
            n_step_after_states.push(n_step_after_state);
            actions.push(action);
            n_step_discounted_rewards.push(n_step_discounted_reward);
        }
        for _ in range(self.epoch) {
            let (action_distribs, values) = self.model.forward(&states);
            let td_errors = self._compute_td_error(values, n_step_discounted_rewards);
            let loss = self._compute_policy_loss(action_distribs, actions, td_errors) + td_errors.square();
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
        }
    }

    fn _compute_td_error(&self, values: Tensor, n_step_discounted_rewards: Vec<f64>, n_step_after_states: &Vec<Tensor>) -> Tensor {
        assert_eq!(n_step_after_states.len(), n_step_discounted_rewards.len());
        let _states = batch_states(n_step_after_states, self.model.is_cuda());
        let (_, pred_values) = self.model.forward(&_states);
        let n_step_discounted_rewards_tensor = Tensor::from_slice(&n_step_discounted_rewards);
        let gamma_n = self.gamma.powi(self.n_steps as i32);
        let td_error = n_step_discounted_rewards_tensor + pred_values * gamma_n - values;
        td_error
    }

    fn _compute_policy_loss(&self, action_distribs, actions, td_errors) -> Tensor {
        let log_probs = action_distribs.log_prob(actions);
        let policy_loss = -1.0 * log_probs * td_errors.detach();
        policy_loss
    }
}

impl BaseAgent for PPO {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.model.is_cuda());
            let (action_distrib, _) = self.model.forward(&state);
            let action = action_distrib.most_probable().to_device(Device::Cpu);
            action
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;

        let state = batch_states(&vec![obs.shallow_clone()], self.model.is_cuda());
        let (action_distrib, _) = self.model.forward(&state);
        let action = action_distrib.sample().to_device(Device::Cpu);

        self.transition_buffer.append(
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
        let state = batch_states(&vec![obs.shallow_clone()], self.model.is_cuda());
        self.transition_buffer
            .append(state, None, reward, true, self.gamma);
    }
}
