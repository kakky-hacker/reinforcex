use super::base_agent::BaseAgent;
use crate::memory::TransitionBuffer;
use crate::misc::batch_states::batch_states;
use crate::models::BasePolicy;
use crate::prob_distributions::BaseDistribution;
use tch::{nn, no_grad, Device, Kind, Tensor};

pub struct PPO {
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    gamma: f64,
    update_interval: usize,
    transition_buffer: TransitionBuffer,
    epoch: usize,
    n_steps: usize,
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
            n_steps,
            t: 0,
        }
    }

    fn _update(&mut self) {
        let experiences = self
            .transition_buffer
            .sample(self.transition_buffer.len(), false);
        let mut states: Vec<Tensor> = vec![];
        let mut n_step_after_states: Vec<Tensor> = vec![];
        let mut actions: Vec<Tensor> = vec![];
        let mut n_step_discounted_rewards: Vec<f64> = vec![];
        for experience in experiences {
            let state = experience.state.shallow_clone();
            let n_step_after_state = experience
                .n_step_after_experience
                .borrow()
                .as_ref()
                .unwrap()
                .state
                .shallow_clone();
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
        let _batch_states = batch_states(&states, self.model.is_cuda());
        let _batch_n_step_after_states = batch_states(&n_step_after_states, self.model.is_cuda());
        let _batch_actions = batch_states(&actions, self.model.is_cuda());
        for _ in 0..self.epoch {
            let (action_distribs, values) = self.model.forward(&_batch_states);
            let td_errors = self._compute_td_error(
                values.unwrap(),
                &n_step_discounted_rewards,
                &_batch_n_step_after_states,
            );
            let loss = self
                ._compute_policy_loss(action_distribs, &_batch_actions, &td_errors)
                .sum(Kind::Double)
                + td_errors.square().mean(tch::Kind::Float).sum(Kind::Double);
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
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
        let gamma_n = self.gamma.powi(self.n_steps as i32);
        let td_error = n_step_discounted_rewards_tensor + pred_values.unwrap() * gamma_n - values;
        td_error
    }

    fn _compute_policy_loss(
        &self,
        action_distribs: Box<dyn BaseDistribution>,
        actions: &Tensor,
        td_errors: &Tensor,
    ) -> Tensor {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FCSoftmaxPolicyWithValue;
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    #[test]
    fn test_ppo_new() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(&vs, 4, 2, 2, Some(64), 0.0);

        let ppo = PPO::new(Box::new(model), optimizer, 0.99, 100, 3, 8);

        assert_eq!(ppo.update_interval, 100);
        assert_eq!(ppo.epoch, 8);
        assert_eq!(ppo.gamma, 0.99);
        assert_eq!(ppo.n_steps, 3);
        assert_eq!(ppo.t, 0);
    }

    #[test]
    fn test_ppo_act_and_train() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(&vs, 4, 4, 2, Some(64), 0.0);

        let mut ppo = PPO::new(Box::new(model), optimizer, 0.5, 50, 1, 4);

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
            if ppo.t > 1000 {
                assert_eq!(action_value, 2);
            }
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
