use super::base_agent::BaseAgent;
use crate::memory::TransitionBuffer;
use crate::misc::batch_states::batch_states;
use crate::misc::cumsum;
use crate::models::BasePolicy;
use crate::prob_distributions::BaseDistribution;
use std::collections::HashSet;
use std::fs;
use tch::{nn, no_grad, Device, Tensor};

pub struct PPO {
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    gamma: f64,
    update_interval: usize,
    transition_buffer: TransitionBuffer,
    t: usize,
}

impl PPO {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        gamma: f64,
        update_interval: usize,
        n_steps: usize,
    ) -> Self {
        PPO {
            model,
            optimizer,
            gamma,
            update_interval,
            transition_buffer: TransitionBuffer::new(100000000, n_steps),
            t: 0,
        }
    }

    fn _update(&mut self) {
        let experiences = self
            .transition_buffer
            .sample(self.transition_buffer.len(), false);
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
        let (action_distrib, value) = self.model.forward(&state);
        let action = action_distrib.most_probable().to_device(Device::Cpu);

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
