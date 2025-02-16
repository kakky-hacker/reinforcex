use tch::{nn, Device, Tensor};
use super::base_agent::BaseAgent;
use crate::replay_buffer::ReplayBuffer;
use crate::explorers::BaseExplorer;
use crate::misc::batch_states::batch_states;

pub struct DQN {
    model: nn::VarStore,
    optimizer: nn::Optimizer,
    replay_buffer: ReplayBuffer,
    explorer: Box<dyn BaseExplorer>,
    batch_size: usize,
    update_interval: usize,
    last_state: Option<Tensor>,
    last_action: Option<Tensor>,
    t: usize,
}


impl DQN {
    pub fn new(
        model: nn::VarStore,
        optimizer: nn::Optimizer,
        batch_size: usize,
        update_interval: usize,
    ) -> Self {
        DQN {
            model,
            optimizer,
            replay_buffer: ReplayBuffer::new(1000000, 1),
            explorer: EpsilonGreedy::new(1.0, 0.1, 1000000),
            batch_size,
            update_interval,
            last_state: None,
            last_action: None,
            t: 0,
        }
    }

    fn update() {
        
    }
}


impl BaseAgent for DQN {
    fn act(&self, obs: &Tensor) -> Tensor {
        let mut action = None;
        no_grad(|| {
            let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
            let q_values = self.model.forward(&state);
            let action = q_values.argmax(-1, false);
        });
        action
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.replay_buffer.append(self.last_state, self.last_action, reward);
        let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
        let q_values = self.model.forward(&state);
        let action = self.explorer.select_action(self.t, q_values.argmax(-1, false));
        self.last_state = state;
        self.last_action = action;
        self.t += 1;
        if self.t % self.update_interval == 0 {
            self.update();
        }
        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64, done: bool) {

    }
}