use tch::{nn, Device, Tensor, no_grad};
use super::base_agent::BaseAgent;
use crate::models::BasePolicy;
use crate::replay_buffer::ReplayBuffer;
use crate::explorers::BaseExplorer;
use crate::misc::batch_states::batch_states;

pub struct DQN {
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    replay_buffer: ReplayBuffer,
    explorer: Box<dyn BaseExplorer>,
    batch_size: usize,
    update_interval: usize,
    target_model: Option<Box<dyn BasePolicy>>,
    target_update_interval: usize,
    t: usize,
}


impl DQN {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        batch_size: usize,
        update_interval: usize,
        target_update_interval: usize,
        explorer: Box<dyn BaseExplorer>,
    ) -> Self {
        DQN {
            model,
            optimizer,
            replay_buffer: ReplayBuffer::new(1000000, 1),
            explorer,
            batch_size,
            update_interval,
            target_model: None,
            target_update_interval,
            t: 0,
        }
    }

    fn update(&mut self) {
        let experiences = self.replay_buffer.sample(self.batch_size);
    }

    fn sync_target_model(&mut self) {

    }
}


impl BaseAgent for DQN {
    fn act(&self, obs: &Tensor) -> Tensor {
        let mut action = None;
        no_grad(|| {
            let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
            let q_values = self.model.forward(&state);
            //let _action = q_values.argmax(-1, false);
            let _action = 0;
            let action = Tensor::from_slice(&[_action as i64]);
        });
        action.unwrap()
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;
        let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
        let q_values = self.model.forward(&state);
        //let _action = self.explorer.select_action(self.t, q_values.argmax(-1, false));
        let _action = 0;
        let action = Tensor::from_slice(&[_action as i64]).detach().to_device(Device::Cpu);
        self.replay_buffer.append(state, Some(Tensor::from_slice(&[_action as i64])), reward, false);
        if self.t % self.target_update_interval == 0 {
            self.sync_target_model();
        }
        if self.t % self.update_interval == 0 {
            self.update();
        }
        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
        self.replay_buffer.append(state, None, reward, false);
    }
}