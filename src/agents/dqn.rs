use super::base_agent::BaseAgent;
use crate::explorers::BaseExplorer;
use crate::misc::batch_states::batch_states;
use crate::models::BaseQFunction;
use crate::replay_buffer::ReplayBuffer;
use tch::{nn, no_grad, Device, Tensor};

pub struct DQN {
    model: Box<dyn BaseQFunction>,
    optimizer: nn::Optimizer,
    replay_buffer: ReplayBuffer,
    explorer: Box<dyn BaseExplorer>,
    action_size: usize,
    batch_size: usize,
    update_interval: usize,
    target_model: Box<dyn BaseQFunction>,
    target_update_interval: usize,
    gamma: f64,
    n_steps: usize,
    t: usize,
}

impl DQN {
    pub fn new(
        model: Box<dyn BaseQFunction>,
        optimizer: nn::Optimizer,
        action_size: usize,
        batch_size: usize,
        update_interval: usize,
        target_update_interval: usize,
        explorer: Box<dyn BaseExplorer>,
        gamma: f64,
        n_steps: usize,
    ) -> Self {
        let target_model = model.clone();
        DQN {
            model,
            optimizer,
            replay_buffer: ReplayBuffer::new(1000000, n_steps),
            explorer,
            action_size,
            batch_size,
            update_interval,
            target_model,
            target_update_interval,
            gamma,
            n_steps,
            t: 0,
        }
    }

    fn update(&mut self) {
        let experiences = self.replay_buffer.sample(self.batch_size);
        let mut states: Vec<Tensor> = vec![];
        let mut actions: Vec<Tensor> = vec![];
        let mut n_step_discounted_rewards: Vec<f64> = vec![];
        for experience in experiences {
            let state = experience.state.shallow_clone();
            let action = experience.action.as_ref().unwrap().shallow_clone();
            let n_step_discounted_reward = experience.n_step_discounted_reward.borrow().unwrap();
            states.push(state);
            actions.push(action);
            n_step_discounted_rewards.push(n_step_discounted_reward);
        }
        let pred_q_values = self._compute_pred_q_values(states, actions);
        let loss = self._compute_loss(
            Tensor::from_slice(&n_step_discounted_rewards),
            pred_q_values,
        );
        loss.backward();
        self.optimizer.step();
    }

    fn sync_target_model(&mut self) {
        self.target_model = self.model.clone();
    }

    fn _compute_q_values(
        &self,
        states: Vec<Tensor>,
        n_step_discounted_rewards: Vec<f64>,
    ) -> Tensor {
        let _states = batch_states(states, self.model.is_cuda());
        let pred_q_values = self.target_model.forward(&_states);
        let max_q_values = pred_q_values.max_dim(1, false).0;
        let gamma_n = self.gamma.powi(self.n_steps as i32);
        let n_step_discounted_rewards_tensor = Tensor::from_slice(&n_step_discounted_rewards);
        let updated_q_values = max_q_values * gamma_n + n_step_discounted_rewards_tensor;
        updated_q_values
    }

    fn _compute_pred_q_values(&self, states: Vec<Tensor>, actions: Vec<Tensor>) -> Tensor {
        let _states = batch_states(states, self.model.is_cuda());
        let pred_q_values = self.model.forward(&_states);
        let actions = Tensor::stack(&actions, 0).to_kind(tch::Kind::Int64);
        let pred_q_values_selected = pred_q_values
            .gather(1, &actions.unsqueeze(1), false)
            .squeeze();
        pred_q_values_selected
    }

    fn _compute_loss(&self, q_values: Tensor, pred_q_values: Tensor) -> Tensor {
        let loss = (q_values - pred_q_values)
            .square()
            .mean(tch::Kind::Float)
            .sqrt();
        loss
    }
}

impl BaseAgent for DQN {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
            let q_values = self.model.forward(&state);
            q_values.argmax(1, false).to_device(Device::Cpu)
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;
        let state = batch_states(vec![obs.shallow_clone()], self.model.is_cuda());
        let q_values = self.model.forward(&state);

        let greedy_action_func = || q_values.argmax(1, false).int64_value(&[0]) as usize;
        let random_action_func = || rand::random::<usize>() % self.action_size;

        let action_idx =
            self.explorer
                .select_action(self.t, &random_action_func, &greedy_action_func);
        let action = Tensor::from_slice(&[action_idx as i64])
            .detach()
            .to_device(Device::Cpu);

        self.replay_buffer.append(
            state,
            Some(action.shallow_clone()),
            reward,
            false,
            self.gamma,
        );
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
        self.replay_buffer
            .append(state, None, reward, true, self.gamma);
    }
}
