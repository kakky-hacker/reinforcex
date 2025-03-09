use super::base_agent::BaseAgent;
use crate::explorers::BaseExplorer;
use crate::memory::ReplayBuffer;
use crate::misc::batch_states::batch_states;
use crate::models::BaseQFunction;
use candle_core::{Device, Result, Tensor};
use candle_nn::optim::Optimizer;

pub struct DQN<T: Optimizer> {
    model: Box<dyn BaseQFunction>,
    optimizer: T,
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

impl<T: Optimizer> DQN<T> {
    pub fn new(
        model: Box<dyn BaseQFunction>,
        optimizer: T,
        action_size: usize,
        batch_size: usize,
        replay_buffer_capacity: usize,
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
            replay_buffer: ReplayBuffer::new(replay_buffer_capacity, n_steps),
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

    fn _update(&mut self) -> Result<()> {
        if self.replay_buffer.len() < self.batch_size {
            return Ok(());
        }
        let experiences = self.replay_buffer.sample(self.batch_size, true);
        let mut states: Vec<Tensor> = vec![];
        let mut n_step_after_states: Vec<Tensor> = vec![];
        let mut actions: Vec<Tensor> = vec![];
        let mut n_step_discounted_rewards: Vec<f64> = vec![];
        for experience in experiences {
            let state = experience.state.clone();
            let n_step_after_state = experience
                .n_step_after_experience
                .borrow()
                .as_ref()
                .unwrap()
                .state
                .clone();
            let action = experience.action.as_ref().unwrap().clone();
            let n_step_discounted_reward = experience
                .n_step_discounted_reward
                .borrow()
                .unwrap_or(experience.reward_for_this_state);
            states.push(state);
            n_step_after_states.push(n_step_after_state);
            actions.push(action);
            n_step_discounted_rewards.push(n_step_discounted_reward);
        }
        let q_values = self._compute_q_values(&n_step_after_states, n_step_discounted_rewards)?;
        let pred_q_values = self._compute_pred_q_values(&states, actions)?;
        let loss = self._compute_loss(q_values, pred_q_values)?;
        self.optimizer.backward_step(&loss)?;
        return Ok(());
    }

    fn _sync_target_model(&mut self) {
        self.target_model = self.model.clone();
    }

    fn _compute_q_values(
        &self,
        n_step_after_states: &Vec<Tensor>,
        n_step_discounted_rewards: Vec<f64>,
    ) -> Result<Tensor> {
        assert_eq!(n_step_after_states.len(), n_step_discounted_rewards.len());
        let _states = batch_states(n_step_after_states, self.model.get_device())?;
        let pred_q_values = self
            .target_model
            .forward(&_states)?
            .to_device(&Device::Cpu)?;
        let max_q_values = pred_q_values.max(1)?;
        let gamma_n = self.gamma.powi(self.n_steps as i32);
        let n_step_discounted_rewards_tensor = Tensor::from_slice(
            &n_step_discounted_rewards,
            &[n_step_discounted_rewards.len()],
            &Device::Cpu,
        )?;
        let q_values = (max_q_values * gamma_n as f64 + n_step_discounted_rewards_tensor)?;
        Ok(q_values)
    }

    fn _compute_pred_q_values(&self, states: &Vec<Tensor>, actions: Vec<Tensor>) -> Result<Tensor> {
        assert_eq!(states.len(), actions.len());
        let _states = batch_states(states, self.model.get_device())?;
        let pred_q_values = self.model.forward(&_states)?.to_device(&Device::Cpu)?;
        let actions = Tensor::stack(&actions, 0)?.to_dtype(candle_core::DType::U32)?;
        let pred_q_values_selected = pred_q_values.gather(&actions, 1)?.squeeze(1)?;
        Ok(pred_q_values_selected)
    }

    fn _compute_loss(&self, q_values: Tensor, pred_q_values: Tensor) -> Result<Tensor> {
        let loss = (q_values - pred_q_values)?.sqr()?.mean_all()?;
        Ok(loss)
    }
}

impl<T: Optimizer> BaseAgent for DQN<T> {
    fn act(&self, obs: &Tensor) -> Result<Tensor> {
        let state = batch_states(&vec![obs.clone()], self.model.get_device())?.detach();
        let q_values = self.model.forward(&state)?.detach();
        let action = q_values.argmax(1)?.to_device(&Device::Cpu)?;
        Ok(action)
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Result<Tensor> {
        self.t += 1;
        let state = batch_states(&vec![obs.clone()], self.model.get_device())?;
        let q_values = self.model.forward(&state)?.to_device(&Device::Cpu)?;

        let greedy_action_func =
            || -> Result<usize> { Ok(q_values.argmax(1)?.to_vec1::<u32>()?[0] as usize) };
        let random_action_func =
            || -> Result<usize> { Ok(rand::random::<usize>() % self.action_size) };

        let action_idx =
            self.explorer
                .select_action(self.t, &random_action_func, &greedy_action_func)?;

        let action = Tensor::from_slice(&[action_idx as u32], &[1], &Device::Cpu)?.detach();

        self.replay_buffer
            .append(state, Some(action.clone()), reward, false, self.gamma);
        if self.t % self.update_interval == 0 {
            self._update()?;
        }
        if self.t % self.target_update_interval == 0 {
            self._sync_target_model();
        }
        Ok(action)
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) -> Result<()> {
        let state = batch_states(&vec![obs.clone()], self.model.get_device())?;
        self.replay_buffer
            .append(state, None, reward, true, self.gamma);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explorers::EpsilonGreedy;
    use crate::models::FCQNetwork;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{optim, VarBuilder, VarMap};

    #[test]
    fn test_dqn_new() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = FCQNetwork::new(vb, 4, 2, 2, Some(64));
        let optimizer =
            optim::AdamW::new(var_map.all_vars(), optim::ParamsAdamW::default()).unwrap();
        let explorer = EpsilonGreedy::new(1.0, 0.1, 1000);

        let dqn = DQN::new(
            Box::new(model),
            optimizer,
            2,
            32,
            1000,
            8,
            100,
            Box::new(explorer),
            0.99,
            3,
        );

        assert_eq!(dqn.action_size, 2);
        assert_eq!(dqn.batch_size, 32);
        assert_eq!(dqn.update_interval, 8);
        assert_eq!(dqn.target_update_interval, 100);
        assert_eq!(dqn.gamma, 0.99);
        assert_eq!(dqn.n_steps, 3);
        assert_eq!(dqn.t, 0);
    }

    #[test]
    fn test_dqn_act_and_train() {
        let var_map = VarMap::new();
        let device = Device::cuda_if_available(0).unwrap();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = FCQNetwork::new(vb, 4, 4, 3, Some(1024));
        let params = optim::ParamsAdamW {
            lr: 0.0001,
            ..Default::default()
        };
        let optimizer = optim::AdamW::new(var_map.all_vars(), params).unwrap();
        let explorer = EpsilonGreedy::new(1.0, 0.0, 1000);
        let mut dqn = DQN::new(
            Box::new(model),
            optimizer,
            4,
            16,
            1000,
            50,
            100,
            Box::new(explorer),
            0.5,
            1,
        );

        let mut reward = 0.0;
        for i in 0..2000 {
            let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap();
            let action = dqn.act_and_train(&obs, reward).unwrap();
            let action_value = action.to_vec1::<u32>().unwrap()[0];
            if action_value == 2 {
                reward = 100.0;
            } else {
                reward = 0.0
            }
            assert!([0, 1, 2, 3].contains(&action_value));
            assert_eq!(dqn.t, i + 1);
            if dqn.t > 1000 {
                assert_eq!(action_value, 2);
            }
        }
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap();
        dqn.stop_episode_and_train(&obs, 1.0).unwrap();

        for _ in 0..1000 {
            let action = dqn.act(&obs).unwrap();
            let action_value = action.to_vec1::<u32>().unwrap()[0];
            assert_eq!(action_value, 2);
        }
    }
}
