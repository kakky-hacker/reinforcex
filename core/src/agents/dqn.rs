use super::base_agent::{ensure_parent_dir, BaseAgent};
use crate::explorers::BaseExplorer;
use crate::memory::{Experience, ReplayBuffer};
use crate::misc::batch_states::batch_states;
use crate::models::BaseQFunction;
use crate::selector::BaseSelector;
use std::{ops::Deref, sync::Arc};
use tch::{nn, no_grad, Tensor};
use ulid::Ulid;

pub struct DQN {
    agent_id: Ulid,
    model: Box<dyn BaseQFunction>,
    optimizer: nn::Optimizer,
    replay_buffer: Arc<ReplayBuffer>,
    explorer: Box<dyn BaseExplorer>,
    selector: Option<Arc<Box<dyn BaseSelector>>>,
    action_size: usize,
    batch_size: usize,
    update_interval: usize,
    target_model: Box<dyn BaseQFunction>,
    target_update_interval: usize,
    gamma: f64,
    t: usize,
    update_count: usize,
    last_loss: Option<f64>,
    current_episode_id: Ulid,
    save_path: Option<String>,
    load_path: Option<String>,
}

unsafe impl Send for DQN {}

impl DQN {
    pub fn new(
        model: Box<dyn BaseQFunction>,
        replay_buffer: Arc<ReplayBuffer>,
        optimizer: nn::Optimizer,
        action_size: usize,
        batch_size: usize,
        update_interval: usize,
        target_update_interval: usize,
        explorer: Box<dyn BaseExplorer>,
        selector: Option<Arc<Box<dyn BaseSelector>>>,
        gamma: f64,
        save_path: Option<String>,
        load_path: Option<String>,
    ) -> Self {
        let target_model = model.clone();
        let mut agent = DQN {
            agent_id: Ulid::new(),
            model,
            optimizer,
            replay_buffer,
            explorer,
            selector,
            action_size,
            batch_size,
            update_interval,
            target_model,
            target_update_interval,
            gamma,
            t: 0,
            update_count: 0,
            last_loss: None,
            current_episode_id: Ulid::new(),
            save_path,
            load_path,
        };
        agent.load();
        agent
    }

    fn _update(&mut self) {
        if self.replay_buffer.len() < self.batch_size {
            return;
        }
        let experiences = self.replay_buffer.sample(self.batch_size, true);
        let mut states: Vec<Tensor> = vec![];
        let mut n_step_after_states: Vec<Tensor> = vec![];
        let mut actions: Vec<Tensor> = vec![];
        let mut n_step_discounted_rewards: Vec<f64> = vec![];
        let mut non_terminal: Vec<f64> = vec![];
        for experience in experiences {
            let state = experience.state.shallow_clone();
            let n_step_after_experience = experience
                .n_step_after_experience
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .clone();
            let n_step_after_state = n_step_after_experience.state.shallow_clone();
            let action = experience.action.as_ref().unwrap().shallow_clone();
            let n_step_discounted_reward = experience
                .n_step_discounted_reward
                .lock()
                .unwrap()
                .unwrap_or(experience.reward);
            states.push(state);
            n_step_after_states.push(n_step_after_state);
            actions.push(action);
            n_step_discounted_rewards.push(n_step_discounted_reward);
            non_terminal.push(if n_step_after_experience.is_episode_terminal {
                0.0
            } else {
                1.0
            });
        }
        let q_values = self._compute_q_values(
            &n_step_after_states,
            &n_step_discounted_rewards,
            &non_terminal,
        );
        let pred_q_values = self._compute_pred_q_values(&states, &actions);
        let loss = self._compute_loss(q_values, pred_q_values);
        self.last_loss = Some(loss.double_value(&[]));
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.clip_grad_norm(10.0);
        self.optimizer.step();
        self.update_count += 1;
    }

    fn _sync_target_model(&mut self) {
        self.target_model = self.model.clone();
    }

    fn _compute_q_values(
        &self,
        n_step_after_states: &Vec<Tensor>,
        n_step_discounted_rewards: &Vec<f64>,
        non_terminal: &Vec<f64>,
    ) -> Tensor {
        assert_eq!(n_step_after_states.len(), n_step_discounted_rewards.len());
        assert_eq!(n_step_after_states.len(), non_terminal.len());
        let _states = batch_states(n_step_after_states, self.model.device());
        // Double-DQN
        let max_q_values = no_grad(|| {
            self.target_model
                .forward(&_states)
                .gather(1, &self.model.forward(&_states).argmax(1, true), false)
                .squeeze_dim(1)
        });
        let gamma_n = self.gamma.powi(self.replay_buffer.get_n_steps() as i32);
        let n_step_discounted_rewards_tensor = Tensor::from_slice(n_step_discounted_rewards)
            .to_kind(tch::Kind::Float)
            .to_device(self.model.device());
        let non_terminal_tensor = Tensor::from_slice(non_terminal)
            .to_kind(tch::Kind::Float)
            .to_device(self.model.device());
        let updated_q_values =
            max_q_values * gamma_n * non_terminal_tensor + n_step_discounted_rewards_tensor;
        updated_q_values
    }

    fn _compute_pred_q_values(&self, states: &Vec<Tensor>, actions: &Vec<Tensor>) -> Tensor {
        assert_eq!(states.len(), actions.len());
        let _states = batch_states(states, self.model.device());
        let pred_q_values = self.model.forward(&_states);
        let actions = Tensor::stack(actions, 0)
            .to_kind(tch::Kind::Int64)
            .to_device(self.model.device());
        let pred_q_values_selected = pred_q_values.gather(1, &actions, false).squeeze();
        pred_q_values_selected
    }

    fn _compute_loss(&self, q_values: Tensor, pred_q_values: Tensor) -> Tensor {
        pred_q_values.huber_loss(&q_values, tch::Reduction::Mean, 1.0)
    }

    pub fn get_model(&self) -> &Box<dyn BaseQFunction> {
        &self.model
    }

    pub fn copy_model_from(&mut self, agent: &DQN) {
        self.model = agent.get_model().deref().clone();
        self._sync_target_model();
    }
}

impl BaseAgent for DQN {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
            let q_values = self.model.forward(&state);
            q_values.argmax(1, false)
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;
        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let q_values = self.model.forward(&state);

        let greedy_action_func = || q_values.argmax(1, false).int64_value(&[0]) as usize;
        let random_action_func = || rand::random::<usize>() % self.action_size;

        let action_idx =
            self.explorer
                .select_action(self.t, &random_action_func, &greedy_action_func);
        let action = Tensor::from_slice(&[action_idx as i64]).detach();

        let experience = Arc::new(Experience::new(
            self.agent_id,
            self.current_episode_id,
            state,
            Some(action.shallow_clone()),
            None,
            reward,
            false,
        ));
        self.replay_buffer.append(experience.clone(), self.gamma);

        if self.selector.is_some() {
            self.selector.as_ref().unwrap().observe(experience.as_ref());
        }

        if self.t % self.update_interval == 0 {
            self._update();
        }
        if self.t % self.target_update_interval == 0 {
            self._sync_target_model();
        }
        action
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let experience = Arc::new(Experience::new(
            self.agent_id,
            self.current_episode_id,
            state,
            None,
            None,
            reward,
            true,
        ));
        self.replay_buffer.append(experience, self.gamma);
        self.current_episode_id = Ulid::new();
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        let mut statistics = vec![
            ("replay_size".to_string(), self.replay_buffer.len() as f64),
            ("updates".to_string(), self.update_count as f64),
        ];
        if let Some(loss) = self.last_loss {
            statistics.push(("loss".to_string(), loss));
        }
        statistics
    }

    fn get_agent_id(&self) -> &Ulid {
        &self.agent_id
    }

    fn save(&self) {
        if let Some(path) = &self.save_path {
            if path.is_empty() {
                return;
            }
            ensure_parent_dir(path);
            self.model.save(path);
        }
    }

    fn load(&mut self) {
        if let Some(path) = self.load_path.clone() {
            if path.is_empty() {
                return;
            }
            self.model.load(&path);
            self._sync_target_model();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explorers::EpsilonGreedy;
    use crate::models::FCQNetwork;
    use std::sync::Arc;
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    #[test]
    fn test_dqn_new() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCQNetwork::new(vs, 4, 2, 2, 64);
        let explorer = EpsilonGreedy::new(1.0, 0.1, 1000);
        let replay_buffer = Arc::new(ReplayBuffer::new(1000, 3));

        let dqn = DQN::new(
            Box::new(model),
            replay_buffer,
            optimizer,
            2,
            32,
            8,
            100,
            Box::new(explorer),
            None,
            0.99,
            None,
            None,
        );

        assert_eq!(dqn.action_size, 2);
        assert_eq!(dqn.batch_size, 32);
        assert_eq!(dqn.update_interval, 8);
        assert_eq!(dqn.target_update_interval, 100);
        assert_eq!(dqn.gamma, 0.99);
        assert_eq!(dqn.t, 0);
    }

    #[test]
    fn test_terminal_target_does_not_bootstrap() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCQNetwork::new(vs, 4, 2, 1, 32);
        let explorer = EpsilonGreedy::new(0.0, 0.0, 1);
        let replay_buffer = Arc::new(ReplayBuffer::new(100, 3));
        let dqn = DQN::new(
            Box::new(model),
            replay_buffer,
            optimizer,
            2,
            8,
            1,
            100,
            Box::new(explorer),
            None,
            0.99,
            None,
            None,
        );
        let next_state = Tensor::from_slice(&[0.1_f32, 0.2, 0.3, 0.4]);
        let targets = dqn._compute_q_values(&vec![next_state], &vec![-1.0], &vec![0.0]);

        assert!((targets.double_value(&[0]) + 1.0).abs() < 1e-6);
        assert!(!targets.requires_grad());
    }

    #[test]
    fn test_dqn_act_and_train() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-2).unwrap();
        let model = FCQNetwork::new(vs, 4, 4, 2, 128);
        let explorer = EpsilonGreedy::new(1.0, 0.0, 1000);
        let replay_buffer = Arc::new(ReplayBuffer::new(1000, 1));
        let mut dqn = DQN::new(
            Box::new(model),
            replay_buffer,
            optimizer,
            4,
            16,
            50,
            100,
            Box::new(explorer),
            None,
            0.5,
            None,
            None,
        );

        let mut reward = 0.0;
        let mut n = 0;
        let mut m = 0;
        for i in 0..2000 {
            let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
            let action = dqn.act_and_train(&obs, reward);
            let action_value = i64::from(action.int64_value(&[]));
            if action_value == 2 {
                reward = 100.0;
            } else {
                reward = 0.0
            }
            assert!([0, 1, 2, 3].contains(&action_value));
            assert_eq!(dqn.t, i + 1);
            if dqn.t > 1000 {
                if action_value == 2 {
                    n += 1;
                } else {
                    m += 1
                }
            }
        }
        assert!(n > m, "the learned action should dominate during training");

        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
        dqn.stop_episode_and_train(&obs, 1.0);

        for _ in 0..1000 {
            let action = dqn.act(&obs);
            let action_value = i64::from(action.int64_value(&[]));
            assert_eq!(action_value, 2);
        }
    }

    #[test]
    fn test_dqn_act_and_train_parallel() {
        use rayon::prelude::*;
        use std::sync::Arc;
        use tch::{Device, Kind, Tensor};

        let buffer = Arc::new(ReplayBuffer::new(10000, 1));
        let n_threads = 3;

        (0..n_threads).into_par_iter().for_each(|_| {
            let vs = nn::VarStore::new(Device::Cpu);
            let opt = nn::Adam::default().build(&vs, 1e-2).unwrap();
            let model = FCQNetwork::new(vs, 4, 4, 2, 128);
            let explorer = EpsilonGreedy::new(1.0, 0.0, 1000);
            let mut dqn = DQN::new(
                Box::new(model),
                Arc::clone(&buffer),
                opt,
                4,
                8,
                16,
                100,
                Box::new(explorer),
                None,
                0.5,
                None,
                None,
            );

            let mut reward = 0.0;

            let mut n = 0;
            let mut m = 0;

            for t in 0..2000 {
                let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
                let action = dqn.act_and_train(&obs, reward);
                let action_value = i64::from(action.int64_value(&[]));
                reward = if action_value == 2 { 100.0 } else { 0.0 };

                assert!([0, 1, 2, 3].contains(&action_value));
                assert_eq!(dqn.t, t + 1);
                if dqn.t > 1000 {
                    if action_value == 2 {
                        n += 1;
                    } else {
                        m += 1
                    }
                }
            }
            assert!((n / (n + m)) as f32 > 0.99);
        });
    }
}
