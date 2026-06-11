use super::base_agent::{ensure_parent_dir, BaseAgent};
use crate::memory::{Experience, ReplayBuffer};
use crate::misc::batch_states::batch_states;
use crate::models::{BasePolicy, BaseQFunction};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tch::{nn, nn::OptimizerConfig, no_grad, Device, Kind, Tensor};
use ulid::Ulid;

const LOG_PROB_EPSILON: f64 = 1e-6;
const DISCRETE_TARGET_ENTROPY_RATIO: f64 = 0.98;
const DISCRETE_ALPHA_LR: f64 = 3e-4;

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
    target_update_interval: usize,
    gamma: f64,
    tau: f64,
    alpha: f64,
    _log_alpha_vs: nn::VarStore,
    log_alpha: Tensor,
    alpha_optimizer: nn::Optimizer,
    discrete_target_entropy: Option<f64>,
    squash_action: bool,
    t: usize,
    current_episode_id: Ulid,
    latest_actor_loss: Option<f64>,
    latest_critic1_loss: Option<f64>,
    latest_critic2_loss: Option<f64>,
    latest_temperature_loss: Option<f64>,
    n_updates: usize,
    save_path: Option<String>,
    load_path: Option<String>,
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
        target_update_interval: usize,
        gamma: f64,
        tau: f64,
        alpha: f64,
        squash_action: bool,
    ) -> Self {
        Self::new_with_save_load(
            actor,
            actor_optimizer,
            critic1,
            critic1_optimizer,
            critic2,
            critic2_optimizer,
            transition_buffer,
            replay_start_size,
            batch_size,
            update_interval,
            target_update_interval,
            gamma,
            tau,
            alpha,
            squash_action,
            None,
            None,
        )
    }

    pub fn new_with_save_load(
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
        target_update_interval: usize,
        gamma: f64,
        tau: f64,
        alpha: f64,
        squash_action: bool,
        save_path: Option<String>,
        load_path: Option<String>,
    ) -> Self {
        assert!(replay_start_size > 0);
        assert!(batch_size > 0);
        assert!(update_interval > 0);
        assert!(target_update_interval > 0);
        assert!((0.0..=1.0).contains(&gamma));
        assert!((0.0..=1.0).contains(&tau));
        assert!(alpha >= 0.0);
        assert_eq!(actor.device(), critic1.device());
        assert_eq!(actor.device(), critic2.device());

        let target_critic1 = critic1.clone();
        let target_critic2 = critic2.clone();
        let log_alpha_vs = nn::VarStore::new(actor.device());
        let log_alpha = log_alpha_vs.root().var(
            "log_alpha",
            &[1],
            nn::Init::Const(alpha.max(LOG_PROB_EPSILON).ln()),
        );
        let alpha_optimizer = nn::Adam::default()
            .build(&log_alpha_vs, DISCRETE_ALPHA_LR)
            .unwrap();

        let mut agent = SAC {
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
            target_update_interval,
            gamma,
            tau,
            alpha,
            _log_alpha_vs: log_alpha_vs,
            log_alpha,
            alpha_optimizer,
            discrete_target_entropy: None,
            squash_action,
            t: 0,
            current_episode_id: Ulid::new(),
            latest_actor_loss: None,
            latest_critic1_loss: None,
            latest_critic2_loss: None,
            latest_temperature_loss: None,
            n_updates: 0,
            save_path,
            load_path,
        };
        agent.load();
        agent
    }

    fn _update(&mut self) {
        if self.transition_buffer.len() < self.replay_start_size.max(self.batch_size) {
            return;
        }

        let batch = self._sample_batch();

        if self._is_discrete_policy(&batch.states) {
            self._update_discrete(&batch);
        } else {
            self._update_continuous(&batch);
        }

        self.n_updates += 1;
    }

    fn _update_target_model(&mut self) {
        self.target_critic1
            .soft_update_from(self.critic1.as_ref(), self.tau);
        self.target_critic2
            .soft_update_from(self.critic2.as_ref(), self.tau);
    }

    fn _update_continuous(&mut self, batch: &SACBatch) {
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
    }

    fn _update_discrete(&mut self, batch: &SACBatch) {
        let batch_size = batch.states.size()[0];
        let gamma_n = self.gamma.powi(self.transition_buffer.get_n_steps() as i32);
        let target_q = no_grad(|| {
            let (next_action_distrib, _) = self.actor.forward(&batch.next_states);
            let next_prob = next_action_distrib.all_prob();
            let next_log_prob = next_action_distrib.all_log_prob();
            let next_q1 = self
                .target_critic1
                .forward(&batch.next_states)
                .view([batch_size, -1]);
            let next_q2 = self
                .target_critic2
                .forward(&batch.next_states)
                .view([batch_size, -1]);
            Self::_assert_discrete_q_shape("target_critic1", &next_q1, &next_prob);
            Self::_assert_discrete_q_shape("target_critic2", &next_q2, &next_prob);

            let next_q = next_q1.minimum(&next_q2);
            let next_v = (&next_prob * (next_q - self.alpha * next_log_prob)).sum_dim_intlist(
                [-1].as_ref(),
                false,
                Kind::Float,
            );
            &batch.rewards + gamma_n * &batch.non_terminal * next_v
        });

        let action_indices = batch.actions.to_kind(Kind::Int64).view([batch_size, 1]);
        let q1_values = self.critic1.forward(&batch.states).view([batch_size, -1]);
        Self::_assert_action_indices(&action_indices, q1_values.size()[1]);
        let pred_q1 = q1_values.gather(1, &action_indices, false).view([-1]);
        let critic1_loss: Tensor = (&pred_q1 - &target_q).square().mean(Kind::Float);
        assert!(critic1_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_critic1_loss = Some(critic1_loss.double_value(&[]));
        self.critic1_optimizer.zero_grad();
        critic1_loss.backward();
        self.critic1_optimizer.step();

        let q2_values = self.critic2.forward(&batch.states).view([batch_size, -1]);
        Self::_assert_action_indices(&action_indices, q2_values.size()[1]);
        let pred_q2 = q2_values.gather(1, &action_indices, false).view([-1]);
        let critic2_loss: Tensor = (&pred_q2 - &target_q).square().mean(Kind::Float);
        assert!(critic2_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_critic2_loss = Some(critic2_loss.double_value(&[]));
        self.critic2_optimizer.zero_grad();
        critic2_loss.backward();
        self.critic2_optimizer.step();

        let (action_distrib, _) = self.actor.forward(&batch.states);
        let prob = action_distrib.all_prob();
        let log_prob = action_distrib.all_log_prob();
        let q = no_grad(|| {
            let q1 = self.critic1.forward(&batch.states).view([batch_size, -1]);
            let q2 = self.critic2.forward(&batch.states).view([batch_size, -1]);
            Self::_assert_discrete_q_shape("critic1", &q1, &prob);
            Self::_assert_discrete_q_shape("critic2", &q2, &prob);
            q1.minimum(&q2)
        });

        let entropies = -(&prob * &log_prob).sum_dim_intlist([-1].as_ref(), false, Kind::Float);
        let actor_loss = (&prob * (self.alpha * &log_prob - q))
            .sum_dim_intlist([-1].as_ref(), false, Kind::Float)
            .mean(Kind::Float);
        assert!(actor_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_actor_loss = Some(actor_loss.double_value(&[]));
        self.actor_optimizer.zero_grad();
        actor_loss.backward();
        self.actor_optimizer.step();

        let target_entropy = self._discrete_target_entropy(&prob);
        let temperature_loss =
            -(&self.log_alpha * (target_entropy - entropies.detach())).mean(Kind::Float);
        assert!(temperature_loss.isnan().any().int64_value(&[]) == 0);

        self.latest_temperature_loss = Some(temperature_loss.double_value(&[]));
        self.alpha_optimizer.zero_grad();
        temperature_loss.backward();
        self.alpha_optimizer.step();
        self.alpha = self.log_alpha.exp().double_value(&[0]);
    }

    fn _is_discrete_policy(&self, states: &Tensor) -> bool {
        no_grad(|| {
            let (action_distrib, _) = self.actor.forward(states);
            action_distrib.is_discrete()
        })
    }

    fn _assert_discrete_q_shape(name: &str, q: &Tensor, prob: &Tensor) {
        assert_eq!(
            q.size(),
            prob.size(),
            "{} output shape must match policy probabilities for Discrete SAC",
            name
        );
    }

    fn _discrete_target_entropy(&mut self, prob: &Tensor) -> f64 {
        *self.discrete_target_entropy.get_or_insert_with(|| {
            let n_actions = prob.size()[1] as f64;
            n_actions.ln() * DISCRETE_TARGET_ENTROPY_RATIO
        })
    }

    fn _assert_action_indices(action_indices: &Tensor, n_actions: i64) {
        assert!(n_actions > 0);
        let min_action = action_indices.min().int64_value(&[]);
        let max_action = action_indices.max().int64_value(&[]);
        assert!(
            0 <= min_action && max_action < n_actions,
            "discrete action index out of range: min={}, max={}, n_actions={}",
            min_action,
            max_action,
            n_actions
        );
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

    fn _checkpoint_path(base_path: &str, component: &str) -> String {
        let path = Path::new(base_path);
        let file_name = path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .filter(|file_name| !file_name.is_empty())
            .unwrap_or("sac");

        let component_file_name = match (
            path.file_stem().and_then(|stem| stem.to_str()),
            path.extension().and_then(|extension| extension.to_str()),
        ) {
            (Some(stem), Some(extension)) if !stem.is_empty() && !extension.is_empty() => {
                format!("{}_{}.{}", stem, component, extension)
            }
            _ => format!("{}_{}", file_name, component),
        };

        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => PathBuf::from(parent)
                .join(component_file_name)
                .to_string_lossy()
                .into_owned(),
            _ => component_file_name,
        }
    }

    fn _checkpoint_paths(base_path: &str) -> [(String, String); 4] {
        [
            (
                "actor".to_string(),
                Self::_checkpoint_path(base_path, "actor"),
            ),
            (
                "critic1".to_string(),
                Self::_checkpoint_path(base_path, "critic1"),
            ),
            (
                "critic2".to_string(),
                Self::_checkpoint_path(base_path, "critic2"),
            ),
            (
                "temperature".to_string(),
                Self::_checkpoint_path(base_path, "temperature"),
            ),
        ]
    }
}

impl BaseAgent for SAC {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.actor.device());
            let (action_distrib, _) = self.actor.forward(&state);
            let action = action_distrib.most_probable();
            if action_distrib.is_discrete() {
                return action.to_device(Device::Cpu);
            }

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
        let action = no_grad(|| {
            let (action_distrib, _) = self.actor.forward(&state);
            if action_distrib.is_discrete() {
                if self.transition_buffer.len() < self.replay_start_size {
                    let probs = action_distrib.all_prob();
                    Tensor::randint(
                        probs.size()[1],
                        &[probs.size()[0]],
                        (Kind::Int64, self.actor.device()),
                    )
                } else {
                    action_distrib.sample()
                }
            } else {
                self._sample_action_and_log_prob(&state).0
            }
        });
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

        if self.t % self.target_update_interval == 0 {
            self._update_target_model();
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
        if let Some(loss) = self.latest_temperature_loss {
            statistics.push(("temperature_loss".to_string(), loss));
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

            let checkpoints = Self::_checkpoint_paths(path);
            for (_, checkpoint_path) in &checkpoints {
                ensure_parent_dir(checkpoint_path);
            }

            self.actor.save(&checkpoints[0].1);
            self.critic1.save(&checkpoints[1].1);
            self.critic2.save(&checkpoints[2].1);
            self._log_alpha_vs
                .save(&checkpoints[3].1)
                .unwrap_or_else(|e| {
                    panic!(
                        "failed to save SAC temperature to {}: {}",
                        checkpoints[3].1, e
                    )
                });
        }
    }

    fn load(&mut self) {
        if let Some(path) = self.load_path.clone() {
            if path.is_empty() {
                return;
            }

            let checkpoints = Self::_checkpoint_paths(&path);
            self.actor.load(&checkpoints[0].1);
            self.critic1.load(&checkpoints[1].1);
            self.critic2.load(&checkpoints[2].1);
            self._log_alpha_vs
                .load(&checkpoints[3].1)
                .unwrap_or_else(|e| {
                    panic!(
                        "failed to load SAC temperature from {}: {}",
                        checkpoints[3].1, e
                    )
                });
            self.alpha = self.log_alpha.exp().double_value(&[0]);
            self.target_critic1.copy_from(self.critic1.as_ref());
            self.target_critic2.copy_from(self.critic2.as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FCGaussianPolicy, FCQNetwork, FCSoftmaxPolicy};
    use crate::prob_distributions::{BaseDistribution, SoftmaxDistribution};
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    struct TestSoftmaxPolicy {
        vs: nn::VarStore,
        linear: nn::Linear,
        obs_size: i64,
    }

    impl TestSoftmaxPolicy {
        fn new(vs: nn::VarStore, obs_size: i64, n_actions: i64) -> Self {
            let linear = nn::linear(
                &vs.root(),
                obs_size,
                n_actions,
                nn::LinearConfig {
                    ws_init: nn::Init::Const(0.0),
                    bs_init: Some(nn::Init::Const(0.0)),
                    bias: true,
                },
            );

            Self {
                vs,
                linear,
                obs_size,
            }
        }
    }

    impl BasePolicy for TestSoftmaxPolicy {
        fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
            let logits = self.linear.forward(&x.view([-1, self.obs_size]));
            (Box::new(SoftmaxDistribution::new(logits, 1.0, 0.0)), None)
        }

        fn device(&self) -> Device {
            self.vs.device()
        }

        fn save(&self, _path: &str) {}

        fn load(&mut self, _path: &str) {}
    }

    struct TestQNetwork {
        vs: nn::VarStore,
        linear: nn::Linear,
        obs_size: i64,
        n_actions: i64,
    }

    impl TestQNetwork {
        fn new(vs: nn::VarStore, obs_size: i64, n_actions: i64) -> Self {
            let linear = nn::linear(
                &vs.root(),
                obs_size,
                n_actions,
                nn::LinearConfig {
                    ws_init: nn::Init::Const(0.0),
                    bs_init: Some(nn::Init::Const(0.0)),
                    bias: true,
                },
            );

            Self {
                vs,
                linear,
                obs_size,
                n_actions,
            }
        }
    }

    impl BaseQFunction for TestQNetwork {
        fn forward(&self, x: &Tensor) -> Tensor {
            self.linear
                .forward(&x.view([-1, self.obs_size]))
                .view([-1, self.n_actions])
        }

        fn device(&self) -> Device {
            self.vs.device()
        }

        fn clone(&self) -> Box<dyn BaseQFunction> {
            let cloned = Self::new(
                nn::VarStore::new(self.device()),
                self.obs_size,
                self.n_actions,
            );
            let mut target_variables = cloned.trainable_variables();
            let source_variables = self.trainable_variables();
            no_grad(|| {
                for (target, source) in target_variables.iter_mut().zip(source_variables.iter()) {
                    target.copy_(source);
                }
            });
            Box::new(cloned)
        }

        fn save(&self, _path: &str) {}

        fn load(&mut self, _path: &str) {}

        fn trainable_variables(&self) -> Vec<Tensor> {
            self.vs.trainable_variables()
        }
    }

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
            1,
            0.99,
            0.005,
            0.2,
            true,
        )
    }

    fn build_discrete_sac() -> SAC {
        let device = Device::Cpu;
        let obs_size = 4;
        let n_actions = 2;
        let hidden_layers = 1;
        let hidden_channels = 16;

        let actor_vs = nn::VarStore::new(device);
        let actor_optimizer = nn::Adam::default().build(&actor_vs, 1e-3).unwrap();
        let actor = FCSoftmaxPolicy::new(
            actor_vs,
            obs_size,
            n_actions,
            hidden_layers,
            hidden_channels,
            0.0,
        );

        let critic1_vs = nn::VarStore::new(device);
        let critic1_optimizer = nn::Adam::default().build(&critic1_vs, 1e-3).unwrap();
        let critic1 = FCQNetwork::new(
            critic1_vs,
            obs_size,
            n_actions,
            hidden_layers,
            hidden_channels,
        );

        let critic2_vs = nn::VarStore::new(device);
        let critic2_optimizer = nn::Adam::default().build(&critic2_vs, 1e-3).unwrap();
        let critic2 = FCQNetwork::new(
            critic2_vs,
            obs_size,
            n_actions,
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
            1,
            0.99,
            0.005,
            0.2,
            false,
        )
    }

    fn build_deterministic_discrete_sac() -> SAC {
        let device = Device::Cpu;
        let obs_size = 4;
        let n_actions = 2;

        let actor_vs = nn::VarStore::new(device);
        let actor_optimizer = nn::Adam::default().build(&actor_vs, 1e-2).unwrap();
        let actor = TestSoftmaxPolicy::new(actor_vs, obs_size, n_actions);

        let critic1_vs = nn::VarStore::new(device);
        let critic1_optimizer = nn::Adam::default().build(&critic1_vs, 1e-2).unwrap();
        let critic1 = TestQNetwork::new(critic1_vs, obs_size, n_actions);

        let critic2_vs = nn::VarStore::new(device);
        let critic2_optimizer = nn::Adam::default().build(&critic2_vs, 1e-2).unwrap();
        let critic2 = TestQNetwork::new(critic2_vs, obs_size, n_actions);

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
            1,
            1,
            0.0,
            1.0,
            0.0,
            false,
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

    #[test]
    fn test_discrete_soft_actor_critic_act() {
        let sac = build_discrete_sac();
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);

        let action = sac.act(&obs);

        assert_eq!(action.size(), vec![1]);
        let action_id = action.int64_value(&[0]);
        assert!(0 <= action_id && action_id < 2);
    }

    #[test]
    fn test_discrete_soft_actor_critic_act_and_train() {
        let mut sac = build_deterministic_discrete_sac();
        let rewarded_action = 1;
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
        let action = Tensor::from_slice(&[rewarded_action]).to_kind(Kind::Int64);

        for _ in 0..6 {
            sac.transition_buffer.append(
                sac.agent_id,
                sac.current_episode_id,
                obs.shallow_clone(),
                Some(action.shallow_clone()),
                None,
                100.0,
                false,
                sac.gamma,
            );
        }

        for _ in 0..200 {
            sac._update();
        }

        assert!(sac.transition_buffer.len() > 0);
        assert!(sac.latest_actor_loss.is_some());
        assert!(sac.latest_critic1_loss.is_some());
        assert!(sac.latest_critic2_loss.is_some());

        for _ in 0..1000 {
            let action = sac.act(&obs);
            let action_id = action.int64_value(&[0]);
            assert_eq!(action_id, rewarded_action);
        }
    }
}
