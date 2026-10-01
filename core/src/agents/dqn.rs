use super::base_agent::{ensure_parent_dir, BaseAgent};
use crate::explorers::BaseExplorer;
use crate::memory::{Experience, ReplayBuffer};
use crate::misc::batch_states::batch_states;
use crate::misc::gradients::clip_grad_norm_f64;
use crate::models::BaseQFunction;
use crate::selector::BaseSelector;
use std::sync::Arc;
use crate::misc::autograd::no_grad;
use tch::{nn, Device, Kind, Tensor};
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
    // Last sampled batch, before the optimizer step: selected Q, target Q,
    // absolute TD error, and largest-minus-second-largest action Q.
    last_batch_diagnostics: Option<[f64; 4]>,
    current_episode_id: Ulid,
    save_path: Option<String>,
    load_path: Option<String>,
}

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
        assert!(action_size > 0);
        assert!(batch_size > 0);
        assert!(update_interval > 0);
        assert!(target_update_interval > 0);
        assert!((0.0..=1.0).contains(&gamma));
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
            last_batch_diagnostics: None,
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
        let mut horizons: Vec<usize> = vec![];
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
            horizons.push(experience.n_step_horizon.lock().unwrap().unwrap());
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
            &horizons,
        );
        let (pred_q_values, all_q_values) = self._compute_pred_q_values(&states, &actions);
        self._record_batch_diagnostics(&pred_q_values, &q_values, &all_q_values);
        let loss = self._compute_loss(&q_values, &pred_q_values);
        assert!(
            loss.isfinite().all().int64_value(&[]) != 0,
            "DQN loss must be finite"
        );
        self.last_loss = Some(loss.double_value(&[]));
        self.optimizer.zero_grad();
        loss.backward();
        clip_grad_norm_f64(&self.optimizer, 10.0, "DQN");
        self.optimizer.step();
        self.update_count += 1;
    }

    fn _sync_target_model(&mut self) {
        if self.target_model.supports_in_place_target_sync()
            && self.model.supports_in_place_target_sync()
        {
            self.target_model.copy_from(self.model.as_ref());
        } else {
            // Custom Q-functions may have state outside trainable_variables.
            self.target_model = self.model.clone();
        }
    }

    fn _compute_q_values(
        &self,
        n_step_after_states: &Vec<Tensor>,
        n_step_discounted_rewards: &Vec<f64>,
        non_terminal: &Vec<f64>,
        horizons: &Vec<usize>,
    ) -> Tensor {
        assert_eq!(n_step_after_states.len(), n_step_discounted_rewards.len());
        assert_eq!(n_step_after_states.len(), non_terminal.len());
        assert_eq!(n_step_after_states.len(), horizons.len());
        let _states = batch_states(n_step_after_states, self.model.device());
        // Double-DQN
        let max_q_values = no_grad(|| {
            self.target_model
                .forward(&_states)
                .gather(1, &self.model.forward(&_states).argmax(1, true), false)
                .squeeze_dim(1)
        });
        let gamma_n = Tensor::from_slice(
            &horizons
                .iter()
                .map(|&n| self.gamma.powi(n as i32))
                .collect::<Vec<_>>(),
        )
        .to_kind(tch::Kind::Float)
        .to_device(self.model.device());
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

    fn _compute_pred_q_values(
        &self,
        states: &Vec<Tensor>,
        actions: &Vec<Tensor>,
    ) -> (Tensor, Tensor) {
        assert_eq!(states.len(), actions.len());
        let _states = batch_states(states, self.model.device());
        let pred_q_values = self.model.forward(&_states);
        let actions = Tensor::stack(actions, 0)
            .to_kind(tch::Kind::Int64)
            .to_device(self.model.device());
        let pred_q_values_selected = pred_q_values.gather(1, &actions, false).squeeze_dim(1);
        (pred_q_values_selected, pred_q_values)
    }

    fn _record_batch_diagnostics(&mut self, prediction: &Tensor, target: &Tensor, all_q: &Tensor) {
        let values = no_grad(|| {
            let gap = if all_q.size()[1] > 1 {
                let best_two = all_q.topk(2, 1, true, true).0;
                (best_two.select(1, 0) - best_two.select(1, 1)).mean(Kind::Float)
            } else {
                Tensor::zeros([], (Kind::Float, all_q.device()))
            };
            // Reuse the training forward pass and transfer these four scalars
            // together. No graph or replay tensors are retained in statistics.
            Tensor::stack(
                &[
                    prediction.mean(Kind::Float),
                    target.mean(Kind::Float),
                    (prediction - target).abs().mean(Kind::Float),
                    gap,
                ],
                0,
            )
            .to_device(Device::Cpu)
        });
        self.last_batch_diagnostics =
            Some(std::array::from_fn(|i| values.double_value(&[i as i64])));
    }

    fn _compute_loss(&self, q_values: &Tensor, pred_q_values: &Tensor) -> Tensor {
        pred_q_values.huber_loss(q_values, tch::Reduction::Mean, 1.0)
    }

    pub fn get_model(&self) -> &Box<dyn BaseQFunction> {
        &self.model
    }

    pub fn copy_model_from(&mut self, agent: &DQN) {
        // Keep the tensors registered with the optimizer alive. Replacing the
        // model would leave subsequent updates attached to the old parameters.
        self.model.copy_from(agent.get_model().as_ref());
        self._sync_target_model();
    }
}

impl Drop for DQN {
    fn drop(&mut self) {
        self.replay_buffer.discard_episode(&self.current_episode_id);
        if let Some(selector) = &self.selector {
            selector.delete(&self.agent_id);
        }
    }
}

impl BaseAgent for DQN {
    fn supports_learning_rate_update(&self) -> bool {
        true
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        assert!(learning_rate.is_finite() && learning_rate > 0.0);
        self.optimizer.set_lr(learning_rate);
    }

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
        let q_values = no_grad(|| self.model.forward(&state));

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
        self.stop_episode_and_train_with_terminal(obs, reward, true);
    }

    fn stop_episode_and_train_with_terminal(
        &mut self,
        obs: &Tensor,
        reward: f64,
        terminated: bool,
    ) {
        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let experience = Arc::new(
            Experience::new(
                self.agent_id,
                self.current_episode_id,
                state,
                None,
                None,
                reward,
                terminated,
            )
            .with_episode_end(true),
        );
        if let Some(selector) = &self.selector {
            selector.observe(experience.as_ref());
        }
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
        if let Some(values) = self.last_batch_diagnostics {
            for (name, value) in [
                "mean_q",
                "mean_target_q",
                "mean_abs_td_error",
                "mean_action_gap",
            ]
            .into_iter()
            .zip(values)
            {
                statistics.push((name.to_string(), value));
            }
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    fn small_dqn(n_steps: usize) -> DQN {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-2).unwrap();
        let model = FCQNetwork::new(vs, 4, 2, 1, 8);
        DQN::new(
            Box::new(model),
            Arc::new(ReplayBuffer::new(100, n_steps)),
            optimizer,
            2,
            1,
            1,
            100,
            Box::new(EpsilonGreedy::new(0.0, 0.0, 1)),
            None,
            0.5,
            None,
            None,
        )
    }

    // A one-action Q-function with an exactly controlled derivative. This lets
    // the tests exercise replay -> Double-DQN target -> Huber -> optimizer,
    // including a finite forward pass whose backward derivative is infinite.
    struct GradientProbeQ {
        weights: Tensor,
        square_root: bool,
    }

    impl BaseQFunction for GradientProbeQ {
        fn forward(&self, x: &Tensor) -> Tensor {
            let weights = if self.square_root {
                self.weights.sqrt()
            } else {
                self.weights.shallow_clone()
            };
            (x.view([-1, 2]) * weights).sum_dim_intlist([1].as_ref(), true, Kind::Float)
        }

        fn device(&self) -> Device {
            Device::Cpu
        }

        fn clone(&self) -> Box<dyn BaseQFunction> {
            Box::new(Self {
                weights: self.weights.detach().copy(),
                square_root: self.square_root,
            })
        }

        fn trainable_variables(&self) -> Vec<Tensor> {
            vec![self.weights.shallow_clone()]
        }

        fn save(&self, _: &str) {
            unimplemented!()
        }

        fn load(&mut self, _: &str) {
            unimplemented!()
        }
    }

    fn gradient_probe_dqn(state: [f32; 2], reward: f64, square_root: bool) -> (DQN, Tensor) {
        let store = nn::VarStore::new(Device::Cpu);
        let weights = store.root().var(
            "weights",
            &[2],
            nn::Init::Const(if square_root { 0.0 } else { 1.0 }),
        );
        let optimizer = nn::Adam::default().build(&store, 0.01).unwrap();
        let agent = DQN::new(
            Box::new(GradientProbeQ {
                weights: weights.shallow_clone(),
                square_root,
            }),
            Arc::new(ReplayBuffer::new(4, 1)),
            optimizer,
            1,
            1,
            1,
            100,
            Box::new(EpsilonGreedy::new(0.0, 0.0, 1)),
            None,
            0.99,
            None,
            None,
        );
        for (observation, action, observed_reward, terminal) in [
            (state, Some(Tensor::from_slice(&[0_i64])), 0.0, false),
            ([0.0, 0.0], None, reward, true),
        ] {
            agent.replay_buffer.append(
                Arc::new(Experience::new(
                    agent.agent_id,
                    agent.current_episode_id,
                    Tensor::from_slice(&observation),
                    action,
                    None,
                    observed_reward,
                    terminal,
                )),
                agent.gamma,
            );
        }
        assert_eq!(agent.replay_buffer.len(), 1);
        (agent, weights)
    }

    fn assert_independent_dqn_still_learns() {
        let (mut healthy, weights) = gradient_probe_dqn([1.0, -1.0], 2.0, false);
        assert!(weights.square().requires_grad());
        healthy._update();
        assert_eq!(healthy.update_count, 1);
        assert_eq!(healthy.last_loss, Some(1.5));
        assert!((weights.double_value(&[0]) - 1.01).abs() < 1e-6);
        assert!((weights.double_value(&[1]) - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_update_clips_large_finite_huber_gradients_without_losing_direction() {
        let (mut agent, weights) = gradient_probe_dqn([1e20, -1e20], 2.0, false);
        agent._update();
        assert_eq!(agent.update_count, 1);
        assert_eq!(agent.last_loss, Some(1.5));
        // Before clipping Huber's derivative is [-1e20, 1e20]. Its f32
        // norm overflows, but the two clipped components must remain nonzero.
        let gradient = weights.grad();
        assert!(gradient.isfinite().all().int64_value(&[]) != 0);
        assert!(gradient.double_value(&[0]) < -7.0);
        assert!(gradient.double_value(&[1]) > 7.0);
        assert!((gradient.to_kind(Kind::Double).norm().double_value(&[]) - 10.0).abs() < 1e-5);
        assert!((weights.double_value(&[0]) - 1.01).abs() < 1e-6);
        assert!((weights.double_value(&[1]) - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_update_rejects_nonfinite_huber_loss_before_backward_or_optimizer_step() {
        for reward in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let (mut agent, weights) = gradient_probe_dqn([1.0, -1.0], reward, false);
            let before = weights.copy();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                agent._update();
            }))
            .is_err());
            assert!(weights.equal(&before));
            assert!(!weights.grad().defined());
            assert_eq!(agent.last_loss, None);
            assert_eq!(agent.update_count, 0);
            assert_independent_dqn_still_learns();
        }
    }

    #[test]
    fn test_update_rejects_nonfinite_gradient_and_restores_autograd_for_another_agent() {
        let (mut agent, weights) = gradient_probe_dqn([1.0, 1.0], 2.0, true);
        let before = weights.copy();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent._update();
        }))
        .is_err());
        assert_eq!(agent.last_loss, Some(1.5)); // finite forward, invalid derivative
        assert!(weights.equal(&before));
        assert_eq!(weights.grad().isfinite().all().int64_value(&[]), 0);
        assert_eq!(agent.update_count, 0);
        assert_independent_dqn_still_learns();
    }

    // This Q-function has non-trainable forward state. Its default sync
    // capability must retain clone semantics rather than copy parameters only.
    struct BufferedQ {
        values: [f32; 2],
        offset: Tensor,
        clones: Arc<AtomicUsize>,
    }

    impl BaseQFunction for BufferedQ {
        fn forward(&self, x: &Tensor) -> Tensor {
            Tensor::from_slice(&self.values)
                .view([1, 2])
                .repeat([x.size()[0], 1])
                + &self.offset
        }

        fn device(&self) -> Device {
            Device::Cpu
        }

        fn clone(&self) -> Box<dyn BaseQFunction> {
            self.clones.fetch_add(1, Ordering::SeqCst);
            Box::new(Self {
                values: self.values,
                offset: self.offset.copy(),
                clones: self.clones.clone(),
            })
        }

        fn trainable_variables(&self) -> Vec<Tensor> {
            vec![]
        }

        fn save(&self, _: &str) {
            unimplemented!()
        }

        fn load(&mut self, _: &str) {
            unimplemented!()
        }
    }

    #[test]
    fn test_learning_rate_change_preserves_adam_moments_and_scales_next_step() {
        fn fixture() -> (DQN, Tensor) {
            let mut agent = small_dqn(1); // Adam, constructor rate 0.01.
            no_grad(|| {
                for mut parameter in agent.model.trainable_variables() {
                    let _ = parameter.fill_(0.25);
                }
            });
            agent._sync_target_model();
            let probe = agent.model.trainable_variables()[0]
                .flatten(0, -1)
                .narrow(0, 0, 1);
            (agent, probe)
        }
        fn step(agent: &mut DQN, probe: &Tensor, gradient: f64) {
            // Isolate optimizer behavior from replay sampling and Bellman targets.
            agent.optimizer.zero_grad();
            (probe.sum(Kind::Float) * gradient).backward();
            agent.optimizer.clip_grad_norm(10.0);
            agent.optimizer.step();
        }
        // The FFI constructs its optimizer before registering model variables.
        // A schedule set before the first update must reach those variables too.
        let (mut initial_rate, initial) = fixture();
        initial_rate.set_learning_rate(0.005);
        let initial_value = initial.double_value(&[0]);
        step(&mut initial_rate, &initial, 0.25);
        assert!((initial_value - initial.double_value(&[0]) - 0.005).abs() < 1e-6);
        let (mut unchanged, full) = fixture();
        let (mut changed, half) = fixture();
        let (mut same_rate, same) = fixture();
        for (agent, probe) in [
            (&mut unchanged, &full),
            (&mut changed, &half),
            (&mut same_rate, &same),
        ] {
            assert!(agent.supports_learning_rate_update());
            step(agent, probe, 0.25);
        }
        assert!(full.equal(&half) && full.equal(&same));
        let parameters = changed.model.trainable_variables();
        let before: Vec<_> = parameters.iter().map(Tensor::copy).collect();
        let statistics = changed.get_statistics();
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                changed.set_learning_rate(invalid);
            }))
            .is_err());
        }
        changed.set_learning_rate(0.005);
        same_rate.set_learning_rate(0.01);
        assert_eq!(changed.get_statistics(), statistics);
        for (parameter, saved) in parameters.iter().zip(&before) {
            assert!(
                parameter.equal(saved),
                "setting LR must not update parameters"
            );
        }
        let starting_value = full.double_value(&[0]);
        // The small opposing gradient cannot reverse Adam's accumulated moment.
        // Recreating Adam here would move in the opposite direction.
        step(&mut unchanged, &full, -0.01);
        step(&mut changed, &half, -0.01);
        step(&mut same_rate, &same, -0.01);
        let full_delta = starting_value - full.double_value(&[0]);
        let half_delta = starting_value - half.double_value(&[0]);
        assert!(
            full_delta > 0.0,
            "the first step's Adam moment must survive"
        );
        assert!((half_delta / full_delta - 0.5).abs() < 1e-4);
        assert!(
            full.equal(&same),
            "setting the existing rate must be a no-op"
        );
    }

    #[test]
    fn test_target_sync_preserves_storage_and_copies_independent_values() {
        let mut agent = small_dqn(1);
        let original_target = agent.target_model.trainable_variables();
        no_grad(|| {
            for (index, mut parameter) in agent.model.trainable_variables().into_iter().enumerate()
            {
                let _ = parameter.fill_((index + 1) as f64 / 10.0);
            }
        });
        agent._sync_target_model();
        let copied: Vec<Tensor> = agent
            .target_model
            .trainable_variables()
            .iter()
            .map(Tensor::copy)
            .collect();
        for ((previous, target), online) in original_target
            .iter()
            .zip(agent.target_model.trainable_variables())
            .zip(agent.model.trainable_variables())
        {
            assert_eq!(previous.data_ptr(), target.data_ptr());
            assert_ne!(online.data_ptr(), target.data_ptr());
            assert!(target.equal(&online));
        }
        no_grad(|| {
            for mut parameter in agent.model.trainable_variables() {
                let _ = parameter.fill_(2.0);
            }
        });
        for (target, saved) in agent.target_model.trainable_variables().iter().zip(&copied) {
            assert!(
                target.equal(saved),
                "target must remain frozen between syncs"
            );
        }

        let obs = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let target =
            agent._compute_q_values(&vec![obs.shallow_clone()], &vec![1.0], &vec![1.0], &vec![1]);
        assert!(!target.requires_grad());
        let (prediction, _) =
            agent._compute_pred_q_values(&vec![obs], &vec![Tensor::from_slice(&[0i64])]);
        agent._compute_loss(&target, &prediction).backward();
        assert!(agent
            .model
            .trainable_variables()
            .iter()
            .any(|p| p.grad().defined()));
        assert!(agent
            .target_model
            .trainable_variables()
            .iter()
            .all(|p| !p.grad().defined()));
    }

    #[test]
    fn test_target_sync_does_not_consume_torch_rng() {
        // Torch's RNG is process-global. Isolate this test from the other tests'
        // concurrent initializations rather than relying on test-thread order.
        const CHILD: &str = "REINFORCEX_DQN_TARGET_SYNC_RNG_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "agents::dqn::tests::test_target_sync_does_not_consume_torch_rng",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(
                status.success(),
                "isolated target-sync RNG regression failed"
            );
            return;
        }
        let mut agent = small_dqn(1);
        tch::manual_seed(137);
        let expected = Tensor::randn([32], (Kind::Float, Device::Cpu));
        tch::manual_seed(137);
        for _ in 0..3 {
            agent._sync_target_model();
        }
        let observed = Tensor::randn([32], (Kind::Float, Device::Cpu));
        assert!(observed.equal(&expected));
    }

    #[test]
    fn test_custom_target_sync_keeps_clone_fallback_for_buffers() {
        let mut agent = small_dqn(1);
        let clones = Arc::new(AtomicUsize::new(0));
        let mut offset = Tensor::from_slice(&[0.0f32]);
        agent.model = Box::new(BufferedQ {
            values: [2.0, 5.0],
            offset: offset.shallow_clone(),
            clones: clones.clone(),
        });
        assert!(!agent.model.supports_in_place_target_sync());
        agent._sync_target_model();
        assert_eq!(clones.load(Ordering::SeqCst), 1);
        let obs = Tensor::zeros([1, 4], (Kind::Float, Device::Cpu));
        let _ = offset.fill_(3.0);
        assert_eq!(agent.target_model.forward(&obs).double_value(&[0, 1]), 5.0);
        agent._sync_target_model();
        assert_eq!(clones.load(Ordering::SeqCst), 2);
        assert_eq!(agent.target_model.forward(&obs).double_value(&[0, 1]), 8.0);
        let _ = offset.fill_(9.0);
        assert_eq!(agent.target_model.forward(&obs).double_value(&[0, 1]), 8.0);
    }

    #[test]
    fn test_double_dqn_targets_use_online_choice_terminal_mask_and_actual_horizons() {
        let mut agent = small_dqn(3);
        let clones = Arc::new(AtomicUsize::new(0));
        agent.model = Box::new(BufferedQ {
            values: [2.0, 5.0],
            offset: Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            clones: clones.clone(),
        });
        agent.target_model = Box::new(BufferedQ {
            values: [10.0, 1.0],
            offset: Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            clones,
        });
        let states = (0..3)
            .map(|_| Tensor::zeros([4], (Kind::Float, Device::Cpu)))
            .collect();
        let targets = agent._compute_q_values(
            &states,
            &vec![1.0, 2.0, 3.0],
            &vec![1.0, 0.0, 1.0],
            &vec![1, 3, 3],
        );
        // Online chooses action 1; target evaluates that action as 1, not max=10.
        assert!(targets.allclose(
            &Tensor::from_slice(&[1.5f32, 2.0, 3.125]),
            1e-6,
            1e-6,
            false
        ));
        assert!(!targets.requires_grad());
    }

    #[test]
    fn test_three_step_episode_flush_preserves_rewards_and_reset_boundary() {
        for terminated in [true, false] {
            let mut agent = small_dqn(3);
            agent.batch_size = 100; // Inspect transitions without optimizer updates.
            for (index, reward) in [777.0, 2.0, 4.0].into_iter().enumerate() {
                let obs = Tensor::from_slice(&[index as f32, 0.0, 0.0, 0.0]);
                let _ = agent.act_and_train(&obs, reward);
            }
            agent.stop_episode_and_train_with_terminal(
                &Tensor::from_slice(&[3.0f32, 0.0, 0.0, 0.0]),
                8.0,
                terminated,
            );
            let _ = agent.act_and_train(&Tensor::from_slice(&[10.0f32, 0.0, 0.0, 0.0]), 999.0);
            agent.stop_episode_and_train(&Tensor::from_slice(&[11.0f32, 0.0, 0.0, 0.0]), 16.0);
            assert_eq!(agent.replay_buffer.len(), 4);
            for sample in agent.replay_buffer.sample(4, false) {
                let index = sample.state.view([-1]).double_value(&[0]) as usize;
                let (reward, horizon, next_state, terminal) = match index {
                    0 => (6.0, 3, 3.0, terminated),
                    1 => (8.0, 2, 3.0, terminated),
                    2 => (8.0, 1, 3.0, terminated),
                    10 => (16.0, 1, 11.0, true),
                    _ => panic!("unexpected stored transition"),
                };
                assert_eq!(
                    *sample.n_step_discounted_reward.lock().unwrap(),
                    Some(reward)
                );
                assert_eq!(*sample.n_step_horizon.lock().unwrap(), Some(horizon));
                let successor = sample.n_step_after_experience.lock().unwrap();
                let successor = successor.as_ref().unwrap();
                assert_eq!(successor.state.view([-1]).double_value(&[0]), next_state);
                assert_eq!(successor.is_episode_terminal, terminal);
                assert!(successor.is_episode_end);
            }
            assert_eq!(agent.update_count, 0);
        }
    }

    #[test]
    fn test_batch_diagnostics_are_pre_update_scalars_without_gradient_effects() {
        let mut agent = small_dqn(1);
        let prediction = Tensor::from_slice(&[2.0f32, 4.0]).set_requires_grad(true);
        let target = Tensor::from_slice(&[1.0f32, 7.0]);
        let all_q = Tensor::from_slice(&[2.0f32, 0.0, 1.0, 4.0]).view([2, 2]);
        agent._record_batch_diagnostics(&prediction, &target, &all_q);
        assert_eq!(agent.last_batch_diagnostics, Some([3.0, 4.0, 2.0, 2.5]));
        assert!(!prediction.grad().defined());
        agent._compute_loss(&target, &prediction).backward();
        assert!(prediction
            .grad()
            .equal(&Tensor::from_slice(&[0.5f32, -0.5])));
        let before = agent.get_statistics();
        let _ = agent.act(&Tensor::zeros([4], (Kind::Float, Device::Cpu)));
        assert_eq!(before, agent.get_statistics());
        agent._record_batch_diagnostics(&prediction, &target, &prediction.detach().view([2, 1]));
        assert_eq!(agent.last_batch_diagnostics.unwrap()[3], 0.0);
    }

    #[test]
    fn test_selector_observes_terminal_rewards() {
        struct Observer(Arc<std::sync::Mutex<Vec<f64>>>);
        impl BaseSelector for Observer {
            fn observe(&self, experience: &Experience) {
                self.0.lock().unwrap().push(experience.reward);
            }
            fn delete(&self, _: &Ulid) {}
            fn find_pareto_dominant(&self, _: &Ulid) -> Vec<Ulid> {
                vec![]
            }
        }
        let rewards = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut agent = small_dqn(1);
        agent.selector = Some(Arc::new(Box::new(Observer(rewards.clone()))));
        let obs = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        let _ = agent.act_and_train(&obs, 0.0);
        agent.stop_episode_and_train(&obs, 42.0);
        assert_eq!(*rewards.lock().unwrap(), vec![0.0, 42.0]);
    }

    #[test]
    fn test_copy_model_preserves_optimizer_parameters_and_learning() {
        let source = small_dqn(1);
        no_grad(|| {
            for mut variable in source.model.trainable_variables() {
                let _ = variable.fill_(0.0);
            }
        });
        let mut target = small_dqn(1);
        let original_parameters = target.model.trainable_variables();
        target.copy_model_from(&source);
        for (before, after) in original_parameters
            .iter()
            .zip(target.model.trainable_variables())
        {
            assert_eq!(before.data_ptr(), after.data_ptr());
        }
        let obs = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0]);
        assert!(target
            .model
            .forward(&obs)
            .equal(&source.model.forward(&obs)));
        let _ = target.act_and_train(&obs, 0.0);
        target.stop_episode_and_train(&obs, 3.0);
        target._update();
        assert_eq!(target.update_count, 1);
        assert!(target.model.forward(&obs).double_value(&[0, 0]) > 0.0);
        assert_eq!(source.model.forward(&obs).double_value(&[0, 0]), 0.0);
    }

    #[test]
    fn test_single_sample_prediction_keeps_batch_dimension() {
        let agent = small_dqn(1);
        let (predictions, _) = agent._compute_pred_q_values(
            &vec![Tensor::zeros([4], (Kind::Float, Device::Cpu))],
            &vec![Tensor::from_slice(&[0i64])],
        );
        assert_eq!(predictions.size(), vec![1]);
    }

    #[test]
    fn test_dropped_agent_releases_unfinished_shared_replay_tail() {
        let agent = small_dqn(5);
        let replay = agent.replay_buffer.clone();
        let pending = Arc::new(Experience::new(
            agent.agent_id,
            agent.current_episode_id,
            Tensor::zeros([4], (Kind::Float, Device::Cpu)),
            Some(Tensor::from_slice(&[0i64])),
            None,
            0.0,
            false,
        ));
        let weak = Arc::downgrade(&pending);
        replay.append(pending, agent.gamma);
        assert!(weak.upgrade().is_some());
        drop(agent);
        assert_eq!(replay.len(), 0);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_truncated_target_uses_actual_horizon() {
        let mut agent = small_dqn(5);
        no_grad(|| {
            for mut variable in agent.model.trainable_variables() {
                let _ = variable.fill_(0.1);
            }
        });
        agent._sync_target_model();
        let obs = Tensor::ones([4], (Kind::Float, Device::Cpu));
        let _ = agent.act_and_train(&obs, 0.0);
        agent.stop_episode_and_train_with_terminal(&obs, 2.0, false);
        let sample = agent.replay_buffer.sample(1, false).pop().unwrap();
        assert_eq!(*sample.n_step_horizon.lock().unwrap(), Some(1));
        assert!(
            !sample
                .n_step_after_experience
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .is_episode_terminal
        );
        let future_q = agent.target_model.forward(&obs).max().double_value(&[]);
        let target = agent._compute_q_values(&vec![obs], &vec![2.0], &vec![1.0], &vec![1]);
        assert!((target.double_value(&[0]) - (2.0 + 0.5 * future_q)).abs() < 1e-6);
    }

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
        let targets = dqn._compute_q_values(&vec![next_state], &vec![-1.0], &vec![0.0], &vec![3]);

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
            assert!(n as f32 / (n + m) as f32 > 0.99);
        });
    }
}
