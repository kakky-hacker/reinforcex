use super::base_agent::{ensure_parent_dir, BaseAgent};
use crate::curiosity::Basecuriosity;
use crate::memory::{Experience, ReplayBuffer};
use crate::misc::batch_states::batch_states;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum::cumsum_rev;
use crate::misc::gradients::clip_grad_norm_f64;
use crate::models::BasePolicy;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{collections::HashMap, sync::Arc};
use crate::misc::autograd::no_grad;
use tch::{nn, Device, Kind, Tensor};
use ulid::Ulid;

const POLICY_LOG_PROB_RATIO_CLAMP_RANGE: f64 = 8.0;

pub struct PPO {
    agent_id: Ulid,
    model: Box<dyn BasePolicy>,
    optimizer: nn::Optimizer,
    experiences_by_episode: HashMap<Ulid, BoundedVecDeque<Arc<Experience>>>,
    buffer_for_share_experience: Option<Arc<ReplayBuffer>>,
    curiosity: Option<Box<dyn Basecuriosity>>,
    curiosity_reward_coef: f64,
    gamma: f64,
    lambda: f64,
    update_interval: usize,
    epoch: usize,
    minibatch_size: usize,
    policy_clip_epsilon: f64,
    value_clip_range: f64,
    value_coef: f64,
    entropy_coef: f64,
    gae_std: bool,
    target_kl: Option<f64>,
    t: usize,
    update_count: usize,
    optimizer_step_count: usize,
    last_optimizer_steps: usize,
    last_approx_kl: Option<f64>,
    last_clip_fraction: Option<f64>,
    last_explained_variance: Option<f64>,
    last_gradient_norm: Option<f64>,
    last_early_stop_kl: bool,
    last_policy_loss: Option<f64>,
    last_value_loss: Option<f64>,
    last_entropy: Option<f64>,
    last_intrinsic_reward_mean: Option<f64>,
    current_episode_id: Ulid,
    save_path: Option<String>,
    load_path: Option<String>,
}

impl PPO {
    pub fn new(
        model: Box<dyn BasePolicy>,
        optimizer: nn::Optimizer,
        gamma: f64,
        lambda: f64,
        update_interval: usize,
        epoch: usize,
        minibatch_size: usize,
        policy_clip_epsilon: f64,
        value_clip_range: f64,
        value_coef: f64,
        entropy_coef: f64,
        gae_std: bool,
        save_path: Option<String>,
        load_path: Option<String>,
    ) -> Self {
        assert!(update_interval > 0);
        assert!(epoch > 0);
        assert!(minibatch_size > 0 && minibatch_size <= update_interval);
        assert!(value_clip_range.is_finite() && value_clip_range >= 0.0);
        let mut agent = PPO {
            agent_id: Ulid::new(),
            model,
            optimizer,
            experiences_by_episode: HashMap::new(),
            buffer_for_share_experience: None,
            curiosity: None,
            curiosity_reward_coef: 0.0,
            gamma,
            lambda,
            update_interval,
            epoch,
            minibatch_size,
            policy_clip_epsilon,
            value_clip_range,
            value_coef,
            entropy_coef,
            gae_std,
            target_kl: None,
            t: 0,
            update_count: 0,
            optimizer_step_count: 0,
            last_optimizer_steps: 0,
            last_approx_kl: None,
            last_clip_fraction: None,
            last_explained_variance: None,
            last_gradient_norm: None,
            last_early_stop_kl: false,
            last_policy_loss: None,
            last_value_loss: None,
            last_entropy: None,
            last_intrinsic_reward_mean: None,
            current_episode_id: Ulid::new(),
            save_path,
            load_path,
        };
        agent.load();
        agent
    }

    /// Stops the remaining minibatches when the sampled reverse KL exceeds
    /// 1.5 times this target. This is an early-stop heuristic, not rollback or a
    /// hard bound on the KL of the already-applied optimizer step.
    pub fn with_target_kl(mut self, target_kl: Option<f64>) -> Self {
        assert!(target_kl.is_none_or(|value| value.is_finite() && value > 0.0));
        self.target_kl = target_kl;
        self
    }

    pub fn add_replay_buffer_for_share_experience(
        &mut self,
        buffer_for_share_experience: Arc<ReplayBuffer>,
    ) {
        self.buffer_for_share_experience = Some(buffer_for_share_experience);
    }

    pub fn add_curiosity<C: Basecuriosity + 'static>(
        &mut self,
        curiosity: C,
        curiosity_reward_coef: f64,
    ) {
        assert!(curiosity_reward_coef.is_finite());
        self.curiosity = Some(Box::new(curiosity));
        self.curiosity_reward_coef = curiosity_reward_coef;
    }

    fn _compute_rewards(&mut self, experiences: &[Arc<Experience>]) -> Tensor {
        let device = self.model.device();
        let extrinsic_rewards = Tensor::from_slice(
            &experiences
                .iter()
                .map(|experience| experience.reward)
                .collect::<Vec<f64>>(),
        )
        .to_kind(Kind::Float)
        .to_device(device);

        if experiences.is_empty() {
            return extrinsic_rewards;
        }

        let Some(curiosity) = self.curiosity.as_mut() else {
            return extrinsic_rewards;
        };

        let intrinsic_rewards = curiosity
            .calc_internal_reward_and_update(experiences)
            .to_kind(Kind::Float)
            .to_device(device)
            .view([-1]);
        assert_eq!(intrinsic_rewards.numel(), experiences.len());
        self.last_intrinsic_reward_mean =
            Some(intrinsic_rewards.mean(Kind::Float).double_value(&[]));

        extrinsic_rewards + self.curiosity_reward_coef * intrinsic_rewards
    }

    fn _compute_advantage_and_value_target(
        &self,
        raw_gae: Tensor,
        old_value: &Tensor,
    ) -> (Tensor, Tensor) {
        let value_target = &raw_gae + old_value;
        let advantage = if self.gae_std && raw_gae.numel() > 1 {
            (&raw_gae - raw_gae.mean(Kind::Float)) / (raw_gae.std(false) + 1e-8)
        } else {
            raw_gae
        };
        (advantage, value_target)
    }

    fn _require_finite(tensor: &Tensor, name: &str) {
        assert!(
            tensor.isfinite().all().int64_value(&[]) != 0,
            "PPO {name} must be finite"
        );
    }

    fn _minibatches(
        transitions: usize,
        minibatch_size: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<Vec<i64>> {
        let mut indices = (0..transitions as i64).collect::<Vec<_>>();
        indices.shuffle(rng);
        // A short final minibatch uses its real size. Padding with a new
        // permutation repeats some transitions and changes the epoch budget.
        indices
            .chunks(minibatch_size)
            .map(<[i64]>::to_vec)
            .collect()
    }

    fn _value_loss(value: &Tensor, old_value: &Tensor, target: &Tensor, clip: f64) -> Tensor {
        let error = (target - value).square();
        if clip == 0.0 {
            // Zero explicitly disables value clipping instead of fixing the
            // clipped prediction at old_value and flattening useful gradients.
            error.mean(Kind::Float)
        } else {
            let clipped = old_value + (value - old_value).clamp(-clip, clip);
            error
                .maximum(&(target - clipped).square())
                .mean(Kind::Float)
        }
    }

    fn _policy_diagnostics(log_ratio: &Tensor, clip: f64) -> (f64, f64) {
        no_grad(|| {
            // Do not use the loss's numerical log-ratio clamp to measure KL.
            // Double precision also avoids cancellation near ratio == 1.
            let log_ratio = log_ratio.detach().to_kind(Kind::Double);
            let delta_ratio = log_ratio.expm1();
            let kl = (&delta_ratio - &log_ratio)
                .mean(Kind::Double)
                .double_value(&[]);
            let clipped = delta_ratio
                .abs()
                .gt(clip)
                .to_kind(Kind::Double)
                .mean(Kind::Double)
                .double_value(&[]);
            assert!(
                kl.is_finite() && clipped.is_finite(),
                "PPO policy diagnostics must be finite"
            );
            (kl.max(0.0), clipped)
        })
    }

    fn _explained_variance(value: &Tensor, target: &Tensor) -> Option<f64> {
        no_grad(|| {
            let target = target.to_kind(Kind::Double);
            let value = value.to_kind(Kind::Double);
            let variance = target.var(false).double_value(&[]);
            if variance <= 0.0 {
                // Undefined for a constant target: omit rather than emit NaN.
                None
            } else {
                let explained = 1.0 - (&target - value).var(false).double_value(&[]) / variance;
                assert!(
                    explained.is_finite(),
                    "PPO explained variance must be finite"
                );
                Some(explained)
            }
        })
    }

    fn _backward_and_step(&mut self, loss: &Tensor) -> f64 {
        Self::_require_finite(loss, "loss");
        self.optimizer.zero_grad();
        loss.backward();
        let norm = clip_grad_norm_f64(&self.optimizer, 0.5, "PPO");
        // Reject both nonfinite losses and nonfinite gradients before touching
        // parameters or Adam state; finite f32 norms can overflow if squared in f32.
        self.optimizer.step();
        self.optimizer_step_count += 1;
        norm
    }

    fn _update(&mut self) {
        let experiences_per_episode = self
            .experiences_by_episode
            .drain()
            .map(|(_episode_id, experiences)| experiences.to_vec())
            .collect::<Vec<Vec<Arc<Experience>>>>();

        // The last action in an unfinished episode has no successor yet. Keep
        // it so the next rollout includes its reward and next observation.
        for experiences in &experiences_per_episode {
            if let Some(last) = experiences.last().filter(|e| e.action.is_some()) {
                let mut retained = BoundedVecDeque::new(1e9 as usize);
                retained.push_back(Arc::clone(last));
                self.experiences_by_episode
                    .insert(last.episode_id, retained);
            }
        }

        let total_transitions = experiences_per_episode
            .iter()
            .map(|v| v.len().saturating_sub(1))
            .sum::<usize>();
        if total_transitions == 0 {
            return;
        }
        let mut rng = thread_rng();

        // Create data.
        let _skip_first = experiences_per_episode
            .iter()
            .flat_map(|v| v.iter().skip(1))
            .cloned()
            .collect::<Vec<Arc<Experience>>>();
        let _skip_last = experiences_per_episode
            .iter()
            .flat_map(|v| v.iter().take(v.len().saturating_sub(1)))
            .cloned()
            .collect::<Vec<Arc<Experience>>>();
        let state = batch_states(
            &_skip_last
                .iter()
                .map(|e| e.state.shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let next_state = batch_states(
            &_skip_first
                .iter()
                .map(|e| e.state.shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let _action = batch_states(
            &_skip_last
                .iter()
                .map(|e| e.action.as_ref().unwrap().shallow_clone())
                .collect::<Vec<Tensor>>(),
            self.model.device(),
        );
        let action = _action.view([total_transitions as i64, *_action.size().last().unwrap()]);
        let reward = self._compute_rewards(&_skip_first);

        let (_, old_value) = no_grad(|| self.model.forward(&state));
        let old_value = old_value.unwrap().flatten(0, 1);
        // A retained boundary action may have been sampled before the previous
        // update, so its behavior likelihood cannot be recomputed now.
        let old_log_prob = Self::_behavior_log_probs(&_skip_last, self.model.device());

        let (_, old_next_value) = no_grad(|| self.model.forward(&next_state));
        let old_next_value = old_next_value.unwrap().flatten(0, 1);
        Self::_require_finite(&reward, "reward");
        Self::_require_finite(&old_value, "old value");
        Self::_require_finite(&old_next_value, "old next value");
        Self::_require_finite(&old_log_prob, "behavior log probability");

        let non_terminal: Tensor = 1.0
            - Tensor::from_slice(
                &_skip_first
                    .iter()
                    .map(|e| if e.is_episode_terminal { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            )
            .to_kind(Kind::Float)
            .to_device(self.model.device());

        let old_next_value = (old_next_value * non_terminal).detach();

        // Compute GAE
        let td_error = (reward + self.gamma * &old_next_value - &old_value).to_device(Device::Cpu);
        let _gae = Tensor::from_slice(&cumsum_rev(
            &(0..td_error.size()[0])
                .map(|i| td_error.double_value(&[i]))
                .collect::<Vec<f64>>(),
            &Self::_gae_discounts(&experiences_per_episode, self.gamma * self.lambda),
        ))
        .to_kind(Kind::Float)
        .to_device(self.model.device())
        .detach();
        let (gae, value_target) = self._compute_advantage_and_value_target(_gae, &old_value);
        Self::_require_finite(&gae, "advantage");
        Self::_require_finite(&value_target, "value target");
        self.last_explained_variance = Self::_explained_variance(&old_value, &value_target);
        self.last_early_stop_kl = false;
        self.last_gradient_norm = None;
        let optimizer_steps_before = self.optimizer_step_count;
        let mut diagnostic_samples = 0usize;
        let mut weighted_kl = 0.0;
        let mut weighted_clip_fraction = 0.0;

        'epochs: for _ in 0..self.epoch {
            for indices in Self::_minibatches(total_transitions, self.minibatch_size, &mut rng) {
                let minibatch_indice = Tensor::from_slice(&indices).to_device(self.model.device());

                let minibatch_state = state.index_select(0, &minibatch_indice);
                let minibatch_action = action.index_select(0, &minibatch_indice);

                // Forward only the current minibatch.
                let (action_distrib, value) = self.model.forward(&minibatch_state);
                let value = value.unwrap().flatten(0, 1);

                // Compute policy ratio.
                let log_prob = action_distrib.log_prob(&minibatch_action.detach());
                let minibatch_old_log_prob = old_log_prob.index_select(0, &minibatch_indice);
                Self::_require_finite(&log_prob, "log probability");
                Self::_require_finite(&value, "value prediction");
                let log_ratio = log_prob - minibatch_old_log_prob;
                let (approx_kl, clip_fraction) =
                    Self::_policy_diagnostics(&log_ratio, self.policy_clip_epsilon);
                diagnostic_samples += indices.len();
                weighted_kl += approx_kl * indices.len() as f64;
                weighted_clip_fraction += clip_fraction * indices.len() as f64;
                if self
                    .target_kl
                    .is_some_and(|target| approx_kl > 1.5 * target)
                {
                    self.last_early_stop_kl = true;
                    break 'epochs;
                }
                let policy_ratio = log_ratio
                    .clamp(
                        -POLICY_LOG_PROB_RATIO_CLAMP_RANGE,
                        POLICY_LOG_PROB_RATIO_CLAMP_RANGE,
                    )
                    .exp();
                let clipped_policy_ratio = policy_ratio.clamp(
                    1.0 - self.policy_clip_epsilon,
                    1.0 + self.policy_clip_epsilon,
                );

                let _old_value = old_value.index_select(0, &minibatch_indice).detach();

                let minibatch_gae = gae.index_select(0, &minibatch_indice).detach();

                // Compute loss
                let _value_target = (&value_target).index_select(0, &minibatch_indice).detach();
                let policy_loss: Tensor = -1.0
                    * (policy_ratio * &minibatch_gae)
                        .minimum(&(clipped_policy_ratio * &minibatch_gae))
                        .mean(Kind::Float);
                let value_loss =
                    Self::_value_loss(&value, &_old_value, &_value_target, self.value_clip_range);
                let entropy_regularized = action_distrib.entropy().mean(Kind::Float);

                Self::_require_finite(&policy_loss, "policy loss");
                Self::_require_finite(&value_loss, "value loss");
                Self::_require_finite(&entropy_regularized, "entropy");

                self.last_policy_loss = Some(policy_loss.double_value(&[]));
                self.last_value_loss = Some(value_loss.double_value(&[]));
                self.last_entropy = Some(entropy_regularized.double_value(&[]));

                let loss: Tensor = policy_loss + self.value_coef * value_loss
                    - self.entropy_coef * entropy_regularized;

                self.last_gradient_norm = Some(self._backward_and_step(&loss));
            }
        }
        self.last_optimizer_steps = self.optimizer_step_count - optimizer_steps_before;
        // Sample-weighted over inspected minibatches, including the minibatch
        // that triggers KL stopping. These are not post-update full-rollout KL.
        self.last_approx_kl = Some(weighted_kl / diagnostic_samples as f64);
        self.last_clip_fraction = Some(weighted_clip_fraction / diagnostic_samples as f64);
        self.update_count += 1;
    }

    fn _update_if_ready(&mut self) {
        // Count completed transitions, not calls to act. Terminal observations
        // complete a transition too, while a freshly sampled action does not.
        let completed = self
            .experiences_by_episode
            .values()
            .map(|episode| episode.len().saturating_sub(1))
            .sum::<usize>();
        if completed >= self.update_interval {
            self._update();
        }
    }

    fn _act_and_train_with_replay_input(
        &mut self,
        obs: &Tensor,
        reward: f64,
        replay_obs: &Tensor,
        replay_reward: f64,
    ) -> Tensor {
        self.t += 1;

        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let action_distrib = no_grad(|| {
            let (action_distrib, _) = self.model.forward(&state);
            action_distrib
        });
        // PPO likelihoods are defined on the original policy-space action.
        // Only the environment and off-policy replay receive the clipped action.
        let policy_action = action_distrib.sample().detach();
        let action = action_distrib
            .to_env_action(&policy_action)
            .detach()
            .to_device(Device::Cpu);

        let experience = Arc::new(Experience::new(
            self.agent_id,
            self.current_episode_id,
            state,
            Some(policy_action.to_device(Device::Cpu)),
            Some(action_distrib),
            reward,
            false,
        ));
        self.experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        if let Some(buffer_for_share_experience) = &self.buffer_for_share_experience {
            // The off-policy consumer only needs the transition itself.  Keeping
            // PPO's CUDA distribution tensors in a long-lived replay buffer can
            // otherwise exhaust device memory during shared training.
            let replay_experience = Arc::new(Experience::new(
                self.agent_id,
                self.current_episode_id,
                batch_states(&vec![replay_obs.shallow_clone()], Device::Cpu),
                Some(action.shallow_clone()),
                None,
                replay_reward,
                false,
            ));
            buffer_for_share_experience.append(replay_experience, self.gamma);
        }

        self._update_if_ready();

        action
    }

    fn _stop_episode_and_train_with_replay_input(
        &mut self,
        obs: &Tensor,
        reward: f64,
        terminated: bool,
        replay_obs: &Tensor,
        replay_reward: f64,
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
        self.experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        if let Some(buffer_for_share_experience) = &self.buffer_for_share_experience {
            let replay_experience = Arc::new(
                Experience::new(
                    self.agent_id,
                    self.current_episode_id,
                    batch_states(&vec![replay_obs.shallow_clone()], Device::Cpu),
                    None,
                    None,
                    replay_reward,
                    terminated,
                )
                .with_episode_end(true),
            );
            buffer_for_share_experience.append(replay_experience, self.gamma);
        }
        self.current_episode_id = Ulid::new();
        self._update_if_ready();
    }

    fn _behavior_log_probs(experiences: &[Arc<Experience>], device: Device) -> Tensor {
        let log_probs = experiences
            .iter()
            .map(|experience| {
                let distribution = experience.action_distrib.as_ref().unwrap();
                let action = experience
                    .action
                    .as_ref()
                    .unwrap()
                    .to_device(device)
                    .view([1, -1]);
                distribution.log_prob(&action).view([-1]).detach()
            })
            .collect::<Vec<_>>();
        Tensor::cat(&log_probs, 0).to_device(device)
    }

    fn _gae_discounts(episodes: &[Vec<Arc<Experience>>], discount: f64) -> Vec<f64> {
        episodes
            .iter()
            .flat_map(|episode| {
                episode
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(move |(index, experience)| {
                        // Every rollout/episode boundary ends the GAE recurrence,
                        // even when a time limit still permits value bootstrapping.
                        if experience.is_episode_terminal || index + 1 == episode.len() {
                            0.0
                        } else {
                            discount
                        }
                    })
            })
            .collect()
    }
}

impl BaseAgent for PPO {
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
            let (action_distrib, _) = self.model.forward(&state);
            let action = action_distrib
                .to_env_action(&action_distrib.most_probable())
                .to_device(Device::Cpu);
            action
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self._act_and_train_with_replay_input(obs, reward, obs, reward)
    }

    fn supports_separate_replay_input(&self) -> bool {
        true
    }

    fn act_and_train_with_replay_input(
        &mut self,
        obs: &Tensor,
        reward: f64,
        replay_obs: &Tensor,
        replay_reward: f64,
    ) -> Tensor {
        self._act_and_train_with_replay_input(obs, reward, replay_obs, replay_reward)
    }

    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64) {
        self._stop_episode_and_train_with_replay_input(obs, reward, true, obs, reward);
    }

    fn stop_episode_and_train_with_terminal(
        &mut self,
        obs: &Tensor,
        reward: f64,
        terminated: bool,
    ) {
        self._stop_episode_and_train_with_replay_input(obs, reward, terminated, obs, reward);
    }

    fn stop_episode_and_train_with_replay_input(
        &mut self,
        obs: &Tensor,
        reward: f64,
        terminated: bool,
        replay_obs: &Tensor,
        replay_reward: f64,
    ) {
        self._stop_episode_and_train_with_replay_input(
            obs,
            reward,
            terminated,
            replay_obs,
            replay_reward,
        );
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        let mut statistics = vec![
            ("updates".to_string(), self.update_count as f64),
            (
                "optimizer_steps".to_string(),
                self.optimizer_step_count as f64,
            ),
        ];
        if self.update_count > 0 {
            statistics.push((
                "optimizer_steps_last_update".to_string(),
                self.last_optimizer_steps as f64,
            ));
            statistics.push((
                "early_stop_kl".to_string(),
                if self.last_early_stop_kl { 1.0 } else { 0.0 },
            ));
        }
        for (name, value) in [
            ("approx_kl", self.last_approx_kl),
            ("clip_fraction", self.last_clip_fraction),
            ("explained_variance", self.last_explained_variance),
            ("gradient_norm", self.last_gradient_norm),
        ] {
            if let Some(value) = value {
                statistics.push((name.to_string(), value));
            }
        }
        if let Some(loss) = self.last_policy_loss {
            statistics.push(("policy_loss".to_string(), loss));
        }
        if let Some(loss) = self.last_value_loss {
            statistics.push(("value_loss".to_string(), loss));
        }
        if let Some(entropy) = self.last_entropy {
            statistics.push(("entropy".to_string(), entropy));
        }
        if let Some(intrinsic_reward_mean) = self.last_intrinsic_reward_mean {
            statistics.push(("intrinsic_reward_mean".to_string(), intrinsic_reward_mean));
            statistics.push((
                "curiosity_coefficient".to_string(),
                self.curiosity_reward_coef,
            ));
        }
        statistics
    }

    fn get_agent_id(&self) -> &Ulid {
        &self.agent_id
    }

    fn save(&self) {
        if let Some(path) = &self.save_path {
            if !path.is_empty() {
                ensure_parent_dir(path);
                self.model.save(path);
            }
        }
        if let Some(curiosity) = &self.curiosity {
            curiosity.save();
        }
    }

    fn load(&mut self) {
        if let Some(path) = self.load_path.clone() {
            if path.is_empty() {
                return;
            }
            self.model.load(&path);
        }
    }
}

impl Drop for PPO {
    fn drop(&mut self) {
        if let Some(replay) = &self.buffer_for_share_experience {
            replay.discard_episode(&self.current_episode_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ReplayBuffer;
    use crate::models::FCSoftmaxPolicyWithValue;
    use crate::prob_distributions::{BaseDistribution, GaussianDistribution, SoftmaxDistribution};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    struct ConstantCuriosity {
        reward: f64,
        calc_count: Arc<AtomicUsize>,
        update_count: Arc<AtomicUsize>,
        updated_experience_count: Arc<AtomicUsize>,
    }

    impl Basecuriosity for ConstantCuriosity {
        fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor {
            self.calc_count.fetch_add(1, Ordering::Relaxed);
            Tensor::full(
                [experiences.len() as i64],
                self.reward,
                (Kind::Double, Device::Cpu),
            )
        }

        fn update(&mut self, experiences: &[Arc<Experience>]) {
            self.update_count.fetch_add(1, Ordering::Relaxed);
            self.updated_experience_count
                .fetch_add(experiences.len(), Ordering::Relaxed);
        }

        fn save(&self) {}

        fn load(&mut self) {}
    }

    struct BoundaryGaussianPolicy;

    struct ConstantValuePolicy {
        value: Tensor,
        logits: Tensor,
    }

    impl BasePolicy for ConstantValuePolicy {
        fn forward(&self, x: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
            let batch = x.size()[0];
            (
                Box::new(SoftmaxDistribution::new(
                    self.logits.expand([batch, 2], true),
                    1.0,
                    0.0,
                )),
                Some(self.value.expand([batch, 1], true)),
            )
        }
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn save(&self, _: &str) {}
        fn load(&mut self, _: &str) {}
    }

    fn constant_value_ppo() -> PPO {
        let vs = nn::VarStore::new(Device::Cpu);
        let value = vs.root().var("value", &[1, 1], nn::Init::Const(2.0));
        let logits = vs.root().var("logits", &[1, 2], nn::Init::Const(0.0));
        let optimizer = nn::Sgd::default().build(&vs, 0.0).unwrap();
        PPO::new(
            Box::new(ConstantValuePolicy { value, logits }),
            optimizer,
            0.5,
            0.95,
            1,
            1,
            1,
            0.2,
            0.2,
            1.0,
            0.0,
            false,
            None,
            None,
        )
    }

    fn optimizer_fixture(initial_value: f64) -> (PPO, Tensor) {
        let vs = nn::VarStore::new(Device::Cpu);
        let value = vs
            .root()
            .var("value", &[1, 1], nn::Init::Const(initial_value));
        let logits = vs.root().var("logits", &[1, 2], nn::Init::Const(0.0));
        let optimizer = nn::Sgd::default().build(&vs, 0.1).unwrap();
        let model = ConstantValuePolicy {
            value: value.shallow_clone(),
            logits,
        };
        (
            PPO::new(
                Box::new(model),
                optimizer,
                0.0,
                0.0,
                4,
                3,
                2,
                0.2,
                0.0,
                0.0,
                0.0,
                false,
                None,
                None,
            ),
            value,
        )
    }

    #[test]
    fn test_learning_rate_change_preserves_adam_moments_and_scales_next_step() {
        fn fixture() -> (PPO, Tensor) {
            let vs = nn::VarStore::new(Device::Cpu);
            // Match the FFI: optimizer first, model parameters second.
            let optimizer = nn::Adam::default().build(&vs, 0.01).unwrap();
            let value = vs.root().var("value", &[1, 1], nn::Init::Const(1.0));
            let logits = vs.root().var("logits", &[1, 2], nn::Init::Const(0.0));
            let model = ConstantValuePolicy {
                value: value.shallow_clone(),
                logits,
            };
            (
                PPO::new(
                    Box::new(model),
                    optimizer,
                    0.0,
                    0.0,
                    4,
                    3,
                    2,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    false,
                    None,
                    None,
                ),
                value,
            )
        }
        let (mut initial_rate, initial) = fixture();
        initial_rate.set_learning_rate(0.005);
        initial_rate._backward_and_step(&(&initial * 0.25).sum(Kind::Float));
        assert!((initial.double_value(&[0, 0]) - 0.995).abs() < 1e-6);
        let (mut unchanged, full) = fixture();
        let (mut changed, half) = fixture();
        let (mut same_rate, same) = fixture();
        for (agent, value) in [
            (&mut unchanged, &full),
            (&mut changed, &half),
            (&mut same_rate, &same),
        ] {
            assert!(agent.supports_learning_rate_update());
            agent._backward_and_step(&(value * 0.25).sum(Kind::Float));
        }
        assert!(full.equal(&half) && full.equal(&same));
        let before = half.copy();
        let statistics = changed.get_statistics();
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                changed.set_learning_rate(invalid);
            }))
            .is_err());
        }
        changed.set_learning_rate(0.005);
        same_rate.set_learning_rate(0.01);
        assert!(half.equal(&before));
        assert_eq!(changed.optimizer_step_count, 1);
        assert_eq!(changed.get_statistics(), statistics);
        let starting_value = full.double_value(&[0, 0]);
        for (agent, value) in [
            (&mut unchanged, &full),
            (&mut changed, &half),
            (&mut same_rate, &same),
        ] {
            // A reset Adam would follow this gradient in the opposite direction.
            agent._backward_and_step(&(value * -0.01).sum(Kind::Float));
            assert_eq!(agent.optimizer_step_count, 2);
        }
        let full_delta = starting_value - full.double_value(&[0, 0]);
        let half_delta = starting_value - half.double_value(&[0, 0]);
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
    fn test_each_epoch_visits_every_transition_once_without_padding() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..3 {
            let batches = PPO::_minibatches(9, 4, &mut rng);
            assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [4, 4, 1]);
            let mut visited = batches.into_iter().flatten().collect::<Vec<_>>();
            visited.sort_unstable();
            assert_eq!(visited, (0_i64..9).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_zero_value_clip_disables_clipping_and_preserves_improvement_gradient() {
        let value = Tensor::from_slice(&[0.5_f32]).set_requires_grad(true);
        let old = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let target = Tensor::ones([1], (Kind::Float, Device::Cpu));
        let loss = PPO::_value_loss(&value, &old, &target, 0.0);
        assert_eq!(loss.double_value(&[]), 0.25);
        loss.backward();
        assert_eq!(value.grad().double_value(&[0]), -1.0);

        let clipped_value = Tensor::from_slice(&[0.5_f32]).set_requires_grad(true);
        let clipped_loss = PPO::_value_loss(&clipped_value, &old, &target, 0.2);
        assert!((clipped_loss.double_value(&[]) - 0.64).abs() < 1e-6);
        clipped_loss.backward();
        assert_eq!(clipped_value.grad().double_value(&[0]), 0.0);
    }

    #[test]
    fn test_nonfinite_loss_and_gradient_are_rejected_before_optimizer_step() {
        for infinite_gradient in [false, true] {
            let (mut agent, value) = optimizer_fixture(if infinite_gradient { 0.0 } else { 1.0 });
            let before = value.copy();
            let loss = if infinite_gradient {
                value.sqrt().sum(Kind::Float)
            } else {
                (&value * f64::INFINITY).sum(Kind::Float)
            };
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                agent._backward_and_step(&loss);
            }))
            .is_err());
            assert!(value.equal(&before));
            assert_eq!(agent.optimizer_step_count, 0);
        }
        // A caught error in one agent must not disable another agent's
        // autograd on the same caller thread.
        let (mut healthy, value) = optimizer_fixture(1.0);
        let loss = value.square().sum(Kind::Float);
        assert!(loss.requires_grad());
        healthy._backward_and_step(&loss);
        assert_eq!(healthy.optimizer_step_count, 1);
        assert!(value.double_value(&[0, 0]) < 1.0);
    }

    #[test]
    fn test_large_finite_gradients_are_clipped_without_f32_norm_overflow() {
        let (mut agent, value) = optimizer_fixture(1.0);
        let norm = agent._backward_and_step(&(&value * 1e30).sum(Kind::Float));
        assert!(norm.is_finite() && norm > 1e29);
        assert!((value.double_value(&[0, 0]) - 0.95).abs() < 1e-6);
        assert_eq!(agent.optimizer_step_count, 1);
    }

    #[test]
    fn test_policy_diagnostics_and_explained_variance_have_known_values() {
        let ratios = Tensor::from_slice(&[0.0_f64, 2.0_f64.ln(), 0.5_f64.ln()]);
        let (kl, fraction) = PPO::_policy_diagnostics(&ratios, 0.2);
        assert!((kl - 1.0 / 6.0).abs() < 1e-12);
        assert!((fraction - 2.0 / 3.0).abs() < 1e-12);
        let (large_kl, _) = PPO::_policy_diagnostics(&Tensor::from_slice(&[9.0_f64]), 0.2);
        assert!((large_kl - (9.0_f64.exp() - 10.0)).abs() < 1e-9);
        let target = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]);
        assert_eq!(PPO::_explained_variance(&target, &target), Some(1.0));
        assert_eq!(
            PPO::_explained_variance(&Tensor::from_slice(&[3.0_f32, 2.0, 1.0]), &target),
            Some(-3.0)
        );
        assert_eq!(
            PPO::_explained_variance(&target, &Tensor::ones([3], (Kind::Float, Device::Cpu))),
            None
        );
    }

    #[test]
    fn test_target_kl_rejects_invalid_values() {
        for target in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| constant_value_ppo().with_target_kl(Some(target)))
                    .is_err()
            );
        }
        assert_eq!(constant_value_ppo().with_target_kl(None).target_kl, None);
        assert_eq!(
            constant_value_ppo().with_target_kl(Some(0.02)).target_kl,
            Some(0.02)
        );
    }

    fn policy_change_fixture(target_kl: Option<f64>) -> PPO {
        let (mut agent, _) = optimizer_fixture(0.0);
        agent.optimizer.set_lr(4.0);
        agent.epoch = 10;
        agent.minibatch_size = 4;
        agent = agent.with_target_kl(target_kl);
        // Four identical, fixed one-step transitions make the first SGD policy
        // change deterministic; no sampled actions or global seeds are needed.
        for _ in 0..4 {
            let episode = Ulid::new();
            let mut experiences = BoundedVecDeque::new(2);
            experiences.push_back(Arc::new(Experience::new(
                agent.agent_id,
                episode,
                Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)),
                Some(Tensor::zeros([1], (Kind::Int64, Device::Cpu))),
                Some(Box::new(SoftmaxDistribution::new(
                    Tensor::zeros([1, 2], (Kind::Float, Device::Cpu)),
                    1.0,
                    0.0,
                ))),
                0.0,
                false,
            )));
            experiences.push_back(Arc::new(Experience::new(
                agent.agent_id,
                episode,
                Tensor::zeros([1, 1], (Kind::Float, Device::Cpu)),
                None,
                None,
                1.0,
                true,
            )));
            agent.experiences_by_episode.insert(episode, experiences);
        }
        agent
    }

    #[test]
    fn test_kl_stopping_counts_actual_optimizer_steps_without_stopping_zero_kl() {
        let mut stopped = policy_change_fixture(Some(0.01));
        stopped._update();
        assert_eq!(stopped.optimizer_step_count, 1);
        assert_eq!(stopped.last_optimizer_steps, 1);
        assert_eq!(stopped.update_count, 1);
        assert!(stopped.last_early_stop_kl);
        assert!(stopped.last_approx_kl.unwrap() > 0.01);

        let mut disabled = policy_change_fixture(None);
        disabled._update();
        assert_eq!(disabled.optimizer_step_count, 10);
        assert!(!disabled.last_early_stop_kl);

        let mut stationary = policy_change_fixture(Some(0.01));
        stationary.optimizer.set_lr(0.0);
        stationary._update();
        assert_eq!(stationary.optimizer_step_count, 10);
        assert_eq!(stationary.last_approx_kl, Some(0.0));
        assert!(!stationary.last_early_stop_kl);
        assert!(stationary
            .get_statistics()
            .iter()
            .all(|(_, value)| value.is_finite()));
    }

    #[test]
    fn test_truncation_bootstraps_but_termination_does_not() {
        let obs = Tensor::ones([1], (Kind::Float, Device::Cpu));
        let mut terminated = constant_value_ppo();
        let _ = terminated.act_and_train(&obs, 0.0);
        terminated.stop_episode_and_train_with_terminal(&obs, 1.0, true);
        assert_eq!(terminated.update_count, 1);
        assert_eq!(terminated.last_value_loss, Some(1.0));

        let mut truncated = constant_value_ppo();
        let _ = truncated.act_and_train(&obs, 0.0);
        truncated.stop_episode_and_train_with_terminal(&obs, 1.0, false);
        assert_eq!(truncated.update_count, 1);
        assert_eq!(truncated.last_value_loss, Some(0.0));
        assert!(truncated.experiences_by_episode.is_empty());
    }

    #[test]
    fn test_gae_never_crosses_unfinished_or_truncated_episode_boundaries() {
        let make_episode = || {
            let episode_id = Ulid::new();
            (0..3)
                .map(|_| {
                    Arc::new(Experience::new(
                        Ulid::new(),
                        episode_id,
                        Tensor::zeros([1], (Kind::Float, Device::Cpu)),
                        None,
                        None,
                        0.0,
                        false,
                    ))
                })
                .collect::<Vec<_>>()
        };
        let episodes = vec![make_episode(), make_episode()];
        let discounts = PPO::_gae_discounts(&episodes, 0.5);
        let advantages = cumsum_rev(&[1.0, 2.0, 100.0, 200.0], &discounts);
        assert_eq!(advantages, [2.0, 2.0, 200.0, 200.0]);
    }

    #[test]
    fn test_behavior_likelihood_comes_from_action_time_distribution() {
        let distribution =
            SoftmaxDistribution::new(Tensor::from_slice(&[0.0_f32, 9.0]).view([1, 2]), 1.0, 0.0);
        let action = Tensor::from_slice(&[0_i64]);
        let expected = distribution.log_prob(&action.view([1, 1]));
        let experience = Arc::new(Experience::new(
            Ulid::new(),
            Ulid::new(),
            Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            Some(action),
            Some(Box::new(distribution)),
            0.0,
            false,
        ));
        let actual = PPO::_behavior_log_probs(&[experience], Device::Cpu);
        assert!(actual.equal(&expected));
        assert!(actual.double_value(&[0]) < -9.0);
        assert!(!actual.requires_grad());
    }

    #[test]
    fn test_update_interval_one_preserves_all_transitions_and_terminal_reward() {
        let updated_experience_count = Arc::new(AtomicUsize::new(0));
        let mut ppo = test_ppo_with_update_interval(1);
        ppo.add_curiosity(
            ConstantCuriosity {
                reward: 1.0,
                calc_count: Arc::new(AtomicUsize::new(0)),
                update_count: Arc::new(AtomicUsize::new(0)),
                updated_experience_count: Arc::clone(&updated_experience_count),
            },
            0.1,
        );
        let obs = Tensor::ones([4], (Kind::Float, Device::Cpu));
        for step in 0..5 {
            let _ = ppo.act_and_train(&obs, step as f64);
        }
        assert_eq!(updated_experience_count.load(Ordering::Relaxed), 4);
        ppo.stop_episode_and_train(&obs, 42.0);
        assert_eq!(updated_experience_count.load(Ordering::Relaxed), 5);
        assert_eq!(ppo.update_count, 5);
        assert!(ppo.experiences_by_episode.is_empty());
    }

    #[test]
    fn test_curiosity_uses_terminal_next_observation_before_predictor_update() {
        struct StateCuriosity {
            trained_states: Arc<std::sync::Mutex<Vec<f64>>>,
        }
        impl Basecuriosity for StateCuriosity {
            fn calc_internal_reward(&self, experiences: &[Arc<Experience>]) -> Tensor {
                assert!(self.trained_states.lock().unwrap().is_empty());
                Tensor::from_slice(
                    &experiences
                        .iter()
                        .map(|e| e.state.flatten(0, -1).double_value(&[0]) as f32)
                        .collect::<Vec<_>>(),
                )
            }
            fn update(&mut self, experiences: &[Arc<Experience>]) {
                self.trained_states.lock().unwrap().extend(
                    experiences
                        .iter()
                        .map(|e| e.state.flatten(0, -1).double_value(&[0])),
                );
            }
            fn save(&self) {}
            fn load(&mut self) {}
        }
        let trained_states = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut ppo = constant_value_ppo();
        ppo.add_curiosity(
            StateCuriosity {
                trained_states: Arc::clone(&trained_states),
            },
            1.0,
        );
        let _ = ppo.act_and_train(&Tensor::from_slice(&[1.0_f32]), 0.0);
        ppo.stop_episode_and_train(&Tensor::from_slice(&[3.0_f32]), 1.0);
        assert_eq!(ppo.last_intrinsic_reward_mean, Some(3.0));
        assert_eq!(ppo.last_value_loss, Some(4.0));
        assert_eq!(*trained_states.lock().unwrap(), [3.0]);
    }

    impl BasePolicy for BoundaryGaussianPolicy {
        fn forward(&self, _: &Tensor) -> (Box<dyn BaseDistribution>, Option<Tensor>) {
            // Keep samples far outside both bounds so this exercises clipping,
            // independent of the random seed used by other tests.
            let mean = Tensor::from_slice(&[5.0_f32, -5.0]).view([1, 2]);
            let var = Tensor::full([1, 2], 1e-12, (Kind::Float, Device::Cpu));
            (
                Box::new(GaussianDistribution::new_bounded(mean, var, -1.0, 1.0)),
                None,
            )
        }

        fn device(&self) -> Device {
            Device::Cpu
        }

        fn save(&self, _: &str) {}

        fn load(&mut self, _: &str) {}
    }

    #[test]
    fn test_ppo_new() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 2, 2, 64, 0.0);

        let ppo = PPO::new(
            Box::new(model),
            optimizer,
            0.99,
            0.99,
            100,
            8,
            16,
            0.1,
            0.2,
            1.0,
            1.0,
            false,
            None,
            None,
        );

        assert_eq!(ppo.update_interval, 100);
        assert_eq!(ppo.epoch, 8);
        assert_eq!(ppo.gamma, 0.99);
        assert_eq!(ppo.t, 0);
        assert!(ppo.buffer_for_share_experience.is_none());
        assert!(ppo.curiosity.is_none());
    }

    #[test]
    fn test_standardized_advantage_does_not_change_value_target() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 2, 2, 1, 8, 0.0);
        let ppo = PPO::new(
            Box::new(model),
            optimizer,
            0.99,
            0.95,
            2,
            1,
            1,
            0.2,
            0.2,
            0.5,
            0.01,
            true,
            None,
            None,
        );
        let raw_gae = Tensor::from_slice(&[2.0_f32, 4.0]);
        let old_value = Tensor::from_slice(&[10.0_f32, 20.0]);

        let (advantage, value_target) =
            ppo._compute_advantage_and_value_target(raw_gae, &old_value);

        assert!(advantage.mean(Kind::Float).double_value(&[]).abs() < 1e-6);
        assert!((value_target.double_value(&[0]) - 12.0).abs() < 1e-6);
        assert!((value_target.double_value(&[1]) - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_transition_keeps_its_policy_learning_signal() {
        let mut ppo = constant_value_ppo();
        ppo.gae_std = true;
        let (advantage, value_target) = ppo._compute_advantage_and_value_target(
            Tensor::from_slice(&[3.0_f32]),
            &Tensor::from_slice(&[2.0_f32]),
        );
        assert_eq!(advantage.double_value(&[0]), 3.0);
        assert_eq!(value_target.double_value(&[0]), 5.0);
    }

    #[test]
    fn test_separate_replay_inputs_preserve_raw_n_step_transitions_and_boundaries() {
        for terminated in [false, true] {
            for n_steps in [1, 3] {
                let vs = nn::VarStore::new(Device::Cpu);
                let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
                let replay = Arc::new(ReplayBuffer::new(100, n_steps));
                let mut ppo = PPO::new(
                    Box::new(BoundaryGaussianPolicy),
                    optimizer,
                    0.5,
                    0.95,
                    100,
                    1,
                    1,
                    0.2,
                    0.2,
                    0.5,
                    0.01,
                    true,
                    None,
                    None,
                );
                ppo.add_replay_buffer_for_share_experience(replay.clone());
                assert!(ppo.supports_separate_replay_input());
                let episode_id = ppo.current_episode_id;
                let raw = |value| Tensor::full([4], value, (Kind::Float, Device::Cpu));
                let first =
                    ppo.act_and_train_with_replay_input(&(raw(1.0) * 10.0), 0.0, &raw(1.0), 0.0);
                let second =
                    ppo.act_and_train_with_replay_input(&(raw(2.0) * 10.0), 0.2, &raw(2.0), 2.0);
                ppo.stop_episode_and_train_with_replay_input(
                    &(raw(3.0) * 10.0),
                    0.4,
                    terminated,
                    &raw(3.0),
                    4.0,
                );
                assert_ne!(ppo.current_episode_id, episode_id);
                assert_eq!(ppo.t, 2);
                assert_eq!(ppo.update_count, 0);
                let learner = ppo.experiences_by_episode[&episode_id].to_vec();
                assert_eq!(learner.len(), 3);
                for (index, experience) in learner.iter().enumerate() {
                    assert!(experience
                        .state
                        .equal(&raw((index + 1) as f64 * 10.0).unsqueeze(0)));
                    assert_eq!(experience.reward, [0.0, 0.2, 0.4][index]);
                    assert_eq!(experience.is_episode_end, index == 2);
                    assert_eq!(experience.is_episode_terminal, index == 2 && terminated);
                }
                assert_eq!(replay.len(), 2);
                let mut shared = replay.sample(2, false);
                shared.sort_by(|a, b| {
                    a.state
                        .double_value(&[0, 0])
                        .total_cmp(&b.state.double_value(&[0, 0]))
                });
                for (index, experience) in shared.iter().enumerate() {
                    assert_eq!(experience.agent_id, ppo.agent_id);
                    assert_eq!(experience.episode_id, episode_id);
                    assert!(experience
                        .state
                        .equal(&raw((index + 1) as f64).unsqueeze(0)));
                    assert_eq!(experience.reward, [0.0, 2.0][index]);
                    assert!(experience.action.as_ref().unwrap().equal(if index == 0 {
                        &first
                    } else {
                        &second
                    }));
                    assert!(experience.action_distrib.is_none());
                    assert!(
                        learner[index]
                            .action
                            .as_ref()
                            .unwrap()
                            .double_value(&[0, 0])
                            > 1.0
                    );
                    assert_eq!(
                        experience.action.as_ref().unwrap().double_value(&[0, 0]),
                        1.0
                    );
                    let expected_horizon = if n_steps == 1 { 1 } else { 2 - index };
                    assert_eq!(
                        *experience.n_step_horizon.lock().unwrap(),
                        Some(expected_horizon)
                    );
                    let expected_reward = if index == 0 && n_steps == 1 { 2.0 } else { 4.0 };
                    assert_eq!(
                        *experience.n_step_discounted_reward.lock().unwrap(),
                        Some(expected_reward)
                    );
                    let next = experience
                        .n_step_after_experience
                        .lock()
                        .unwrap()
                        .clone()
                        .unwrap();
                    let reaches_end = index + expected_horizon == 2;
                    assert_eq!(
                        next.state.double_value(&[0, 0]),
                        (index + expected_horizon + 1) as f64
                    );
                    assert_eq!(next.is_episode_end, reaches_end);
                    assert_eq!(next.is_episode_terminal, reaches_end && terminated);
                    assert_eq!(next.episode_id, episode_id);
                }
                // A fresh episode cannot extend a flushed time-limit/terminal tail.
                let _ = ppo.act_and_train_with_replay_input(&raw(90.0), 0.0, &raw(9.0), 0.0);
                ppo.stop_episode_and_train_with_replay_input(
                    &raw(100.0),
                    2.0,
                    true,
                    &raw(10.0),
                    20.0,
                );
                assert_eq!(replay.len(), 3);
                let last = replay
                    .sample(3, false)
                    .into_iter()
                    .find(|e| e.episode_id != episode_id)
                    .unwrap();
                assert_eq!(*last.n_step_discounted_reward.lock().unwrap(), Some(20.0));
                assert_eq!(*last.n_step_horizon.lock().unwrap(), Some(1));
            }
        }
    }

    #[test]
    fn test_separate_replay_reward_never_enters_learner_bootstrap_target() {
        for terminated in [false, true] {
            let mut ppo = constant_value_ppo();
            let replay = Arc::new(ReplayBuffer::new(100, 1));
            ppo.add_replay_buffer_for_share_experience(replay.clone());
            let raw = Tensor::ones([4], (Kind::Float, Device::Cpu));
            let _ = ppo.act_and_train_with_replay_input(&(&raw * 10.0), 0.0, &raw, 0.0);
            ppo.stop_episode_and_train_with_replay_input(
                &(&raw * 20.0),
                1.0,
                terminated,
                &(&raw * 2.0),
                10.0,
            );
            // The constant value is 2 and gamma is 0.5: target is 1 at true
            // terminal, 2 at truncation. A leaked replay reward would give 10/11.
            assert_eq!(
                ppo.last_value_loss,
                Some(if terminated { 1.0 } else { 0.0 })
            );
            assert_eq!(ppo.update_count, 1);
            let experience = replay.sample(1, false).pop().unwrap();
            assert_eq!(
                *experience.n_step_discounted_reward.lock().unwrap(),
                Some(10.0)
            );
        }
    }

    #[test]
    fn test_ppo_shares_experience_with_replay_buffer() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 2, 2, 64, 0.0);
        let buffer_for_share_experience = Arc::new(ReplayBuffer::new(100, 1));
        let mut ppo = PPO::new(
            Box::new(model),
            optimizer,
            0.99,
            0.99,
            100,
            8,
            16,
            0.1,
            0.2,
            1.0,
            1.0,
            false,
            None,
            None,
        );
        ppo.add_replay_buffer_for_share_experience(buffer_for_share_experience.clone());

        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
        let _ = ppo.act_and_train(&obs, 0.0);
        ppo.stop_episode_and_train(&obs, 1.0);

        assert_eq!(buffer_for_share_experience.len(), 1);
        let experience = buffer_for_share_experience.sample(1, false).pop().unwrap();
        let on_policy_experiences = ppo
            .experiences_by_episode
            .get(&experience.episode_id)
            .unwrap()
            .to_vec();
        assert!(!Arc::ptr_eq(&on_policy_experiences[0], &experience));
        assert!(experience.action_distrib.is_none());
        assert_eq!(experience.state.device(), Device::Cpu);
        assert_eq!(experience.state.size(), [1, 4]);
        assert!(experience.action.is_some());
        assert!(!experience.is_episode_terminal);
        assert_eq!(experience.reward, 0.0);
        assert_eq!(
            experience
                .n_step_after_experience
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .is_episode_terminal,
            true
        );
        assert_eq!(
            *experience.n_step_discounted_reward.lock().unwrap(),
            Some(1.0)
        );
    }

    #[test]
    fn test_ppo_retains_raw_action_but_shares_executed_action() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let replay = Arc::new(ReplayBuffer::new(100, 1));
        let mut ppo = PPO::new(
            Box::new(BoundaryGaussianPolicy),
            optimizer,
            0.99,
            0.95,
            100,
            1,
            1,
            0.2,
            0.2,
            0.5,
            0.01,
            true,
            None,
            None,
        );
        ppo.add_replay_buffer_for_share_experience(replay.clone());
        let obs = Tensor::zeros([4], (Kind::Float, Device::Cpu));
        let executed_action = ppo.act_and_train(&obs, 0.0);
        assert_eq!(executed_action.double_value(&[0, 0]), 1.0);
        assert_eq!(executed_action.double_value(&[0, 1]), -1.0);
        assert!(ppo.act(&obs).equal(&executed_action));
        ppo.stop_episode_and_train(&obs, 2.0);

        let shared = replay.sample(1, false).pop().unwrap();
        let on_policy = ppo.experiences_by_episode[&shared.episode_id].to_vec();
        let raw_action = on_policy[0].action.as_ref().unwrap();
        assert!(raw_action.double_value(&[0, 0]) > 1.0);
        assert!(raw_action.double_value(&[0, 1]) < -1.0);
        let distribution = on_policy[0].action_distrib.as_ref().unwrap();
        let raw_log_prob = distribution.log_prob(raw_action).double_value(&[0]);
        let clipped_log_prob = distribution.log_prob(&executed_action).double_value(&[0]);
        assert!(raw_log_prob.is_finite());
        assert!(raw_log_prob > clipped_log_prob);

        assert!(shared.action.as_ref().unwrap().equal(&executed_action));
        assert_eq!(shared.action.as_ref().unwrap().device(), Device::Cpu);
        assert!(shared.action_distrib.is_none());
        assert_eq!(*shared.n_step_discounted_reward.lock().unwrap(), Some(2.0));
    }

    #[test]
    fn test_ppo_adds_curiosity_reward_before_updating() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 2, 2, 64, 0.0);
        let calc_count = Arc::new(AtomicUsize::new(0));
        let update_count = Arc::new(AtomicUsize::new(0));
        let updated_experience_count = Arc::new(AtomicUsize::new(0));
        let mut ppo = PPO::new(
            Box::new(model),
            optimizer,
            0.99,
            0.99,
            100,
            8,
            16,
            0.1,
            0.2,
            1.0,
            1.0,
            false,
            None,
            None,
        );
        ppo.add_curiosity(
            Box::new(ConstantCuriosity {
                reward: 4.0,
                calc_count: calc_count.clone(),
                update_count: update_count.clone(),
                updated_experience_count: updated_experience_count.clone(),
            }),
            0.25,
        );
        let experience1 = Arc::new(Experience::new(
            Ulid::new(),
            Ulid::new(),
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]),
            None,
            None,
            2.0,
            false,
        ));
        let experience2 = Arc::new(Experience::new(
            Ulid::new(),
            Ulid::new(),
            Tensor::from_slice(&[4.0, 3.0, 2.0, 1.0]),
            None,
            None,
            3.0,
            false,
        ));

        let reward = ppo._compute_rewards(&[experience1, experience2]);

        assert_eq!(reward.double_value(&[0]), 3.0);
        assert_eq!(reward.double_value(&[1]), 4.0);
        assert_eq!(calc_count.load(Ordering::Relaxed), 1);
        assert_eq!(update_count.load(Ordering::Relaxed), 1);
        assert_eq!(updated_experience_count.load(Ordering::Relaxed), 2);
        assert_eq!(ppo.curiosity_reward_coef, 0.25);
    }

    #[test]
    fn test_ppo_update_updates_curiosity_with_rollout_batch() {
        let calc_count = Arc::new(AtomicUsize::new(0));
        let update_count = Arc::new(AtomicUsize::new(0));
        let updated_experience_count = Arc::new(AtomicUsize::new(0));
        let mut ppo = test_ppo_with_update_interval(2);
        ppo.add_curiosity(
            ConstantCuriosity {
                reward: 1.0,
                calc_count: Arc::clone(&calc_count),
                update_count: Arc::clone(&update_count),
                updated_experience_count: Arc::clone(&updated_experience_count),
            },
            0.5,
        );
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);

        let _ = ppo.act_and_train(&obs, 0.0);
        let _ = ppo.act_and_train(&obs, 1.0);
        assert_eq!(update_count.load(Ordering::Relaxed), 0);
        let _ = ppo.act_and_train(&obs, 1.0);

        assert_eq!(calc_count.load(Ordering::Relaxed), 1);
        assert_eq!(update_count.load(Ordering::Relaxed), 1);
        assert_eq!(updated_experience_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_parallel_ppo_agents_share_curiosity() {
        let calc_count = Arc::new(AtomicUsize::new(0));
        let update_count = Arc::new(AtomicUsize::new(0));
        let updated_experience_count = Arc::new(AtomicUsize::new(0));
        let shared_curiosity = Arc::new(std::sync::Mutex::new(ConstantCuriosity {
            reward: 1.0,
            calc_count: Arc::clone(&calc_count),
            update_count: Arc::clone(&update_count),
            updated_experience_count: Arc::clone(&updated_experience_count),
        }));
        let mut ppo1 = test_ppo_with_update_interval(2);
        let mut ppo2 = test_ppo_with_update_interval(2);
        ppo1.add_curiosity(Arc::clone(&shared_curiosity), 0.5);
        ppo2.add_curiosity(Arc::clone(&shared_curiosity), 0.5);

        std::thread::scope(|scope| {
            for ppo in [&mut ppo1, &mut ppo2] {
                scope.spawn(move || {
                    let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
                    let _ = ppo.act_and_train(&obs, 0.0);
                    let _ = ppo.act_and_train(&obs, 1.0);
                    let _ = ppo.act_and_train(&obs, 1.0);
                });
            }
        });

        assert_eq!(calc_count.load(Ordering::Relaxed), 2);
        assert_eq!(update_count.load(Ordering::Relaxed), 2);
        assert_eq!(updated_experience_count.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_update_cadence_counts_completed_transitions_for_every_episode_length() {
        for episode_length in [1, 2, 7, 1000] {
            let updated_count = Arc::new(AtomicUsize::new(0));
            let mut ppo = test_ppo_with_update_interval(128);
            ppo.minibatch_size = 32;
            ppo.add_curiosity(
                ConstantCuriosity {
                    reward: 1.0,
                    calc_count: Arc::new(AtomicUsize::new(0)),
                    update_count: Arc::new(AtomicUsize::new(0)),
                    updated_experience_count: Arc::clone(&updated_count),
                },
                0.1,
            );
            let obs = Tensor::ones([4], (Kind::Float, Device::Cpu));
            for step in 0..256 {
                let _ = ppo.act_and_train(&obs, if step % episode_length == 0 { 0.0 } else { 1.0 });
                assert_eq!(
                    ppo.update_count,
                    step / 128,
                    "episode_length={episode_length}, step={step}"
                );
                if (step + 1) % episode_length == 0 || step == 255 {
                    ppo.stop_episode_and_train(&obs, 1.0);
                    assert_eq!(ppo.update_count, (step + 1) / 128);
                }
            }
            assert_eq!(ppo.update_count, 2);
            assert_eq!(updated_count.load(Ordering::Relaxed), 256);
        }
    }

    fn test_ppo_with_update_interval(update_interval: usize) -> PPO {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 2, 1, 8, 0.0);
        PPO::new(
            Box::new(model),
            optimizer,
            0.99,
            0.95,
            update_interval,
            1,
            1,
            0.2,
            0.2,
            0.5,
            0.01,
            false,
            None,
            None,
        )
    }

    #[test]
    fn test_ppo_act_and_train() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 4, 2, 64, 0.0);

        let mut ppo = PPO::new(
            Box::new(model),
            optimizer,
            0.5,
            0.99,
            100,
            3,
            32,
            0.1,
            0.2,
            1.0,
            1.0,
            false,
            None,
            None,
        );

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
        }
        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);
        ppo.stop_episode_and_train(&obs, 1.0);

        assert!(ppo.update_count > 0);
        assert!(ppo.last_policy_loss.is_some_and(f64::is_finite));
        assert!(ppo.last_value_loss.is_some_and(f64::is_finite));
        assert!(ppo
            .last_entropy
            .is_some_and(|entropy| entropy.is_finite() && entropy >= 0.0));

        let action = ppo.act(&obs);
        let action_value = action.int64_value(&[]);
        assert!([0, 1, 2, 3].contains(&action_value));
    }

    #[test]
    fn test_ppo_act_and_train_multi_branch_discrete() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new_multi(vs, 4, vec![3, 2], 1, 8, 0.0);

        let mut ppo = PPO::new(
            Box::new(model),
            optimizer,
            0.5,
            0.99,
            2,
            1,
            1,
            0.1,
            0.2,
            1.0,
            1.0,
            false,
            None,
            None,
        );

        let obs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(Kind::Float);

        for i in 0..100 {
            let action = ppo.act_and_train(&obs, 1.0);
            assert_multi_branch_action(&action);
            assert_eq!(ppo.t, i + 1);

            let action = ppo.act(&obs);
            assert_multi_branch_action(&action);
        }
    }

    fn assert_multi_branch_action(action: &Tensor) {
        assert_eq!(action.size(), [1, 2]);

        let branch0 = action.int64_value(&[0, 0]);
        let branch1 = action.int64_value(&[0, 1]);
        assert!(0 <= branch0 && branch0 < 3);
        assert!(0 <= branch1 && branch1 < 2);
    }
}
