use super::base_agent::{ensure_parent_dir, BaseAgent};
use crate::curiosity::Basecuriosity;
use crate::memory::{Experience, ReplayBuffer};
use crate::misc::batch_states::batch_states;
use crate::misc::bounded_vec_deque::BoundedVecDeque;
use crate::misc::cumsum::cumsum_rev;
use crate::models::BasePolicy;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{collections::HashMap, sync::Arc};
use tch::{nn, no_grad, Device, Kind, Tensor};
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
    t: usize,
    current_episode_id: Ulid,
    save_path: Option<String>,
    load_path: Option<String>,
}

unsafe impl Send for PPO {}

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
        assert!(minibatch_size <= update_interval);
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
            t: 0,
            current_episode_id: Ulid::new(),
            save_path,
            load_path,
        };
        agent.load();
        agent
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
        .to_device(device);

        if experiences.is_empty() {
            return extrinsic_rewards;
        }

        let Some(curiosity) = self.curiosity.as_mut() else {
            return extrinsic_rewards;
        };

        let intrinsic_rewards = curiosity
            .calc_internal_reward(experiences)
            .to_device(device)
            .view([-1]);
        assert_eq!(intrinsic_rewards.numel(), experiences.len());

        for experience in experiences {
            curiosity.observe(Arc::clone(experience));
        }

        extrinsic_rewards + self.curiosity_reward_coef * intrinsic_rewards
    }

    fn _update(&mut self) {
        let experiences_per_episode = self
            .experiences_by_episode
            .drain()
            .map(|(_episode_id, experiences)| experiences.to_vec())
            .collect::<Vec<Vec<Arc<Experience>>>>();

        let total_transitions = experiences_per_episode
            .iter()
            .map(|v| v.len())
            .sum::<usize>()
            - experiences_per_episode.len();
        let n_iter = total_transitions.div_ceil(self.minibatch_size);
        let n_data_per_epoch = n_iter * self.minibatch_size;
        let n_data = n_data_per_epoch * self.epoch;

        // Create shuffled indice for minibatch.
        let mut rng = thread_rng();
        let mut batch_indice = (0..total_transitions).collect::<Vec<usize>>();
        let mut all_indice =
            Vec::with_capacity(total_transitions * n_data.div_ceil(total_transitions));
        for _ in 0..n_data.div_ceil(total_transitions) {
            batch_indice.shuffle(&mut rng);
            all_indice.extend(batch_indice.iter().cloned());
        }
        let all_indice = all_indice
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();

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

        let (old_action_distrib, old_value) = self.model.forward(&state);
        let old_value = old_value.unwrap().flatten(0, 1).detach();
        let old_log_prob = old_action_distrib.log_prob(&action.detach()).detach();

        let (_, old_next_value) = self.model.forward(&next_state);
        let old_next_value = old_next_value.unwrap().flatten(0, 1).detach();

        let non_terminal: Tensor = 1.0
            - Tensor::from_slice(
                &_skip_first
                    .iter()
                    .map(|e| if e.is_episode_terminal { 1.0 } else { 0.0 })
                    .collect::<Vec<f64>>(),
            )
            .to_device(self.model.device());

        let old_next_value = (old_next_value * non_terminal).detach();

        // Compute GAE
        let td_error = (reward + self.gamma * &old_next_value - &old_value).to_device(Device::Cpu);
        let _gae = Tensor::from_slice(&cumsum_rev(
            &(0..td_error.size()[0])
                .map(|i| td_error.double_value(&[i]))
                .collect::<Vec<f64>>(),
            &_skip_first
                .iter()
                .map(|e| {
                    if e.is_episode_terminal {
                        0.0 // For preventing td_error from passing through between different episodes.
                    } else {
                        self.gamma * self.lambda
                    }
                })
                .collect::<Vec<f64>>(),
        ))
        .to_device(self.model.device())
        .detach();
        let gae = if self.gae_std {
            (&_gae - (&_gae).mean(Kind::Float)) / ((&_gae).std(false) + 1e-8)
        } else {
            _gae
        };

        // Compute value target
        let value_target = &gae + &old_value;

        for i in 0..self.epoch {
            for j in 0..n_iter {
                let minibatch_indice = Tensor::from_slice(
                    &all_indice[i * n_data_per_epoch + j * self.minibatch_size
                        ..i * n_data_per_epoch + (j + 1) * self.minibatch_size],
                )
                .to_device(self.model.device());

                // Forward
                let (action_distrib, value) = self.model.forward(&state);
                let value = value
                    .unwrap()
                    .flatten(0, 1)
                    .index_select(0, &minibatch_indice);

                // Compute policy ratio.
                let log_prob = action_distrib.log_prob(&action.detach());
                let policy_ratio = (log_prob - &old_log_prob)
                    .index_select(0, &minibatch_indice)
                    .clamp(
                        -POLICY_LOG_PROB_RATIO_CLAMP_RANGE,
                        POLICY_LOG_PROB_RATIO_CLAMP_RANGE,
                    )
                    .exp();
                let clipped_policy_ratio = policy_ratio.clamp(
                    1.0 - self.policy_clip_epsilon,
                    1.0 + self.policy_clip_epsilon,
                );

                // Compute value ratio.
                let _old_value = old_value.index_select(0, &minibatch_indice).detach();
                let clipped_value = &_old_value
                    + (&value - &_old_value).clamp(-self.value_clip_range, self.value_clip_range);

                let minibatch_gae = gae.index_select(0, &minibatch_indice).detach();

                // Compute loss
                let _value_target = (&value_target).index_select(0, &minibatch_indice).detach();
                let policy_loss: Tensor = -1.0
                    * (policy_ratio * &minibatch_gae)
                        .minimum(&(clipped_policy_ratio * &minibatch_gae))
                        .mean(Kind::Float);
                let value_loss = (&_value_target - value)
                    .square()
                    .maximum(&(&_value_target - clipped_value).square())
                    .mean(Kind::Float);
                let entropy_regularized = action_distrib
                    .entropy()
                    .index_select(0, &minibatch_indice)
                    .mean(Kind::Float);

                // Check Nan
                assert!(policy_loss.isnan().any().int64_value(&[]) == 0);
                assert!(value_loss.isnan().any().int64_value(&[]) == 0);
                assert!(entropy_regularized.isnan().any().int64_value(&[]) == 0);

                let loss: Tensor = policy_loss + self.value_coef * value_loss
                    - self.entropy_coef * entropy_regularized;

                // Backward
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            }
        }
    }
}

impl BaseAgent for PPO {
    fn act(&self, obs: &Tensor) -> Tensor {
        no_grad(|| {
            let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
            let (action_distrib, _) = self.model.forward(&state);
            let action = action_distrib.most_probable().to_device(Device::Cpu);
            action
        })
    }

    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor {
        self.t += 1;

        let state = batch_states(&vec![obs.shallow_clone()], self.model.device());
        let action_distrib = no_grad(|| {
            let (action_distrib, _) = self.model.forward(&state);
            action_distrib
        });
        let action = action_distrib.sample().detach().to_device(Device::Cpu);

        let experience = Arc::new(Experience::new(
            self.agent_id,
            self.current_episode_id,
            state,
            Some(action.shallow_clone()),
            Some(action_distrib),
            reward,
            false,
        ));
        self.experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        if let Some(buffer_for_share_experience) = &self.buffer_for_share_experience {
            buffer_for_share_experience.append(experience, self.gamma);
        }

        if self.t % self.update_interval == 0 {
            self._update();
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
        self.experiences_by_episode
            .entry(experience.episode_id)
            .or_insert_with(|| BoundedVecDeque::new(1e9 as usize))
            .push_back(experience.clone());

        if let Some(buffer_for_share_experience) = &self.buffer_for_share_experience {
            buffer_for_share_experience.append(experience, self.gamma);
        }
        self.current_episode_id = Ulid::new();
    }

    fn get_statistics(&self) -> Vec<(String, f64)> {
        vec![]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ReplayBuffer;
    use crate::models::FCSoftmaxPolicyWithValue;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    struct ConstantCuriosity {
        reward: f64,
        calc_count: Arc<AtomicUsize>,
        observe_count: Arc<AtomicUsize>,
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

        fn observe(&mut self, _experience: Arc<Experience>) {
            self.observe_count.fetch_add(1, Ordering::Relaxed);
        }

        fn save(&self) {}

        fn load(&mut self) {}
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
        assert!(Arc::ptr_eq(&on_policy_experiences[0], &experience));
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
    fn test_ppo_adds_curiosity_reward_and_observes_experience() {
        let vs = nn::VarStore::new(Device::Cpu);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let model = FCSoftmaxPolicyWithValue::new(vs, 4, 2, 2, 64, 0.0);
        let calc_count = Arc::new(AtomicUsize::new(0));
        let observe_count = Arc::new(AtomicUsize::new(0));
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
                observe_count: observe_count.clone(),
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
        assert_eq!(observe_count.load(Ordering::Relaxed), 2);
        assert_eq!(ppo.curiosity_reward_coef, 0.25);
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

        for _ in 0..1000 {
            let action = ppo.act(&obs);
            let action_value = i64::from(action.int64_value(&[]));
            assert_eq!(action_value, 2);
        }
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
