use std::path::Path;

use tch::Tensor;

use ulid::Ulid;

pub trait BaseAgent {
    /// Whether the existing optimizer supports runtime learning-rate changes.
    fn supports_learning_rate_update(&self) -> bool {
        false
    }
    /// Set a finite, strictly positive learning rate without resetting optimizer
    /// state, model parameters, or counters. Unsupported agents and invalid rates
    /// panic before mutation; dynamic callers should check the capability first.
    /// This does not change checkpoint contents or implement a schedule.
    fn set_learning_rate(&mut self, _learning_rate: f64) {
        panic!("runtime learning-rate changes are not supported by this agent");
    }
    fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;
    /// Whether this agent can keep learner inputs separate from shared replay.
    fn supports_separate_replay_input(&self) -> bool {
        false
    }
    /// `reward` and `replay_reward` both belong to the previous action; the
    /// observations must describe the same current environment state.
    fn act_and_train_with_replay_input(
        &mut self,
        _obs: &Tensor,
        _reward: f64,
        _replay_obs: &Tensor,
        _replay_reward: f64,
    ) -> Tensor {
        panic!("separate replay inputs are not supported by this agent");
    }
    fn act(&self, obs: &Tensor) -> Tensor;
    fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64);
    /// End the rollout. A time limit (`terminated = false`) still bootstraps
    /// from `obs`; a true MDP terminal state does not.
    fn stop_episode_and_train_with_terminal(
        &mut self,
        obs: &Tensor,
        reward: f64,
        _terminated: bool,
    ) {
        self.stop_episode_and_train(obs, reward);
    }
    /// End the same episode in learner and replay streams. `terminated = false`
    /// preserves bootstrap in both streams while still ending the episode.
    fn stop_episode_and_train_with_replay_input(
        &mut self,
        _obs: &Tensor,
        _reward: f64,
        _terminated: bool,
        _replay_obs: &Tensor,
        _replay_reward: f64,
    ) {
        panic!("separate replay inputs are not supported by this agent");
    }
    fn get_statistics(&self) -> Vec<(String, f64)>;
    fn get_agent_id(&self) -> &Ulid;
    fn save(&self);
    fn load(&mut self);
}

pub(crate) fn ensure_parent_dir(path: &str) {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("failed to create model directory {:?}: {}", parent, e));
        }
    }
}
