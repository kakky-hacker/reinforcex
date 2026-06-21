use rayon::{prelude::*, ThreadPoolBuilder};
use reinforcex::agents::{BaseAgent, PPO, SAC};
use reinforcex::curiousity::{BaseCuriousity, RND};
use reinforcex::memory::{Experience, ReplayBuffer};
use reinforcex::models::{FCGaussianPolicy, FCGaussianPolicyWithValue, FCQNetwork, FCRNDModel};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use ulid::Ulid;

#[derive(Clone, Copy)]
pub(super) struct PpoHyperparameters {
    pub learning_rate: f64,
    pub hidden_layers: usize,
    pub hidden_channels: i64,
    pub gamma: f64,
    pub lambda: f64,
    pub update_interval: usize,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub policy_clip: f64,
    pub value_clip: f64,
    pub value_coefficient: f64,
    pub entropy_coefficient: f64,
    pub minimum_variance: f64,
    pub gae_standardize: bool,
}

impl Default for PpoHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            hidden_layers: 2,
            hidden_channels: 256,
            gamma: 0.99,
            lambda: 0.95,
            update_interval: 2048,
            epochs: 10,
            minibatch_size: 64,
            policy_clip: 0.2,
            value_clip: 0.2,
            value_coefficient: 0.5,
            entropy_coefficient: 0.005,
            minimum_variance: 0.1,
            gae_standardize: true,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SacHyperparameters {
    pub learning_rate: f64,
    pub hidden_layers: usize,
    pub hidden_channels: i64,
    pub replay_capacity: usize,
    pub replay_start_size: usize,
    pub batch_size: usize,
    pub update_interval: usize,
    pub target_update_interval: usize,
    pub gamma: f64,
    pub tau: f64,
    pub initial_temperature: f64,
    pub minimum_variance: f64,
}

impl Default for SacHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            hidden_layers: 2,
            hidden_channels: 256,
            replay_capacity: 300_000,
            replay_start_size: 10_000,
            batch_size: 256,
            update_interval: 1,
            target_update_interval: 1,
            gamma: 0.99,
            tau: 0.005,
            initial_temperature: 0.2,
            minimum_variance: 1e-3,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct RndHyperparameters {
    pub learning_rate: f64,
    pub feature_size: i64,
    pub hidden_layers: usize,
    pub hidden_channels: i64,
    pub update_interval: usize,
    pub intrinsic_reward_scale: f64,
}

impl Default for RndHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            feature_size: 128,
            hidden_layers: 2,
            hidden_channels: 256,
            update_interval: 128,
            intrinsic_reward_scale: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ContinuousEnvConfig {
    pub label: &'static str,
    pub gym_id: &'static str,
    pub obs_size: i64,
    pub action_size: i64,
    pub action_low: f64,
    pub action_high: f64,
    pub episodes: usize,
    pub max_steps: usize,
    pub log_interval: usize,
    pub ppo_reward_clip: f64,
    pub ppo: PpoHyperparameters,
    pub sac: SacHyperparameters,
    pub rnd: RndHyperparameters,
}

#[derive(Deserialize)]
struct ResetResponse {
    session_id: String,
    observation: Vec<f32>,
}

#[derive(Deserialize)]
struct StepResponse {
    observation: Vec<f32>,
    reward: f64,
    done: bool,
}

fn client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed")
}

fn reset(client: &Client, base_url: &str, gym_id: &str) -> ResetResponse {
    client
        .post(format!("{}/reset", base_url))
        .json(&serde_json::json!({ "env": gym_id }))
        .send()
        .expect("reset failed")
        .json::<ResetResponse>()
        .expect("reset JSON parse failed")
}

fn step(client: &Client, base_url: &str, session_id: &str, action: &[f32]) -> StepResponse {
    client
        .post(format!("{}/step", base_url))
        .json(&serde_json::json!({ "session_id": session_id, "action": action }))
        .send()
        .expect("step failed")
        .json::<StepResponse>()
        .expect("step JSON parse failed")
}

fn observation_tensor(observation: &[f32], expected_size: i64) -> Tensor {
    assert_eq!(
        observation.len(),
        expected_size as usize,
        "environment observation size does not match example configuration"
    );
    Tensor::from_slice(observation).to_kind(Kind::Float)
}

fn direct_action(action: &Tensor, config: ContinuousEnvConfig) -> Vec<f32> {
    let action = action
        .flatten(0, -1)
        .clamp(config.action_low, config.action_high);
    (0..action.size()[0])
        .map(|i| action.double_value(&[i]) as f32)
        .collect()
}

fn normalized_action(action: &Tensor, config: ContinuousEnvConfig) -> Vec<f32> {
    let action = action.flatten(0, -1).clamp(-1.0, 1.0);
    let scale = (config.action_high - config.action_low) / 2.0;
    let midpoint = (config.action_high + config.action_low) / 2.0;
    (0..action.size()[0])
        .map(|i| (action.double_value(&[i]) * scale + midpoint) as f32)
        .collect()
}

fn build_ppo(config: ContinuousEnvConfig, save: Option<String>, load: Option<String>) -> PPO {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let hp = config.ppo;
    let optimizer = nn::Adam::default().build(&vs, hp.learning_rate).unwrap();
    let model = FCGaussianPolicyWithValue::new(
        vs,
        config.obs_size,
        config.action_size,
        hp.hidden_layers,
        hp.hidden_channels,
        Some(config.action_low),
        Some(config.action_high),
        true,
        "spherical",
        hp.minimum_variance,
    );
    PPO::new(
        Box::new(model),
        optimizer,
        hp.gamma,
        hp.lambda,
        hp.update_interval,
        hp.epochs,
        hp.minibatch_size,
        hp.policy_clip,
        hp.value_clip,
        hp.value_coefficient,
        hp.entropy_coefficient,
        hp.gae_standardize,
        save,
        load,
    )
}

fn build_rnd(config: ContinuousEnvConfig, save: Option<String>, load: Option<String>) -> RND {
    let device = Device::cuda_if_available();
    let hp = config.rnd;
    let model = FCRNDModel::new(
        nn::VarStore::new(device),
        nn::VarStore::new(device),
        config.obs_size,
        hp.feature_size,
        hp.hidden_layers,
        hp.hidden_channels,
    );
    let optimizer = nn::Adam::default()
        .build(model.predictor_var_store(), hp.learning_rate)
        .unwrap();
    RND::new(Box::new(model), optimizer, hp.update_interval, save, load)
}

fn build_sac(
    config: ContinuousEnvConfig,
    replay: Arc<ReplayBuffer>,
    save: Option<String>,
    load: Option<String>,
) -> SAC {
    let device = Device::cuda_if_available();
    let hp = config.sac;
    let actor_vs = nn::VarStore::new(device);
    let actor_optimizer = nn::Adam::default()
        .build(&actor_vs, hp.learning_rate)
        .unwrap();
    let actor = FCGaussianPolicy::new(
        actor_vs,
        config.obs_size,
        config.action_size,
        hp.hidden_layers,
        hp.hidden_channels,
        None,
        None,
        false,
        "diagonal",
        hp.minimum_variance,
    );

    let critic1_vs = nn::VarStore::new(device);
    let critic1_optimizer = nn::Adam::default()
        .build(&critic1_vs, hp.learning_rate)
        .unwrap();
    let critic1 = FCQNetwork::new(
        critic1_vs,
        config.obs_size + config.action_size,
        1,
        hp.hidden_layers,
        hp.hidden_channels,
    );

    let critic2_vs = nn::VarStore::new(device);
    let critic2_optimizer = nn::Adam::default()
        .build(&critic2_vs, hp.learning_rate)
        .unwrap();
    let critic2 = FCQNetwork::new(
        critic2_vs,
        config.obs_size + config.action_size,
        1,
        hp.hidden_layers,
        hp.hidden_channels,
    );

    SAC::new_with_save_load(
        Box::new(actor),
        actor_optimizer,
        Box::new(critic1),
        critic1_optimizer,
        Box::new(critic2),
        critic2_optimizer,
        replay,
        hp.replay_start_size,
        hp.batch_size,
        hp.update_interval,
        hp.target_update_interval,
        hp.gamma,
        hp.tau,
        hp.initial_temperature,
        true,
        save,
        load,
    )
}

fn curiosity_experience(state: &Tensor, terminal: bool) -> Arc<Experience> {
    Arc::new(Experience::new(
        Ulid::new(),
        Ulid::new(),
        state.shallow_clone(),
        None,
        None,
        0.0,
        terminal,
        Mutex::new(None),
        Mutex::new(None),
    ))
}

fn intrinsic_reward(curiosity: &mut RND, state: &Tensor, terminal: bool) -> f64 {
    let experience = curiosity_experience(state, terminal);
    let reward = curiosity
        .calc_reward(Arc::clone(&experience))
        .to_device(Device::Cpu)
        .mean(Kind::Float)
        .double_value(&[]);
    curiosity.observe(experience);
    reward
}

fn rnd_path(path: &Option<String>) -> Option<String> {
    path.as_ref().map(|path| format!("{}.rnd", path))
}

fn shared_rnd_path(path: &Option<String>) -> Option<String> {
    path.as_ref()
        .map(|path| format!("{}.rnd", path.replace("{agent_id}", "shared")))
}

fn run_ppo_worker(
    config: ContinuousEnvConfig,
    port: u16,
    agent_id: usize,
    save: Option<String>,
    load: Option<String>,
    curiosity: Option<Arc<Mutex<RND>>>,
    save_curiosity: bool,
) {
    let client = client();
    let base_url = format!("http://localhost:{}", port);
    let mut agent = build_ppo(config, save, load);
    let start = Instant::now();
    let mut total_reward = 0.0;
    let mut total_intrinsic = 0.0;
    let mut total_steps = 0;

    for episode in 1..=config.episodes {
        let reset = reset(&client, &base_url, config.gym_id);
        let mut observation = reset.observation;
        let initial_state = observation_tensor(&observation, config.obs_size);
        if let Some(curiosity) = &curiosity {
            curiosity
                .lock()
                .unwrap()
                .observe(curiosity_experience(&initial_state, false));
        }
        let mut reward = 0.0;

        for current_step in 0..config.max_steps {
            let state = observation_tensor(&observation, config.obs_size);
            let action_tensor = agent.act_and_train(&state, reward);
            let action = direct_action(&action_tensor, config);
            let response = step(&client, &base_url, &reset.session_id, &action);
            observation = response.observation;
            let next_state = observation_tensor(&observation, config.obs_size);
            let terminal = response.done || current_step + 1 == config.max_steps;
            let intrinsic = curiosity
                .as_ref()
                .map(|curiosity| {
                    intrinsic_reward(&mut curiosity.lock().unwrap(), &next_state, terminal)
                })
                .unwrap_or(0.0);
            reward = response
                .reward
                .clamp(-config.ppo_reward_clip, config.ppo_reward_clip)
                + config.rnd.intrinsic_reward_scale * intrinsic;
            total_reward += response.reward;
            total_intrinsic += intrinsic;
            total_steps += 1;

            if terminal {
                agent.stop_episode_and_train(&next_state, reward);
                break;
            }
        }

        if episode % config.log_interval == 0 {
            println!(
                "[{} PPO{} {}] Episode {}, Avg Reward: {:.2}, Avg Intrinsic: {:.3}, Avg Steps: {}, Elapsed: {:?}",
                config.label,
                if curiosity.is_some() { "+RND" } else { "" },
                agent_id,
                episode,
                total_reward / config.log_interval as f64,
                total_intrinsic / config.log_interval as f64,
                total_steps / config.log_interval,
                start.elapsed()
            );
            total_reward = 0.0;
            total_intrinsic = 0.0;
            total_steps = 0;
            agent.save();
            if save_curiosity {
                if let Some(curiosity) = &curiosity {
                    curiosity.lock().unwrap().save();
                }
            }
        }
    }
    agent.save();
    if save_curiosity {
        if let Some(curiosity) = curiosity {
            curiosity.lock().unwrap().save();
        }
    }
}

fn run_sac_worker(
    config: ContinuousEnvConfig,
    port: u16,
    agent_id: usize,
    replay: Arc<ReplayBuffer>,
    save: Option<String>,
    load: Option<String>,
) {
    let client = client();
    let base_url = format!("http://localhost:{}", port);
    let mut agent = build_sac(config, Arc::clone(&replay), save, load);
    let start = Instant::now();
    let mut total_reward = 0.0;
    let mut total_steps = 0;

    for episode in 1..=config.episodes {
        let reset = reset(&client, &base_url, config.gym_id);
        let mut observation = reset.observation;
        let mut reward = 0.0;

        for current_step in 0..config.max_steps {
            let state = observation_tensor(&observation, config.obs_size);
            let normalized = agent.act_and_train(&state, reward);
            let action = normalized_action(&normalized, config);
            let response = step(&client, &base_url, &reset.session_id, &action);
            observation = response.observation;
            reward = response.reward;
            total_reward += reward;
            total_steps += 1;

            if response.done || current_step + 1 == config.max_steps {
                let next_state = observation_tensor(&observation, config.obs_size);
                agent.stop_episode_and_train(&next_state, reward);
                break;
            }
        }

        if episode % config.log_interval == 0 {
            println!(
                "[{} SAC {}] Episode {}, Avg Reward: {:.2}, Avg Steps: {}, Replay: {}, Stats: {:?}, Elapsed: {:?}",
                config.label,
                agent_id,
                episode,
                total_reward / config.log_interval as f64,
                total_steps / config.log_interval,
                replay.len(),
                agent.get_statistics(),
                start.elapsed()
            );
            total_reward = 0.0;
            total_steps = 0;
            agent.save();
        }
    }
    agent.save();
}

fn install_workers<F>(parallel_count: usize, work: F)
where
    F: Fn(usize, u16) + Sync + Send,
{
    let ports = super::environment_ports(parallel_count);
    ThreadPoolBuilder::new()
        .num_threads(parallel_count)
        .build()
        .expect("failed to build continuous training thread pool")
        .install(|| {
            ports
                .into_par_iter()
                .enumerate()
                .for_each(|(agent_id, port)| work(agent_id, port));
        });
}

pub(super) fn train_ppo(
    config: ContinuousEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    install_workers(parallel_count, |agent_id, port| {
        run_ppo_worker(
            config,
            port,
            agent_id,
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
            None,
            false,
        );
    });
}

pub(super) fn train_ppo_rnd(
    config: ContinuousEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    install_workers(parallel_count, |agent_id, port| {
        let agent_save = super::path_for_agent(&save_path, agent_id);
        let agent_load = super::path_for_agent(&load_path, agent_id);
        let curiosity = Arc::new(Mutex::new(build_rnd(
            config,
            rnd_path(&agent_save),
            rnd_path(&agent_load),
        )));
        run_ppo_worker(
            config,
            port,
            agent_id,
            agent_save,
            agent_load,
            Some(curiosity),
            true,
        );
    });
}

pub(super) fn train_ppo_shared_rnd(
    config: ContinuousEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let curiosity = Arc::new(Mutex::new(build_rnd(
        config,
        shared_rnd_path(&save_path),
        shared_rnd_path(&load_path),
    )));
    install_workers(parallel_count, |agent_id, port| {
        run_ppo_worker(
            config,
            port,
            agent_id,
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
            Some(Arc::clone(&curiosity)),
            agent_id == 0,
        );
    });
    curiosity.lock().unwrap().save();
}

pub(super) fn train_sac(
    config: ContinuousEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let replay = Arc::new(ReplayBuffer::new(config.sac.replay_capacity, 1));
    install_workers(parallel_count, |agent_id, port| {
        run_sac_worker(
            config,
            port,
            agent_id,
            Arc::clone(&replay),
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ContinuousEnvConfig {
        ContinuousEnvConfig {
            label: "test",
            gym_id: "test-v0",
            obs_size: 3,
            action_size: 2,
            action_low: -2.0,
            action_high: 2.0,
            episodes: 1,
            max_steps: 1,
            log_interval: 1,
            ppo_reward_clip: 1.0,
            ppo: PpoHyperparameters::default(),
            sac: SacHyperparameters::default(),
            rnd: RndHyperparameters::default(),
        }
    }

    #[test]
    fn normalized_action_maps_to_environment_bounds() {
        let action = Tensor::from_slice(&[-1.0f32, 1.0]);
        assert_eq!(normalized_action(&action, config()), vec![-2.0, 2.0]);
    }

    #[test]
    fn shared_rnd_checkpoint_replaces_agent_placeholder() {
        let path = Some("models/model_{agent_id}.ot".to_string());
        assert_eq!(
            shared_rnd_path(&path).as_deref(),
            Some("models/model_shared.ot.rnd")
        );
    }
}
