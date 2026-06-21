use rayon::{prelude::*, ThreadPoolBuilder};
use reinforcex::agents::{BaseAgent, DQN, PPO, SAC};
use reinforcex::curiousity::{BaseCuriousity, RND};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::{Experience, ReplayBuffer};
use reinforcex::models::{FCQNetwork, FCRNDModel, FCSoftmaxPolicy, FCSoftmaxPolicyWithValue};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use ulid::Ulid;

#[derive(Clone, Copy)]
pub(super) struct DqnHyperparameters {
    pub learning_rate: f64,
    pub hidden_layers: usize,
    pub hidden_channels: i64,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub update_interval: usize,
    pub target_update_interval: usize,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay_steps: usize,
    pub gamma: f64,
}

impl Default for DqnHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            hidden_layers: 2,
            hidden_channels: 128,
            replay_capacity: 100_000,
            batch_size: 64,
            update_interval: 4,
            target_update_interval: 200,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 50_000,
            gamma: 0.99,
        }
    }
}

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
    pub gae_standardize: bool,
}

impl Default for PpoHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 2.5e-4,
            hidden_layers: 2,
            hidden_channels: 128,
            gamma: 0.99,
            lambda: 0.95,
            update_interval: 1024,
            epochs: 10,
            minibatch_size: 64,
            policy_clip: 0.2,
            value_clip: 0.2,
            value_coefficient: 0.5,
            entropy_coefficient: 0.01,
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
}

impl Default for SacHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            hidden_layers: 2,
            hidden_channels: 128,
            replay_capacity: 100_000,
            replay_start_size: 1000,
            batch_size: 128,
            update_interval: 4,
            target_update_interval: 8,
            gamma: 0.99,
            tau: 0.01,
            initial_temperature: 0.2,
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
pub(super) struct DiscreteEnvConfig {
    pub label: &'static str,
    pub gym_id: &'static str,
    pub state_count: usize,
    pub action_count: usize,
    pub one_hot_state: bool,
    pub episodes: usize,
    pub max_steps: usize,
    pub log_interval: usize,
    pub dqn: DqnHyperparameters,
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

fn step(client: &Client, base_url: &str, session_id: &str, action: usize) -> StepResponse {
    client
        .post(format!("{}/step", base_url))
        .json(&serde_json::json!({ "session_id": session_id, "action": action }))
        .send()
        .expect("step failed")
        .json::<StepResponse>()
        .expect("step JSON parse failed")
}

fn encode_observation(observation: &[f32], config: DiscreteEnvConfig) -> Tensor {
    if config.one_hot_state {
        assert_eq!(
            observation.len(),
            1,
            "one-hot environment must return one scalar observation"
        );
        let state = observation[0] as usize;
        assert!(
            state < config.state_count,
            "discrete observation is out of range"
        );
        let mut encoded = vec![0.0f32; config.state_count];
        encoded[state] = 1.0;
        Tensor::from_slice(&encoded)
    } else {
        assert_eq!(
            observation.len(),
            config.state_count,
            "vector observation size does not match example configuration"
        );
        Tensor::from_slice(observation)
    }
}

fn build_dqn(
    config: DiscreteEnvConfig,
    replay: Arc<ReplayBuffer>,
    save: Option<String>,
    load: Option<String>,
) -> DQN {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let hp = config.dqn;
    let optimizer = nn::Adam::default().build(&vs, hp.learning_rate).unwrap();
    let model = FCQNetwork::new(
        vs,
        config.state_count as i64,
        config.action_count as i64,
        hp.hidden_layers,
        hp.hidden_channels,
    );
    DQN::new(
        Box::new(model),
        replay,
        optimizer,
        config.action_count,
        hp.batch_size,
        hp.update_interval,
        hp.target_update_interval,
        Box::new(EpsilonGreedy::new(
            hp.epsilon_start,
            hp.epsilon_end,
            hp.epsilon_decay_steps,
        )),
        None,
        hp.gamma,
        save,
        load,
    )
}

fn build_ppo(config: DiscreteEnvConfig, save: Option<String>, load: Option<String>) -> PPO {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let hp = config.ppo;
    let optimizer = nn::Adam::default().build(&vs, hp.learning_rate).unwrap();
    let model = FCSoftmaxPolicyWithValue::new(
        vs,
        config.state_count as i64,
        config.action_count as i64,
        hp.hidden_layers,
        hp.hidden_channels,
        0.0,
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

fn build_sac(
    config: DiscreteEnvConfig,
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
    let actor = FCSoftmaxPolicy::new(
        actor_vs,
        config.state_count as i64,
        config.action_count as i64,
        hp.hidden_layers,
        hp.hidden_channels,
        0.0,
    );

    let critic1_vs = nn::VarStore::new(device);
    let critic1_optimizer = nn::Adam::default()
        .build(&critic1_vs, hp.learning_rate)
        .unwrap();
    let critic1 = FCQNetwork::new(
        critic1_vs,
        config.state_count as i64,
        config.action_count as i64,
        hp.hidden_layers,
        hp.hidden_channels,
    );

    let critic2_vs = nn::VarStore::new(device);
    let critic2_optimizer = nn::Adam::default()
        .build(&critic2_vs, hp.learning_rate)
        .unwrap();
    let critic2 = FCQNetwork::new(
        critic2_vs,
        config.state_count as i64,
        config.action_count as i64,
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
        false,
        save,
        load,
    )
}

fn build_rnd(config: DiscreteEnvConfig, save: Option<String>, load: Option<String>) -> RND {
    let device = Device::cuda_if_available();
    let hp = config.rnd;
    let model = FCRNDModel::new(
        nn::VarStore::new(device),
        nn::VarStore::new(device),
        config.state_count as i64,
        hp.feature_size,
        hp.hidden_layers,
        hp.hidden_channels,
    );
    let optimizer = nn::Adam::default()
        .build(model.predictor_var_store(), hp.learning_rate)
        .unwrap();
    RND::new(Box::new(model), optimizer, hp.update_interval, save, load)
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

fn run_worker<A, F>(
    config: DiscreteEnvConfig,
    algorithm: &str,
    port: u16,
    agent_id: usize,
    mut agent: A,
    encode: F,
) where
    A: BaseAgent,
    F: Fn(&[f32]) -> Tensor,
{
    let client = client();
    let base_url = format!("http://localhost:{}", port);
    let start = Instant::now();
    let mut total_reward = 0.0;
    let mut total_steps = 0;

    for episode in 1..=config.episodes {
        let reset = reset(&client, &base_url, config.gym_id);
        let mut observation = reset.observation;
        let mut reward = 0.0;

        for current_step in 0..config.max_steps {
            let state = encode(&observation);
            let action_tensor = agent.act_and_train(&state, reward).flatten(0, -1);
            let action = action_tensor.int64_value(&[0]) as usize;
            let response = step(&client, &base_url, &reset.session_id, action);
            observation = response.observation;
            reward = response.reward;
            total_reward += reward;
            total_steps += 1;

            if response.done || current_step + 1 == config.max_steps {
                agent.stop_episode_and_train(&encode(&observation), reward);
                break;
            }
        }

        if episode % config.log_interval == 0 {
            println!(
                "[{} {} {}] Episode {}, Avg Reward: {:.3}, Avg Steps: {}, Stats: {:?}, Elapsed: {:?}",
                config.label,
                algorithm,
                agent_id,
                episode,
                total_reward / config.log_interval as f64,
                total_steps / config.log_interval,
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

fn run_ppo_rnd_worker(
    config: DiscreteEnvConfig,
    port: u16,
    agent_id: usize,
    mut agent: PPO,
    curiosity: Arc<Mutex<RND>>,
    save_curiosity: bool,
) {
    let client = client();
    let base_url = format!("http://localhost:{}", port);
    let start = Instant::now();
    let mut total_reward = 0.0;
    let mut total_intrinsic = 0.0;
    let mut total_steps = 0;

    for episode in 1..=config.episodes {
        let reset = reset(&client, &base_url, config.gym_id);
        let mut observation = reset.observation;
        let initial_state = encode_observation(&observation, config);
        curiosity
            .lock()
            .unwrap()
            .observe(curiosity_experience(&initial_state, false));
        let mut reward = 0.0;

        for current_step in 0..config.max_steps {
            let state = encode_observation(&observation, config);
            let action = agent
                .act_and_train(&state, reward)
                .flatten(0, -1)
                .int64_value(&[0]) as usize;
            let response = step(&client, &base_url, &reset.session_id, action);
            observation = response.observation;
            let next_state = encode_observation(&observation, config);
            let terminal = response.done || current_step + 1 == config.max_steps;
            let intrinsic = intrinsic_reward(&mut curiosity.lock().unwrap(), &next_state, terminal);
            reward = response.reward + config.rnd.intrinsic_reward_scale * intrinsic;
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
                "[{} PPO+RND {}] Episode {}, Avg Reward: {:.3}, Avg Intrinsic: {:.3}, Avg Steps: {}, Elapsed: {:?}",
                config.label,
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
                curiosity.lock().unwrap().save();
            }
        }
    }
    agent.save();
    if save_curiosity {
        curiosity.lock().unwrap().save();
    }
}

fn install_workers<F>(parallel_count: usize, work: F)
where
    F: Fn(usize, u16) + Sync + Send,
{
    let ports = super::environment_ports(parallel_count);
    ThreadPoolBuilder::new()
        .num_threads(parallel_count)
        .build()
        .expect("failed to build discrete training thread pool")
        .install(|| {
            ports
                .into_par_iter()
                .enumerate()
                .for_each(|(agent_id, port)| work(agent_id, port));
        });
}

pub(super) fn train_dqn(
    config: DiscreteEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let replay = Arc::new(ReplayBuffer::new(config.dqn.replay_capacity, 1));
    install_workers(parallel_count, |agent_id, port| {
        let agent = build_dqn(
            config,
            Arc::clone(&replay),
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
        );
        run_worker(config, "DQN", port, agent_id, agent, |observation| {
            encode_observation(observation, config)
        });
    });
}

pub(super) fn train_ppo(
    config: DiscreteEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    install_workers(parallel_count, |agent_id, port| {
        let agent = build_ppo(
            config,
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
        );
        run_worker(config, "PPO", port, agent_id, agent, |observation| {
            encode_observation(observation, config)
        });
    });
}

pub(super) fn train_sac(
    config: DiscreteEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let replay = Arc::new(ReplayBuffer::new(config.sac.replay_capacity, 1));
    install_workers(parallel_count, |agent_id, port| {
        let agent = build_sac(
            config,
            Arc::clone(&replay),
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
        );
        run_worker(config, "SAC", port, agent_id, agent, |observation| {
            encode_observation(observation, config)
        });
    });
}

pub(super) fn train_ppo_rnd(
    config: DiscreteEnvConfig,
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    install_workers(parallel_count, |agent_id, port| {
        let agent_save = super::path_for_agent(&save_path, agent_id);
        let agent_load = super::path_for_agent(&load_path, agent_id);
        let agent = build_ppo(config, agent_save.clone(), agent_load.clone());
        let curiosity = Arc::new(Mutex::new(build_rnd(
            config,
            rnd_path(&agent_save),
            rnd_path(&agent_load),
        )));
        run_ppo_rnd_worker(config, port, agent_id, agent, curiosity, true);
    });
}

pub(super) fn train_ppo_shared_rnd(
    config: DiscreteEnvConfig,
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
        let agent = build_ppo(
            config,
            super::path_for_agent(&save_path, agent_id),
            super::path_for_agent(&load_path, agent_id),
        );
        run_ppo_rnd_worker(
            config,
            port,
            agent_id,
            agent,
            Arc::clone(&curiosity),
            agent_id == 0,
        );
    });
    curiosity.lock().unwrap().save();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(one_hot_state: bool, state_count: usize) -> DiscreteEnvConfig {
        DiscreteEnvConfig {
            label: "test",
            gym_id: "test-v0",
            state_count,
            action_count: 2,
            one_hot_state,
            episodes: 1,
            max_steps: 1,
            log_interval: 1,
            dqn: DqnHyperparameters::default(),
            ppo: PpoHyperparameters::default(),
            sac: SacHyperparameters::default(),
            rnd: RndHyperparameters::default(),
        }
    }

    #[test]
    fn scalar_discrete_observation_is_one_hot_encoded() {
        let state = encode_observation(&[3.0], config(true, 5));
        assert_eq!(state.size(), vec![5]);
        assert_eq!(state.argmax(0, false).int64_value(&[]), 3);
        assert_eq!(state.sum(Kind::Float).double_value(&[]), 1.0);
    }

    #[test]
    fn vector_discrete_observation_is_preserved() {
        let state = encode_observation(&[1.0, 2.0, 3.0, 4.0], config(false, 4));
        assert_eq!(state, Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]));
    }
}
