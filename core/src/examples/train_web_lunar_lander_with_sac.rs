use rayon::prelude::*;
use reinforcex::agents::{BaseAgent, SAC};
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::{FCGaussianPolicy, FCQNetwork};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

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

fn build_sac_agent(
    shared_buffer: Arc<ReplayBuffer>,
    save_path: Option<String>,
    load_path: Option<String>,
) -> SAC {
    let device = Device::cuda_if_available();
    let obs_size = 8;
    let action_size = 2;
    let n_hidden_layers = 2;
    let n_hidden_channels = 256;

    let actor_vs = nn::VarStore::new(device);
    let actor_optimizer = nn::Adam::default().build(&actor_vs, 3e-4).unwrap();
    let actor = FCGaussianPolicy::new(
        actor_vs,
        obs_size,
        action_size,
        n_hidden_layers,
        n_hidden_channels,
        None,
        None,
        false,
        "diagonal",
        1e-3,
    );

    let critic1_vs = nn::VarStore::new(device);
    let critic1_optimizer = nn::Adam::default().build(&critic1_vs, 3e-4).unwrap();
    let critic1 = FCQNetwork::new(
        critic1_vs,
        obs_size + action_size,
        1,
        n_hidden_layers,
        n_hidden_channels,
    );

    let critic2_vs = nn::VarStore::new(device);
    let critic2_optimizer = nn::Adam::default().build(&critic2_vs, 3e-4).unwrap();
    let critic2 = FCQNetwork::new(
        critic2_vs,
        obs_size + action_size,
        1,
        n_hidden_layers,
        n_hidden_channels,
    );

    SAC::new_with_save_load(
        Box::new(actor),
        actor_optimizer,
        Box::new(critic1),
        critic1_optimizer,
        Box::new(critic2),
        critic2_optimizer,
        shared_buffer,
        1000,
        256,
        1,
        1,
        0.99,
        0.005,
        0.2,
        true,
        save_path,
        load_path,
    )
}

fn run_agent_on_env(
    env_port: u16,
    agent_id: usize,
    shared_buffer: Arc<ReplayBuffer>,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");
    let base_url = format!("http://localhost:{}", env_port);

    let mut agent = build_sac_agent(shared_buffer, save_path, load_path);
    let episodes = 5000;
    let max_steps = 1000;
    let log_interval = 10;
    let mut total_reward = 0.0;
    let mut total_steps = 0;
    let start = Instant::now();

    for episode in 1..=episodes {
        let resp = client
            .post(format!("{}/reset", base_url))
            .json(&serde_json::json!({ "env": "LunarLanderContinuous-v3" }))
            .send()
            .expect("reset failed")
            .json::<ResetResponse>()
            .expect("reset JSON parse failed");

        let mut obs = resp.observation;
        let session_id = resp.session_id;
        let mut reward = 0.0;

        for _ in 0..max_steps {
            let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_tensor = agent.act_and_train(&obs_tensor, reward).flatten(0, -1);
            let action = (0..action_tensor.size()[0])
                .map(|i| action_tensor.double_value(&[i]) as f32)
                .collect::<Vec<f32>>();

            let resp = client
                .post(format!("{}/step", base_url))
                .json(&serde_json::json!({ "session_id": session_id, "action": action }))
                .send()
                .expect("step failed")
                .json::<StepResponse>()
                .expect("step JSON parse failed");

            obs = resp.observation;
            reward = resp.reward;
            total_reward += resp.reward;
            total_steps += 1;

            if resp.done {
                let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agent.stop_episode_and_train(&obs_tensor, reward);
                break;
            }
        }

        if episode % log_interval == 0 {
            let statistics = agent.get_statistics();
            println!(
                "[Agent {}] Episode {}, Avg Reward: {:.1}, Avg Steps: {}, Stats: {:?}, Elapsed: {:?}",
                agent_id,
                episode,
                total_reward / log_interval as f64,
                total_steps / log_interval,
                statistics,
                start.elapsed()
            );
            total_reward = 0.0;
            total_steps = 0;
            agent.save();
        }
    }
}

pub fn train_web_lunar_lander_with_sac(
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    assert!(parallel_count > 0);

    let shared_buffer = Arc::new(ReplayBuffer::new(300000, 1));
    let ports = (0..parallel_count)
        .map(|i| {
            8001u16
                .checked_add(u16::try_from(i).expect("parallel count is too large"))
                .expect("parallel count is too large")
        })
        .collect::<Vec<u16>>();

    ports.into_par_iter().enumerate().for_each(|(i, port)| {
        run_agent_on_env(
            port,
            i,
            Arc::clone(&shared_buffer),
            super::path_for_agent(&save_path, i),
            super::path_for_agent(&load_path, i),
        )
    });
}
