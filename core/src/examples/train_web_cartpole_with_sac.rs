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

fn build_sac_agent() -> SAC {
    let device = Device::cuda_if_available();
    let obs_size = 4;
    let action_size = 1;
    let n_hidden_layers = 2;
    let n_hidden_channels = 128;

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

    SAC::new(
        Box::new(actor),
        actor_optimizer,
        Box::new(critic1),
        critic1_optimizer,
        Box::new(critic2),
        critic2_optimizer,
        Arc::new(ReplayBuffer::new(100000, 1)),
        128,
        1,
        0.99,
        0.005,
        0.2,
        true,
    )
}

pub fn train_web_cartpole_with_sac() {
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");
    let base_url = "http://localhost:8001";

    let mut agent = build_sac_agent();
    let episodes = 10000;
    let max_steps = 500;
    let log_interval = 100;
    let mut total_reward = 0.0;
    let mut total_steps = 0;
    let start = Instant::now();

    for episode in 1..=episodes {
        let resp = client
            .post(format!("{}/reset", base_url))
            .json(&serde_json::json!({ "env": "CartPole-v1" }))
            .send()
            .expect("reset failed")
            .json::<ResetResponse>()
            .expect("reset JSON parse failed");

        let mut obs = resp.observation;
        let session_id = resp.session_id;
        let mut reward = 0.0;

        for step in 0..max_steps {
            let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_tensor = agent.act_and_train(&obs_tensor, reward).flatten(0, -1);
            let continuous_action = action_tensor.double_value(&[0]);
            let env_action = if continuous_action >= 0.0 { 1 } else { 0 };

            let resp = client
                .post(format!("{}/step", base_url))
                .json(&serde_json::json!({ "session_id": session_id, "action": env_action }))
                .send()
                .expect("step failed")
                .json::<StepResponse>()
                .expect("step JSON parse failed");

            obs = resp.observation;
            reward = if (step + 1) % 20 == 0 { 5.0 } else { 0.0 };
            if resp.done && (max_steps - step) > 10 {
                reward = -30.0;
            }
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
                "Episode {}, Avg Reward: {:.1}, Avg Steps: {}, Stats: {:?}, Elapsed: {:?}",
                episode,
                total_reward / log_interval as f64,
                total_steps / log_interval,
                statistics,
                start.elapsed()
            );
            total_reward = 0.0;
            total_steps = 0;
        }
    }
}
