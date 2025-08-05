use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::FCQNetwork;
use std::time::Instant;

use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
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

fn run_agent_on_env(env_port: u16, agent_id: usize, shared_buffer: Arc<ReplayBuffer>) {
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");

    let base_url = format!("http://localhost:{}", env_port);

    // --- Agent Setup ---
    let device = Device::Cuda(0);
    let vs = nn::VarStore::new(device);
    let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
    let model = Box::new(FCQNetwork::new(vs, 4, 2, 2, 200));
    let explorer = EpsilonGreedy::new(0.5, 0.0, 20000);

    let mut agent = DQN::new(
        model,
        Arc::clone(&shared_buffer),
        optimizer,
        2,
        128,
        16,
        100,
        Box::new(explorer),
        None,
        0.97,
    );

    // --- Training Loop ---
    let episodes = 10000;
    let max_steps = 500;
    let mut total_reward = 0.0;
    let mut total_steps = 0;

    let start = Instant::now();

    for episode in 1..episodes {
        // /reset
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
            let action_tensor = agent.act_and_train(&obs_tensor, reward);
            let action = action_tensor.int64_value(&[]) as usize;

            // /step
            let resp = client
                .post(format!("{}/step", base_url))
                .json(&serde_json::json!({ "session_id": session_id, "action": action }))
                .send()
                .expect("step failed")
                .json::<StepResponse>()
                .expect("step JSON parse failed");

            obs = resp.observation;
            if (step + 1) % 20 == 0 {
                reward = 5.0;
            } else {
                reward = 0.0;
            }
            if resp.done && (max_steps - step) > 10 {
                reward = -30.0;
            }
            total_reward += reward;
            total_steps += 1;

            if resp.done {
                let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agent.stop_episode_and_train(&obs_tensor, reward);
                break;
            }
        }

        if episode % 100 == 0 {
            println!(
                "[Agent {}] Episode {}, Avg Reward: {:.1}, Avg Steps: {}",
                agent_id,
                episode,
                total_reward / 100.0,
                total_steps / 100,
            );
            total_reward = 0.0;
            total_steps = 0;
        }
    }
}

pub fn train_web_cartpole_with_dqn() {
    let shared_buffer = Arc::new(ReplayBuffer::new(300000, 5));
    let ports: Vec<u16> = (8001..=8004).collect();

    ports
        .into_par_iter()
        .enumerate()
        .for_each(|(i, port)| run_agent_on_env(port, i, Arc::clone(&shared_buffer)));
}
