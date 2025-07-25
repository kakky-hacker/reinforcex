use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::TransitionBuffer;
use reinforcex::models::FCQNetwork;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Duration;
use tch::{nn, nn::OptimizerConfig, Cuda, Device, Kind, Tensor};

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

async fn run_agent_on_env(
    env_port: u16,
    agent_id: usize,
    shared_buffer: Arc<TransitionBuffer>,
    reward_log_map: Arc<Mutex<HashMap<usize, Vec<f64>>>>,
    is_cuda: bool,
) {
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");

    let base_url = format!("http://localhost:{}", env_port);

    // --- Agent Setup ---
    let device;
    if is_cuda {
        device = Device::Cuda(0);
    } else {
        device = Device::Cpu;
    }
    let vs = nn::VarStore::new(device);
    let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
    let model = Box::new(FCQNetwork::new(vs, 8, 4, 2, 300));

    let explorer = EpsilonGreedy::new(1.0, 0.05, 10000);
    let mut agent = DQN::new(
        model,
        Arc::clone(&shared_buffer),
        optimizer,
        4,
        64,
        8,
        50,
        Box::new(explorer),
        0.99,
    );

    // --- Training Loop ---
    let episodes = 1000;
    let max_steps = 100000;
    let mut total_reward = 0.0;
    let start = Instant::now();

    for episode in 1..=episodes {
        // /reset
        let resp = client
            .post(format!("{}/reset", base_url))
            .json(&serde_json::json!({ "env": "LunarLander-v3" }))
            .send()
            .await
            .expect("reset failed")
            .json::<ResetResponse>()
            .await
            .expect("reset JSON parse failed");

        let mut obs = resp.observation;
        let session_id = resp.session_id;
        let mut reward = 0.0;

        for _ in 0..max_steps {
            let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_tensor = agent.act_and_train(&obs_tensor, reward);
            let action = action_tensor.int64_value(&[]) as usize;

            // /step
            let resp = client
                .post(format!("{}/step", base_url))
                .json(&serde_json::json!({ "session_id": session_id, "action": action }))
                .send()
                .await
                .expect("step failed")
                .json::<StepResponse>()
                .await
                .expect("step JSON parse failed");

            obs = resp.observation;
            reward = resp.reward;
            total_reward += reward;

            if resp.done {
                let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agent.stop_episode_and_train(&obs_tensor, reward);
                break;
            }
        }

        if episode % 10 == 0 {
            let avg_reward = total_reward / 10.0;
            println!(
                "[Agent {}] Episode {}, Avg Reward: {:.1}, Elapsed time: {:?}",
                agent_id,
                episode,
                avg_reward,
                start.elapsed()
            );

            // ログ追加
            {
                let mut map = reward_log_map.lock().unwrap();
                map.entry(agent_id)
                    .or_insert_with(Vec::new)
                    .push(avg_reward);
            }

            total_reward = 0.0;
        }
    }
}

pub async fn train_web_LunarLander_with_dqn() {
    let shared_buffer1 = Arc::new(TransitionBuffer::new(4000, 1));
    let shared_buffer2 = Arc::new(TransitionBuffer::new(36000, 1));
    let ports: Vec<u16> = (8001..=8010).collect();

    let reward_log_map: Arc<Mutex<HashMap<usize, Vec<f64>>>> = Arc::new(Mutex::new(HashMap::new()));

    let mut handles = Vec::new();

    for (i, port) in ports.into_iter().enumerate() {
        let buffer = if i == 0 {
            Arc::clone(&shared_buffer1)
        } else {
            Arc::clone(&shared_buffer2)
        };

        let reward_log_map = Arc::clone(&reward_log_map);

        let handle = tokio::spawn(async move {
            run_agent_on_env(port, i, buffer, reward_log_map, Cuda::is_available()).await;
        });

        handles.push(handle);
    }

    join_all(handles).await;

    let map = reward_log_map.lock().unwrap();
    let json = serde_json::to_string_pretty(&*map).unwrap();
    fs::write("output.json", json).expect("Unable to write file");
}
