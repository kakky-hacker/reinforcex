use rand::seq::SliceRandom;
use rand::thread_rng;
use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::FCQNetwork;
use reinforcex::selector::{BaseSelector, RewardBasedSelector};
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
    agents: Arc<Vec<Mutex<DQN>>>,
    selector: Arc<Box<dyn BaseSelector>>,
    reward_log_map: Arc<Mutex<HashMap<usize, Vec<f64>>>>,
    is_cuda: bool,
) {
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");

    let base_url = format!("http://localhost:{}", env_port);

    // --- Training Loop ---
    let episodes = 3000;
    let max_steps = 100000;
    let mut total_reward = 0.0;
    let start = Instant::now();
    let mut j = 0;

    let mut flag = false;

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

        for t in 0..max_steps {
            let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_tensor = agents[agent_id]
                .lock()
                .unwrap()
                .act_and_train(&obs_tensor, reward);
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

            let mut prune = false;

            if flag && j % 1000 == 0 {
                let dominants =
                    selector.find_pareto_dominant(agents[agent_id].lock().unwrap().get_agent_id());
                if dominants.len() > 0 {
                    println!("[Agent {}] pruned, dominants:{:?}", agent_id, dominants);
                    selector.delete(agents[agent_id].lock().unwrap().get_agent_id());
                    let mut rng = thread_rng();
                    let d_id = dominants.choose(&mut rng).unwrap();
                    for (k, agent) in agents.iter().enumerate() {
                        if agent.lock().unwrap().get_agent_id() == d_id {
                            println!("[Agent {}] copy {}", agent_id, k);
                            agents[agent_id]
                                .lock()
                                .unwrap()
                                .copy_model_from(&*agent.lock().unwrap());
                            break;
                        }
                    }
                    prune = true;
                }
            }

            if resp.done || prune {
                let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agents[agent_id]
                    .lock()
                    .unwrap()
                    .stop_episode_and_train(&obs_tensor, reward);
                break;
            }

            j += 1;
        }

        if episode % 10 == 0 {
            let avg_reward = total_reward / 10.0;
            if avg_reward > 100.0 {
                flag = true;
            }
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
    let shared_buffer1 = Arc::new(ReplayBuffer::new(36000, 1));
    let shared_buffer2 = Arc::new(ReplayBuffer::new(36000, 1));
    let selector1: Arc<Box<dyn BaseSelector>> =
        Arc::new(Box::new(RewardBasedSelector::new(-1.96, 5000, 5000)));
    let selector2: Arc<Box<dyn BaseSelector>> =
        Arc::new(Box::new(RewardBasedSelector::new(-1.96, 5000, 5000)));

    let ports: Vec<u16> = (8001..=8010).collect();

    let reward_log_map: Arc<Mutex<HashMap<usize, Vec<f64>>>> = Arc::new(Mutex::new(HashMap::new()));

    let mut handles = Vec::new();

    let mut agents: Vec<Mutex<DQN>> = vec![];
    // --- Agent Setup ---
    for i in 0..ports.len() {
        let buffer = if i == 0 {
            Arc::clone(&shared_buffer1)
        } else {
            Arc::clone(&shared_buffer2)
        };

        let selector: Arc<Box<dyn BaseSelector>> = if i == 0 {
            Arc::clone(&selector1)
        } else {
            Arc::clone(&selector2)
        };
        let device;
        if Cuda::is_available() {
            device = Device::Cuda(0);
        } else {
            device = Device::Cpu;
        }
        let vs = nn::VarStore::new(device);
        let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
        let model = Box::new(FCQNetwork::new(vs, 8, 4, 2, 300));

        let explorer = EpsilonGreedy::new(1.0, 0.05, 10000);
        let agent = Mutex::new(DQN::new(
            model,
            buffer,
            optimizer,
            4,
            64,
            8,
            50,
            Box::new(explorer),
            Some(selector.clone()),
            0.99,
        ));
        agents.push(agent);
    }

    let arc_agents = Arc::new(agents);

    for (i, port) in ports.into_iter().enumerate() {
        let selector: Arc<Box<dyn BaseSelector>> = if i == 0 {
            Arc::clone(&selector1)
        } else {
            Arc::clone(&selector2)
        };
        let reward_log_map = Arc::clone(&reward_log_map);

        let agents = Arc::clone(&arc_agents);

        let handle = tokio::spawn(async move {
            run_agent_on_env(
                port,
                i,
                agents,
                selector,
                reward_log_map,
                Cuda::is_available(),
            )
            .await;
        });

        handles.push(handle);
    }

    join_all(handles).await;

    let map = reward_log_map.lock().unwrap();
    let json = serde_json::to_string_pretty(&*map).unwrap();
    fs::write("output.json", json).expect("Unable to write file");
}
