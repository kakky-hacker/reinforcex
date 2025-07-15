use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::TransitionBuffer;
use reinforcex::models::FCQNetwork;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use tokio;
use std::fs;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use envserver::env_service_client::EnvServiceClient;
use envserver::{ResetRequest, StepRequest};
use tonic::transport::Channel;
use futures::future::join_all;


mod envserver {
    tonic::include_proto!("envserver");
}

async fn run_agent_on_env_grpc(
    port: u16,
    agent_id: usize,
    shared_buffer: Arc<TransitionBuffer>,
    reward_log_map: Arc<Mutex<HashMap<usize, Vec<f64>>>>,
) {
    let addr = format!("http://127.0.0.1:{}", port);
    let mut client = EnvServiceClient::connect(addr).await.unwrap();

    // setup...
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let model = Box::new(FCQNetwork::new(&vs, 8, 4, 2, Some(300)));
    let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
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

    let episodes = 1000;
    let max_steps = 100000;
    let mut total_reward = 0.0;
    let start = Instant::now();

    for episode in 1..=episodes {
        let response = client
            .reset(ResetRequest {
                env: "LunarLander-v3".to_string(),
            })
            .await
            .unwrap()
            .into_inner();

        let session_id = response.session_id;
        let mut obs = response.observation;
        let mut reward = 0.0;

        for _ in 0..max_steps {
            let obs_tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_tensor = agent.act_and_train(&obs_tensor, reward);
            let action = action_tensor.int64_value(&[]) as i32;

            let step_resp = client
                .step(StepRequest {
                    session_id: session_id.clone(),
                    action,
                })
                .await
                .unwrap()
                .into_inner();

            obs = step_resp.observation;
            reward = step_resp.reward as f64;
            total_reward += reward;

            if step_resp.done {
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

            let mut map = reward_log_map.lock().unwrap();
            map.entry(agent_id)
                .or_insert_with(Vec::new)
                .push(avg_reward);
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
            run_agent_on_env_grpc(port, i, buffer, reward_log_map).await;
        });

        handles.push(handle);
    }

    // 全タスク完了待ち
    join_all(handles).await;

    // 結果のログ保存
    let map = reward_log_map.lock().unwrap();
    let json = serde_json::to_string_pretty(&*map).unwrap();
    fs::write("output.json", json).expect("Unable to write file");
}
