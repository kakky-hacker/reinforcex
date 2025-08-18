use reinforcex::agents::{BaseAgent, PPO};
use reinforcex::memory::OnPolicyBuffer;
use reinforcex::models::FCSoftmaxPolicyWithValue;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use std::time::Instant;

use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::time::Duration;

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

fn run_agent_on_env(env_port: u16, agent_id: usize) {
    println!("train_cartpole_with_ppo_api");

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");

    let base_url = format!("http://localhost:{}", env_port);

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let n_input_channels = 4;
    let action_size = 2;
    let n_hidden_layers = 2;
    let n_hidden_channels = 64;
    let min_prob = 0.0;

    let optimizer = nn::Adam::default().build(&vs, 2.5e-4).unwrap();
    let model = Box::new(FCSoftmaxPolicyWithValue::new(
        vs,
        n_input_channels,
        action_size,
        n_hidden_layers,
        n_hidden_channels,
        min_prob,
    ));

    let gamma = 0.99;
    let lambda = 0.95;
    let epoch = 10;
    let minibatch_size = 32;
    let update_interval = 500;
    let policy_clip_epsilon = 0.2;
    let value_clip_range = 0.2;
    let value_coef = 0.5;
    let entropy_coef = 0.0;

    let buffer = OnPolicyBuffer::new(None);

    let mut agent = PPO::new(
        model,
        optimizer,
        buffer,
        gamma,
        lambda,
        update_interval,
        epoch,
        minibatch_size,
        policy_clip_epsilon,
        value_clip_range,
        value_coef,
        entropy_coef,
        true,
    );
    let mut total_reward = 0.0;
    let mut total_steps = 0;
    let log_interval = 100;
    let max_episode = 10000;

    let start = Instant::now();

    let max_steps = 500;
    for episode in 1..max_episode {
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

pub fn train_cartpole_with_ppo() {
    let ports: Vec<u16> = (8001..=8001).collect();

    ports
        .into_par_iter()
        .enumerate()
        .for_each(|(i, port)| run_agent_on_env(port, i));
}
