use rayon::prelude::*;
use reinforcex::agents::{BaseAgent, PPO};
use reinforcex::curiosity::{Basecuriosity, RND};
use reinforcex::models::{FCRNDModel, FCSoftmaxPolicyWithValue};
use reqwest::blocking::Client;
use serde::Deserialize;
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

fn build_ppo_agent<C: Basecuriosity + 'static>(
    device: Device,
    save_path: Option<String>,
    load_path: Option<String>,
    curiosity: C,
) -> PPO {
    let n_input_channels = 8;
    let action_size = 4;
    let n_hidden_layers = 2;
    let n_hidden_channels = 256;

    let policy_vs = nn::VarStore::new(device);
    let policy_optimizer = nn::Adam::default().build(&policy_vs, 2.5e-4).unwrap();
    let policy_model = Box::new(FCSoftmaxPolicyWithValue::new(
        policy_vs,
        n_input_channels,
        action_size,
        n_hidden_layers,
        n_hidden_channels,
        0.0,
    ));

    let gamma = 0.99;
    let lambda = 0.95;
    let update_interval = 2048;
    let epoch = 10;
    let minibatch_size = 64;
    let policy_clip_epsilon = 0.2;
    let value_clip_range = 0.2;
    let value_coef = 0.5;
    let entropy_coef = 0.01;

    let mut agent = PPO::new(
        policy_model,
        policy_optimizer,
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
        save_path,
        load_path,
    );
    agent.add_curiosity(curiosity, 1.0);
    agent
}

pub(super) fn build_curiosity(
    device: Device,
    save_path: Option<String>,
    load_path: Option<String>,
) -> RND {
    let rnd_model = FCRNDModel::new(
        nn::VarStore::new(device),
        nn::VarStore::new(device),
        8,
        128,
        2,
        256,
    );
    let rnd_optimizer = nn::Adam::default()
        .build(rnd_model.predictor_var_store(), 1e-4)
        .unwrap();
    RND::new(
        Box::new(rnd_model),
        rnd_optimizer,
        128,
        save_path,
        load_path,
    )
}

fn curiosity_checkpoint_path(path: &Option<String>) -> Option<String> {
    path.as_ref().map(|path| format!("{}.rnd", path))
}

pub(super) fn run_agent_on_env<C: Basecuriosity + 'static>(
    env_port: u16,
    agent_id: usize,
    save_path: Option<String>,
    load_path: Option<String>,
    curiosity: C,
) {
    println!("train_lunar_lander_with_ppo_rnd");

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");

    let base_url = format!("http://localhost:{}", env_port);
    let device = Device::cuda_if_available();
    let mut agent = build_ppo_agent(device, save_path, load_path, curiosity);

    let max_episode = 5000;
    let max_steps = 1000;
    let log_interval = 20;
    let start = Instant::now();
    let mut total_extrinsic_reward = 0.0;
    let mut total_steps = 0;

    for episode in 1..=max_episode {
        let resp = client
            .post(format!("{}/reset", base_url))
            .json(&serde_json::json!({ "env": "LunarLander-v3" }))
            .send()
            .expect("reset failed")
            .json::<ResetResponse>()
            .expect("reset JSON parse failed");

        let mut obs = resp.observation;
        let session_id = resp.session_id;
        let mut reward = 0.0;

        for _step in 0..max_steps {
            let state = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_tensor = agent.act_and_train(&state, reward);
            let action = action_tensor.int64_value(&[]) as usize;

            let resp = client
                .post(format!("{}/step", base_url))
                .json(&serde_json::json!({ "session_id": session_id, "action": action }))
                .send()
                .expect("step failed")
                .json::<StepResponse>()
                .expect("step JSON parse failed");

            obs = resp.observation;
            let next_state = Tensor::from_slice(&obs).to_kind(Kind::Float);
            reward = resp.reward;
            total_extrinsic_reward += resp.reward;
            total_steps += 1;

            if resp.done {
                agent.stop_episode_and_train(&next_state, reward);
                break;
            }
        }

        if episode % log_interval == 0 {
            println!(
                "[Agent {}] Episode {}, Avg Extrinsic: {:.1}, Avg Steps: {}, Elapsed: {:?}",
                agent_id,
                episode,
                total_extrinsic_reward / log_interval as f64,
                total_steps / log_interval,
                start.elapsed()
            );
            total_extrinsic_reward = 0.0;
            total_steps = 0;
            agent.save();
        }
    }

    agent.save();
}

pub fn train_lunar_lander_with_ppo_rnd(
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let ports = super::environment_ports(parallel_count);

    ports.into_par_iter().enumerate().for_each(|(i, port)| {
        let save_path = super::path_for_agent(&save_path, i);
        let load_path = super::path_for_agent(&load_path, i);
        let curiosity = build_curiosity(
            Device::cuda_if_available(),
            curiosity_checkpoint_path(&save_path),
            curiosity_checkpoint_path(&load_path),
        );
        run_agent_on_env(port, i, save_path, load_path, curiosity)
    });
}
