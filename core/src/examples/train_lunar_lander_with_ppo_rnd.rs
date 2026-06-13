use reinforcex::agents::{BaseAgent, PPO};
use reinforcex::curiousity::{BaseCuriousity, RND};
use reinforcex::memory::Experience;
use reinforcex::models::{FCRNDModel, FCSoftmaxPolicyWithValue};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use ulid::Ulid;

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

fn curiosity_experience(state: &Tensor, is_episode_terminal: bool) -> Arc<Experience> {
    Arc::new(Experience::new(
        Ulid::new(),
        Ulid::new(),
        state.shallow_clone(),
        None,
        None,
        0.0,
        is_episode_terminal,
        Mutex::new(None),
        Mutex::new(None),
    ))
}

fn calc_intrinsic_reward(curiosity: &RND, state: &Tensor, is_episode_terminal: bool) -> f64 {
    let experience = curiosity_experience(state, is_episode_terminal);
    let reward = curiosity.calc_reward(Arc::clone(&experience));
    reward
        .to_device(Device::Cpu)
        .mean(Kind::Float)
        .double_value(&[])
}

fn observe_curiosity(curiosity: &mut RND, state: &Tensor, is_episode_terminal: bool) {
    curiosity.observe(curiosity_experience(state, is_episode_terminal));
}

fn run_agent_on_env(env_port: u16, agent_id: usize) {
    println!("train_lunar_lander_with_ppo_rnd");

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("HTTP client build failed");

    let base_url = format!("http://localhost:{}", env_port);
    let device = Device::cuda_if_available();

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
        None,
        None,
    );

    let rnd_model = FCRNDModel::new(
        nn::VarStore::new(device),
        nn::VarStore::new(device),
        n_input_channels,
        128,
        n_hidden_layers,
        n_hidden_channels,
    );
    let rnd_optimizer = nn::Adam::default()
        .build(rnd_model.predictor_var_store(), 1e-4)
        .unwrap();
    let mut curiosity = RND::new(Box::new(rnd_model), rnd_optimizer, 128, None, None);
    let intrinsic_reward_scale = 0.01;

    let max_episode = 5000;
    let max_steps = 1000;
    let log_interval = 20;
    let start = Instant::now();
    let mut total_extrinsic_reward = 0.0;
    let mut total_intrinsic_reward = 0.0;
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
        let initial_state = Tensor::from_slice(&obs).to_kind(Kind::Float);
        observe_curiosity(&mut curiosity, &initial_state, false);

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
            let intrinsic_reward = calc_intrinsic_reward(&curiosity, &next_state, resp.done);
            observe_curiosity(&mut curiosity, &next_state, resp.done);

            reward = resp.reward + intrinsic_reward_scale * intrinsic_reward;
            total_extrinsic_reward += resp.reward;
            total_intrinsic_reward += intrinsic_reward;
            total_steps += 1;

            if resp.done {
                agent.stop_episode_and_train(&next_state, reward);
                break;
            }
        }

        if episode % log_interval == 0 {
            println!(
                "[Agent {}] Episode {}, Avg Extrinsic: {:.1}, Avg Intrinsic: {:.3}, Avg Steps: {}, Elapsed: {:?}",
                agent_id,
                episode,
                total_extrinsic_reward / log_interval as f64,
                total_intrinsic_reward / log_interval as f64,
                total_steps / log_interval,
                start.elapsed()
            );
            total_extrinsic_reward = 0.0;
            total_intrinsic_reward = 0.0;
            total_steps = 0;
        }
    }
}

pub fn train_lunar_lander_with_ppo_rnd() {
    let ports: Vec<u16> = (8001..=8001).collect();

    for (i, port) in ports.into_iter().enumerate() {
        run_agent_on_env(port, i);
    }
}
