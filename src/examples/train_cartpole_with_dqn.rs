use gym::client::MakeOptions;
extern crate gym;
use gym::Action;
use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::TransitionBuffer;
use reinforcex::models::FCQNetwork;
use std::sync::Arc;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

pub fn train_cartpole_with_dqn() {
    println!("train_cartpole_with_dqn");

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let n_input_channels = 4;
    let action_size = 2;
    let n_hidden_layers = 2;
    let n_hidden_channels = 200;

    let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
    let model = Box::new(FCQNetwork::new(
        vs,
        n_input_channels,
        action_size,
        n_hidden_layers,
        n_hidden_channels,
    ));
    let explorer = EpsilonGreedy::new(0.5, 0.01, 20000);
    let gamma = 0.97;
    let n_steps = 5;
    let batchsize = 128;
    let update_interval = 16;
    let target_update_interval = 100;
    let replay_buffer_capacity = 20000;

    let transition_buffer = Arc::new(TransitionBuffer::new(replay_buffer_capacity, n_steps));

    let mut agent = DQN::new(
        model,
        transition_buffer,
        optimizer,
        action_size as usize,
        batchsize,
        update_interval,
        target_update_interval,
        Box::new(explorer),
        None,
        gamma,
    );

    let gym = gym::client::GymClient::default();
    let env = gym
        .make(
            "CartPole-v1",
            Some(MakeOptions {
                render_mode: Some(gym::client::RenderMode::Human),
                ..Default::default()
            }),
        )
        .expect("Unable to create environment");

    let mut total_reward = 0.0;
    let mut total_steps = 0;
    let log_interval = 100;
    let max_step = 500;
    let max_episode = 10000;
    for episode in 1..max_episode {
        env.reset(None).unwrap();
        let mut reward = 0.0;
        let mut obs = vec![0.0; 4];
        for step in 1..max_step {
            let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_;
            action_ = agent.act_and_train(&obs_, reward);
            let state = env
                .step(&Action::Discrete(action_.int64_value(&[]) as usize))
                .unwrap();
            obs = state.observation.get_box().unwrap().to_vec();
            if step % 20 == 0 {
                reward = 5.0;
            } else {
                reward = 0.0;
            }
            if state.is_done || step == max_step {
                let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
                if step != max_step {
                    reward = -30.0;
                }
                agent.stop_episode_and_train(&obs_, reward);
                break;
            }
            env.render();
            total_reward += reward;
            total_steps += 1;
        }
        if episode % log_interval == 0 {
            println!(
                "{} episode, average reward:{}, average steps:{}",
                episode,
                total_reward / log_interval as f64,
                total_steps / log_interval,
            );
            total_reward = 0.0;
            total_steps = 0;
        }
    }
    env.close();
}
