use gym::client::MakeOptions;
extern crate gym;
use gym::Action;
use rl::agents::{BaseAgent, DQN};
use rl::explorers::EpsilonGreedy;
use rl::models::FCQNetwork;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

pub fn train_cartpole_with_dqn() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let n_input_channels = 4;
    let action_size = 2;
    let n_hidden_layers = 2;
    let n_hidden_channels = Some(128);

    let model = Box::new(FCQNetwork::new(
        &vs,
        n_input_channels,
        action_size,
        n_hidden_layers,
        n_hidden_channels,
    ));

    let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
    let explorer = EpsilonGreedy::new(0.5, 0.1, 50000);
    let gamma = 0.97;
    let n_steps = 3;
    let batchsize = 16;
    let update_interval = 8;
    let target_update_interval = 100;

    let mut agent = DQN::new(
        model,
        optimizer,
        action_size as usize,
        batchsize,
        2000,
        update_interval,
        target_update_interval,
        Box::new(explorer),
        gamma,
        n_steps,
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
    for episode in 1..1000000 {
        env.reset(None).unwrap();
        let mut reward = 0.0;
        let mut obs = vec![0.0; 4];
        for step in 0..10000 {
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
            env.render();
            total_reward += reward;
            if state.is_done {
                let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agent.stop_episode_and_train(&obs_, -30.0);
                break;
            }
        }
        if episode % 100 == 0 {
            println!(
                "{} episode, average reward:{}",
                episode,
                total_reward / 100 as f64
            );
            total_reward = 0.0;
        }
    }
    env.close();
}
