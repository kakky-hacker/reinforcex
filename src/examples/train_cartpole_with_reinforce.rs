use gym::client::MakeOptions;
extern crate gym;
use gym::Action;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use rl::models::{FCSoftmaxPolicy, FCSoftmaxPolicyWithValue, BasePolicy};
use rl::agents::{BaseAgent, REINFORCE};

pub fn train_cartpole_with_reinforce() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let n_input_channels = 4;
    let n_actions = 2;
    let n_hidden_layers = 2;
    let n_hidden_channels = Some(128);
    let min_prob = 0.0;

    let model = Box::new(FCSoftmaxPolicy::new(
        &vs,
        n_input_channels,
        n_actions,
        n_hidden_layers,
        n_hidden_channels,
        min_prob,
    ));

    let optimizer = nn::Adam::default().build(&vs, 3e-3).unwrap();
    let gamma = 0.9;
    let beta = 0.0;
    let batchsize = 8;
    let act_deterministically = false;
    let average_entropy_decay = 0.9;
    let backward_separately = false;

    let mut agent = REINFORCE::new(
        model,
        optimizer,
        gamma,
        beta,
        batchsize,
        act_deterministically,
        average_entropy_decay,
        backward_separately,
    );

    let gym = gym::client::GymClient::default();
	let env = gym.make(
			"CartPole-v1",
			Some(MakeOptions {
				render_mode: Some(gym::client::RenderMode::Human),
				..Default::default()
			}),
		)
		.expect("Unable to create environment");

    
    let mut total_reward = 0.0;
    for episode in 0..1000000 {
        env.reset(None).unwrap();
        let mut reward = 0.0;
        let mut obs = vec![0.0; 4];
        for step in 0..10000 {
            let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_;
            action_ = agent.act_and_train(&obs_, reward);
            let state = env.step(&Action::Discrete(action_.int64_value(&[]) as usize)).unwrap();
            obs = state.observation.get_box().unwrap().to_vec();
            reward = state.reward / 10.0;
            env.render();
            total_reward += reward;
            if state.is_done {
                let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agent.stop_episode_and_train(&obs_, reward);
                break;
            }
        }
        println!("reward:{}", total_reward);
        println!("==============={}===============", episode);
        total_reward = 0.0;
    }
    env.close();
}
