use gym::client::MakeOptions;
extern crate gym;
use gym::Action;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use rl::models::{FCSoftmaxPolicy, FCSoftmaxPolicyWithValue, BasePolicy};
use rl::agents::{BaseAgent, REINFORCE};

pub fn train_cartpole_with_reinforce() {
    let device: Device = Device::cuda_if_available();
    let vs: nn::VarStore = nn::VarStore::new(device);

    let model: Box<dyn BasePolicy> = Box::new(FCSoftmaxPolicy::new(
        &vs,
        4,
        2,
        2,
        Some(128),
        0.0,
    ));

    let opt: nn::Optimizer = nn::Adam::default().build(&vs, 3e-3).unwrap();

    let mut agent: REINFORCE = REINFORCE::new(
        model,
        opt,
        0.9,
        0.0,
        8,
        false,
        0.9,
        false,
    );

    let gym: gym::client::GymClient = gym::client::GymClient::default();
	let env = gym.make(
			"CartPole-v1",
			Some(MakeOptions {
				render_mode: Some(gym::client::RenderMode::Human),
				..Default::default()
			}),
		)
		.expect("Unable to create environment");

    
    let mut total_reward: f64 = 0.0;
    for episode in 0..1000000 {
        env.reset(None).unwrap();
        let mut reward: f64 = 0.0;
        let mut obs: Vec<f64> = vec![0.0; 4];
        for step in 0..10000 {
            let obs_: Tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_: tch::Tensor;
            action_ = agent.act_and_train(&obs_, reward);
            let state: gym::State = env.step(&Action::Discrete(action_.int64_value(&[]) as usize)).unwrap();
            obs = state.observation.get_box().unwrap().to_vec();
            reward = state.reward / 10.0;
            env.render();
            total_reward += reward;
            if state.is_done {
                let obs_: Tensor = Tensor::from_slice(&obs).to_kind(Kind::Float);
                agent.stop_episode_and_train(&obs_, reward, true);
                break;
            }
        }
        println!("reward:{}", total_reward);
        println!("==============={}===============", episode);
        total_reward = 0.0;
    }
    env.close();
}
