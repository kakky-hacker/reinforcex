use gym::client::MakeOptions;
extern crate gym;
use gym::Action;
use ndarray::Array1;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use rl::models::{FCGaussianPolicyWithValue, FCGaussianPolicy, BasePolicy};
use rl::agents::{BaseAgent, REINFORCE};

pub fn train_reinforce() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let model = Box::new(FCGaussianPolicy::new(
        &vs,
        2,
        1,
        2,
        Some(128),
        Some(Tensor::from_slice(&[-1.0])),
        Some(Tensor::from_slice(&[1.0])),
        true,
        "spherical",
        0.1,
    ));

    let opt = nn::Adam::default().build(&vs, 3e-4).unwrap();

    let mut agent = REINFORCE::new(
        model,
        opt,
        0.997,
        0.01,
        1,
        false,
        0.9,
        false,
    );

    let gym = gym::client::GymClient::default();
	let env = gym.make(
			"MountainCarContinuous-v0",
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
        let mut obs = vec![0.0; 2];
        for step in 0..10000 {
            let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
            let action_;
            action_ = agent.act_and_train(&obs_, reward);
            let array = Array1::from(Vec::<f64>::try_from(action_.view(-1)).unwrap());
            let state = env.step(&Action::Box(array)).unwrap();
            obs = state.observation.get_box().unwrap().to_vec();
            reward = state.reward;
            env.render();
            total_reward += reward;
            if state.is_done || step == 1500 {
                let obs_ = Tensor::from_slice(&obs).to_kind(Kind::Float);
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
