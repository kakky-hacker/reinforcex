use gym::client::MakeOptions;
extern crate gym;
use candle_core::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use gym::Action;
use ndarray::Array1;
use reinforcex::agents::{BaseAgent, REINFORCE};
use reinforcex::models::FCGaussianPolicy;

pub fn train_mountaincar_with_reinforce() {
    println!("train_mountaincar_with_reinforce");

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let n_input_channels = 2;
    let n_actions = 1;
    let n_hidden_layers = 2;
    let n_hidden_channels = Some(128);
    let min_action = Some(Tensor::from_slice(&[-1.0]));
    let max_action = Some(Tensor::from_slice(&[1.0]));
    let bound_mean = true;
    let var_type = "spherical";
    let min_var = 0.1;

    let model = Box::new(FCGaussianPolicy::new(
        &vs,
        n_input_channels,
        n_actions,
        n_hidden_layers,
        n_hidden_channels,
        min_action,
        max_action,
        bound_mean,
        var_type,
        min_var,
    ));

    let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
    let gamma = 0.997;
    let beta = 0.01;
    let batchsize = 1;
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
    let env = gym
        .make(
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
            let obs_ = Tensor::from_slice(&obs).to_kind(DType::F32);
            let action_;
            action_ = agent.act_and_train(&obs_, reward);
            let array = Array1::from(Vec::<f64>::try_from(action_.view(-1)).unwrap());
            let state = env.step(&Action::Box(array)).unwrap();
            obs = state.observation.get_box().unwrap().to_vec();
            reward = state.reward;
            env.render();
            total_reward += reward;
            if state.is_done || step == 1500 {
                let obs_ = Tensor::from_slice(&obs).to_kind(DType::F32);
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
