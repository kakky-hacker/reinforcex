use std::env;

use examples::train_ant_with_ppo;
use examples::train_web_LunarLander_with_dqn;
use examples::train_web_cartpole_with_dqn;

use crate::examples::train_cartpole_with_ppo;
mod examples;
use tch::Cuda;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut env_value = String::new();
    let mut algo_value = String::new();
    let mut save_path: Option<String> = None;
    let mut load_path: Option<String> = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--env" => {
                if let Some(value) = iter.next() {
                    env_value = value.clone();
                }
            }
            "--algo" => {
                if let Some(value) = iter.next() {
                    algo_value = value.clone();
                }
            }
            "--save-path" => {
                if let Some(value) = iter.next() {
                    save_path = Some(value.clone());
                }
            }
            "--load-path" => {
                if let Some(value) = iter.next() {
                    load_path = Some(value.clone());
                }
            }
            _ => {}
        }
    }

    println!(
        "Environment: {}, Algorithm: {}, Save path: {:?}, Load path: {:?}",
        env_value, algo_value, save_path, load_path
    );

    reinforcex::load_cuda_dlls();

    println!("is_cuda: {}", Cuda::is_available());

    if env_value == "cartpole" && algo_value == "dqn" {
        train_web_cartpole_with_dqn(save_path, load_path);
    } else if env_value == "cartpole" && algo_value == "ppo" {
        train_cartpole_with_ppo(save_path, load_path);
    } else if env_value == "ant" && algo_value == "ppo" {
        train_ant_with_ppo(save_path, load_path);
    } else if env_value == "Lunar" && algo_value == "dqn" {
        //train_web_LunarLander_with_dqn(save_path, load_path).await;
    } else {
        panic!("Invalid env or algo");
    }
}
