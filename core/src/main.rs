use std::env;

use examples::train_ant_with_ppo;
use examples::train_web_cartpole_with_dqn;
use examples::train_web_cartpole_with_sac;
use examples::train_web_lunar_lander_with_sac;

use crate::examples::train_cartpole_with_ppo;
mod examples;
use tch::Cuda;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut env_value = String::new();
    let mut algo_value = String::new();
    let mut parallel_count = 4usize;

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
            "--parallel" | "--parallel-count" | "--num-agents" => {
                if let Some(value) = iter.next() {
                    parallel_count = value
                        .parse::<usize>()
                        .expect("parallel count must be a positive integer");
                }
            }
            _ => {}
        }
    }

    assert!(parallel_count > 0, "parallel count must be positive");

    println!(
        "Environment: {}, Algorithm: {}, Parallel: {}",
        env_value, algo_value, parallel_count
    );

    reinforcex::load_cuda_dlls();

    println!("is_cuda: {}", Cuda::is_available());

    let env_key = env_value.to_ascii_lowercase();
    let algo_key = algo_value.to_ascii_lowercase();

    if env_key == "cartpole" && algo_key == "dqn" {
        train_web_cartpole_with_dqn(parallel_count);
    } else if env_key == "cartpole" && algo_key == "ppo" {
        train_cartpole_with_ppo();
    } else if env_key == "cartpole" && algo_key == "sac" {
        train_web_cartpole_with_sac(parallel_count);
    } else if env_key == "ant" && algo_key == "ppo" {
        train_ant_with_ppo();
    } else if env_key == "lunar" && algo_key == "dqn" {
        //train_web_LunarLander_with_dqn().await;
    } else if (env_key == "lunar" || env_key == "lunarlander") && algo_key == "sac" {
        train_web_lunar_lander_with_sac(parallel_count);
    } else {
        panic!("Invalid env or algo");
    }
}
