use std::env;

//use examples::train_cartpole_with_dqn;
use examples::train_web_LunarLander_with_dqn;
//use examples::train_web_cartpole_with_dqn;

//use crate::examples::train_cartpole_with_ppo;
mod examples;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    let mut env_value = String::new();
    let mut algo_value = String::new();

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
            _ => {}
        }
    }

    println!("Environment: {}, Algorithm: {}", env_value, algo_value);

    if env_value == "cartpole" && algo_value == "dqn" {
        //train_web_cartpole_with_dqn();
    } else if env_value == "cartpole" && algo_value == "ppo" {
        //train_cartpole_with_ppo();
    } else if env_value == "Lunar" && algo_value == "dqn" {
        train_web_LunarLander_with_dqn().await;
    } else {
        panic!("Invalid env or algo");
    }
}
