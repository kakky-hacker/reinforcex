use std::env;

use examples::{
    train_cartpole_with_dqn, //train_cartpole_with_reinforce, train_mountaincar_with_reinforce,
};
mod examples;

fn main() {
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
        train_cartpole_with_dqn();
    } else if env_value == "cartpole" && algo_value == "reinforce" {
        //train_cartpole_with_reinforce();
    } else if env_value == "mountaincar" && algo_value == "reinforce" {
        //train_mountaincar_with_reinforce();
    } else {
        panic!("Invalid env or algo");
    }
}
