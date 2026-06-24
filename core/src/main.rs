use std::env;

use experiments::train_ant_with_ppo;
use experiments::train_ant_with_ppo_rnd;
use experiments::train_ant_with_ppo_shared_rnd;
use experiments::train_ant_with_sac;
use experiments::train_bipedal_walker_with_ppo;
use experiments::train_bipedal_walker_with_ppo_rnd;
use experiments::train_bipedal_walker_with_ppo_shared_rnd;
use experiments::train_bipedal_walker_with_sac;
use experiments::train_cartpole_with_dqn;
use experiments::train_cartpole_with_sac;
use experiments::train_frozen_lake_with_dqn;
use experiments::train_frozen_lake_with_ppo;
use experiments::train_frozen_lake_with_sac;
use experiments::train_half_cheetah_with_ppo;
use experiments::train_half_cheetah_with_ppo_rnd;
use experiments::train_half_cheetah_with_ppo_shared_rnd;
use experiments::train_half_cheetah_with_sac;
use experiments::train_lunar_lander_with_dqn;
use experiments::train_lunar_lander_with_ppo;
use experiments::train_lunar_lander_with_ppo_rnd;
use experiments::train_lunar_lander_with_ppo_shared_rnd;
use experiments::train_lunar_lander_with_sac;
use experiments::train_pusher_with_ppo;
use experiments::train_pusher_with_ppo_rnd;
use experiments::train_pusher_with_ppo_shared_rnd;
use experiments::train_pusher_with_sac;
use experiments::train_taxi_with_dqn;
use experiments::train_taxi_with_ppo;
use experiments::train_taxi_with_sac;

use crate::experiments::train_cartpole_with_ppo;
mod experiments;
use tch::Cuda;

fn canonical_environment_name(value: &str) -> String {
    let compact = value.to_ascii_lowercase().replace('-', "").replace('_', "");
    let without_version_digits = compact.trim_end_matches(|c: char| c.is_ascii_digit());
    let canonical = without_version_digits
        .strip_suffix('v')
        .unwrap_or(without_version_digits)
        .to_string();
    match canonical.as_str() {
        "lunarlander" => "lunar".to_string(),
        _ => canonical,
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    let mut env_value = String::new();
    let mut algo_value = String::new();
    let mut save_path: Option<String> = None;
    let mut load_path: Option<String> = None;
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
        "Environment: {}, Algorithm: {}, Save path: {:?}, Load path: {:?} Parallel: {}",
        env_value, algo_value, save_path, load_path, parallel_count
    );

    reinforcex::load_cuda_dlls();

    println!("is_cuda: {}", Cuda::is_available());

    let env_key = canonical_environment_name(&env_value);
    let algo_key = algo_value.to_ascii_lowercase();

    if env_key == "cartpole" && algo_key == "dqn" {
        train_cartpole_with_dqn(parallel_count, save_path, load_path);
    } else if env_key == "cartpole" && algo_key == "ppo" {
        train_cartpole_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "ant" && algo_key == "ppo" {
        train_ant_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "ant" && algo_key == "sac" {
        train_ant_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "ant" && algo_key == "ppo-rnd" {
        train_ant_with_ppo_rnd(parallel_count, save_path, load_path);
    } else if env_key == "ant" && algo_key == "ppo-shared-rnd" {
        train_ant_with_ppo_shared_rnd(parallel_count, save_path, load_path);
    } else if env_key == "lunar" && algo_key == "dqn" {
        train_lunar_lander_with_dqn(parallel_count, save_path, load_path);
    } else if env_key == "lunar" && algo_key == "ppo" {
        train_lunar_lander_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "cartpole" && algo_key == "sac" {
        train_cartpole_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "lunar" && algo_key == "sac" {
        train_lunar_lander_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "lunar" && algo_key == "ppo-rnd" {
        train_lunar_lander_with_ppo_rnd(parallel_count, save_path, load_path);
    } else if env_key == "lunar" && algo_key == "ppo-shared-rnd" {
        train_lunar_lander_with_ppo_shared_rnd(parallel_count, save_path, load_path);
    } else if env_key == "taxi" && algo_key == "dqn" {
        train_taxi_with_dqn(parallel_count, save_path, load_path);
    } else if env_key == "taxi" && algo_key == "ppo" {
        train_taxi_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "taxi" && algo_key == "sac" {
        train_taxi_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "frozenlake" && algo_key == "dqn" {
        train_frozen_lake_with_dqn(parallel_count, save_path, load_path);
    } else if env_key == "frozenlake" && algo_key == "ppo" {
        train_frozen_lake_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "frozenlake" && algo_key == "sac" {
        train_frozen_lake_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "halfcheetah" && algo_key == "ppo" {
        train_half_cheetah_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "halfcheetah" && algo_key == "ppo-rnd" {
        train_half_cheetah_with_ppo_rnd(parallel_count, save_path, load_path);
    } else if env_key == "halfcheetah" && algo_key == "ppo-shared-rnd" {
        train_half_cheetah_with_ppo_shared_rnd(parallel_count, save_path, load_path);
    } else if env_key == "halfcheetah" && algo_key == "sac" {
        train_half_cheetah_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "bipedalwalker" && algo_key == "ppo" {
        train_bipedal_walker_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "bipedalwalker" && algo_key == "ppo-rnd" {
        train_bipedal_walker_with_ppo_rnd(parallel_count, save_path, load_path);
    } else if env_key == "bipedalwalker" && algo_key == "ppo-shared-rnd" {
        train_bipedal_walker_with_ppo_shared_rnd(parallel_count, save_path, load_path);
    } else if env_key == "bipedalwalker" && algo_key == "sac" {
        train_bipedal_walker_with_sac(parallel_count, save_path, load_path);
    } else if env_key == "pusher" && algo_key == "ppo" {
        train_pusher_with_ppo(parallel_count, save_path, load_path);
    } else if env_key == "pusher" && algo_key == "ppo-rnd" {
        train_pusher_with_ppo_rnd(parallel_count, save_path, load_path);
    } else if env_key == "pusher" && algo_key == "ppo-shared-rnd" {
        train_pusher_with_ppo_shared_rnd(parallel_count, save_path, load_path);
    } else if env_key == "pusher" && algo_key == "sac" {
        train_pusher_with_sac(parallel_count, save_path, load_path);
    } else {
        panic!("Invalid env or algo");
    }
}

#[cfg(test)]
mod tests {
    use super::canonical_environment_name;

    #[test]
    fn canonical_environment_name_accepts_gymnasium_ids() {
        assert_eq!(canonical_environment_name("Taxi-v3"), "taxi");
        assert_eq!(canonical_environment_name("FrozenLake-v1"), "frozenlake");
        assert_eq!(canonical_environment_name("HalfCheetah-v5"), "halfcheetah");
        assert_eq!(canonical_environment_name("LunarLander-v3"), "lunar");
        assert_eq!(
            canonical_environment_name("BipedalWalker-v3"),
            "bipedalwalker"
        );
    }
}
