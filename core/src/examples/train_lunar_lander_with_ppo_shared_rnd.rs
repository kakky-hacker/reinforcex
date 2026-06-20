use super::train_lunar_lander_with_ppo_rnd::{build_curiosity, run_agent_on_env};
use rayon::prelude::*;
use reinforcex::curiousity::BaseCuriousity;
use std::sync::{Arc, Mutex};
use tch::Device;

fn shared_curiosity_checkpoint_path(path: &Option<String>) -> Option<String> {
    path.as_ref()
        .map(|path| format!("{}.rnd", path.replace("{agent_id}", "shared")))
}

pub fn train_lunar_lander_with_ppo_shared_rnd(
    parallel_count: usize,
    save_path: Option<String>,
    load_path: Option<String>,
) {
    let curiosity = Arc::new(Mutex::new(build_curiosity(
        Device::cuda_if_available(),
        shared_curiosity_checkpoint_path(&save_path),
        shared_curiosity_checkpoint_path(&load_path),
    )));
    let ports = super::environment_ports(parallel_count);

    ports.into_par_iter().enumerate().for_each(|(i, port)| {
        run_agent_on_env(
            port,
            i,
            super::path_for_agent(&save_path, i),
            super::path_for_agent(&load_path, i),
            Arc::clone(&curiosity),
            i == 0,
        )
    });

    curiosity.lock().unwrap().save();
}
