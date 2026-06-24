use super::continuous_env::{
    self, ContinuousEnvConfig, PpoHyperparameters, RndHyperparameters, SacHyperparameters,
};

fn config() -> ContinuousEnvConfig {
    ContinuousEnvConfig {
        label: "Pusher-v5",
        gym_id: "Pusher-v5",
        obs_size: 23,
        action_size: 7,
        action_low: -2.0,
        action_high: 2.0,
        episodes: 10_000,
        max_steps: 100,
        log_interval: 50,
        ppo_reward_clip: 10.0,
        ppo: PpoHyperparameters {
            update_interval: 1024,
            ..Default::default()
        },
        sac: SacHyperparameters {
            replay_capacity: 100_000,
            replay_start_size: 5000,
            ..Default::default()
        },
        rnd: RndHyperparameters {
            intrinsic_reward_scale: 0.1,
            ..Default::default()
        },
    }
}

pub fn train_pusher_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo(config(), parallel, save, load);
}

pub fn train_pusher_with_ppo_rnd(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo_rnd(config(), parallel, save, load);
}

pub fn train_pusher_with_ppo_shared_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    continuous_env::train_ppo_shared_rnd(config(), parallel, save, load);
}

pub fn train_pusher_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_sac(config(), parallel, save, load);
}
