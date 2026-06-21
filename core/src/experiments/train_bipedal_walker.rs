use super::continuous_env::{
    self, ContinuousEnvConfig, PpoHyperparameters, RndHyperparameters, SacHyperparameters,
};

fn config() -> ContinuousEnvConfig {
    ContinuousEnvConfig {
        label: "BipedalWalker-v3",
        gym_id: "BipedalWalker-v3",
        obs_size: 24,
        action_size: 4,
        action_low: -1.0,
        action_high: 1.0,
        episodes: 5_000,
        max_steps: 1_600,
        log_interval: 20,
        ppo_reward_clip: 10.0,
        ppo: PpoHyperparameters {
            learning_rate: 3e-4,
            entropy_coefficient: 0.001,
            ..Default::default()
        },
        sac: SacHyperparameters {
            replay_start_size: 5000,
            ..Default::default()
        },
        rnd: RndHyperparameters {
            intrinsic_reward_scale: 0.1,
            ..Default::default()
        },
    }
}

pub fn train_bipedal_walker_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo(config(), parallel, save, load);
}

pub fn train_bipedal_walker_with_ppo_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    continuous_env::train_ppo_rnd(config(), parallel, save, load);
}

pub fn train_bipedal_walker_with_ppo_shared_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    continuous_env::train_ppo_shared_rnd(config(), parallel, save, load);
}

pub fn train_bipedal_walker_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_sac(config(), parallel, save, load);
}
