use super::continuous_env::{
    self, ContinuousEnvConfig, PpoHyperparameters, RndHyperparameters, SacHyperparameters,
};

fn config() -> ContinuousEnvConfig {
    ContinuousEnvConfig {
        label: "HalfCheetah-v5",
        gym_id: "HalfCheetah-v5",
        obs_size: 17,
        action_size: 6,
        action_low: -1.0,
        action_high: 1.0,
        episodes: 5_000,
        max_steps: 1_000,
        log_interval: 20,
        ppo_reward_clip: 10.0,
        ppo: PpoHyperparameters {
            learning_rate: 3e-4,
            ..Default::default()
        },
        sac: SacHyperparameters::default(),
        rnd: RndHyperparameters {
            intrinsic_reward_scale: 0.1,
            ..Default::default()
        },
    }
}

pub fn train_half_cheetah_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo(config(), parallel, save, load);
}

pub fn train_half_cheetah_with_ppo_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    continuous_env::train_ppo_rnd(config(), parallel, save, load);
}

pub fn train_half_cheetah_with_ppo_shared_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    continuous_env::train_ppo_shared_rnd(config(), parallel, save, load);
}

pub fn train_half_cheetah_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_sac(config(), parallel, save, load);
}
