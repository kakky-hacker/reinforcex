use super::continuous_env::{
    self, ContinuousEnvConfig, PpoHyperparameters, RndHyperparameters, SacHyperparameters,
};

fn config() -> ContinuousEnvConfig {
    ContinuousEnvConfig {
        label: "Ant-v5",
        gym_id: "Ant-v5",
        obs_size: 105,
        action_size: 8,
        action_low: -1.0,
        action_high: 1.0,
        episodes: 10_000,
        max_steps: 1_000,
        log_interval: 50,
        ppo_reward_clip: 1.0,
        ppo: PpoHyperparameters {
            lambda: 0.99,
            update_interval: 512,
            minibatch_size: 32,
            value_coefficient: 0.003,
            ..Default::default()
        },
        sac: SacHyperparameters::default(),
        rnd: RndHyperparameters {
            intrinsic_reward_scale: 1.0,
            ..Default::default()
        },
    }
}

pub fn train_ant_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo(config(), parallel, save, load);
}

pub fn train_ant_with_ppo_rnd(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo_rnd(config(), parallel, save, load);
}

pub fn train_ant_with_ppo_shared_rnd(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_ppo_shared_rnd(config(), parallel, save, load);
}

pub fn train_ant_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    continuous_env::train_sac(config(), parallel, save, load);
}
