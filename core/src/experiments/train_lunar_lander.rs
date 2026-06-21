use super::discrete_env::{
    self, DiscreteEnvConfig, DqnHyperparameters, PpoHyperparameters, RndHyperparameters,
    SacHyperparameters,
};

fn config() -> DiscreteEnvConfig {
    DiscreteEnvConfig {
        label: "LunarLander-v3",
        gym_id: "LunarLander-v3",
        state_count: 8,
        action_count: 4,
        one_hot_state: false,
        episodes: 5_000,
        max_steps: 1_000,
        log_interval: 20,
        dqn: DqnHyperparameters {
            hidden_channels: 300,
            replay_capacity: 36_000,
            update_interval: 8,
            target_update_interval: 50,
            ..Default::default()
        },
        ppo: PpoHyperparameters {
            hidden_channels: 256,
            update_interval: 2048,
            ..Default::default()
        },
        sac: SacHyperparameters {
            hidden_channels: 256,
            replay_capacity: 300_000,
            replay_start_size: 1000,
            batch_size: 256,
            update_interval: 1,
            target_update_interval: 1,
            tau: 0.005,
            ..Default::default()
        },
        rnd: RndHyperparameters {
            intrinsic_reward_scale: 1.0,
            ..Default::default()
        },
    }
}

pub fn train_lunar_lander_with_dqn(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_dqn(config(), parallel, save, load);
}

pub fn train_lunar_lander_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_ppo(config(), parallel, save, load);
}

pub fn train_lunar_lander_with_ppo_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    discrete_env::train_ppo_rnd(config(), parallel, save, load);
}

pub fn train_lunar_lander_with_ppo_shared_rnd(
    parallel: usize,
    save: Option<String>,
    load: Option<String>,
) {
    discrete_env::train_ppo_shared_rnd(config(), parallel, save, load);
}

pub fn train_lunar_lander_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_sac(config(), parallel, save, load);
}
