use super::discrete_env::{
    self, DiscreteEnvConfig, DqnHyperparameters, PpoHyperparameters, RndHyperparameters,
    SacHyperparameters,
};

fn config() -> DiscreteEnvConfig {
    DiscreteEnvConfig {
        label: "FrozenLake-v1",
        gym_id: "FrozenLake-v1",
        state_count: 16,
        action_count: 4,
        one_hot_state: true,
        episodes: 20_000,
        max_steps: 100,
        log_interval: 200,
        dqn: DqnHyperparameters {
            epsilon_decay_steps: 100_000,
            ..Default::default()
        },
        ppo: PpoHyperparameters {
            update_interval: 2048,
            entropy_coefficient: 0.02,
            ..Default::default()
        },
        sac: SacHyperparameters {
            replay_start_size: 1000,
            ..Default::default()
        },
        rnd: RndHyperparameters::default(),
    }
}

pub fn train_frozen_lake_with_dqn(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_dqn(config(), parallel, save, load);
}

pub fn train_frozen_lake_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_ppo(config(), parallel, save, load);
}

pub fn train_frozen_lake_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_sac(config(), parallel, save, load);
}
