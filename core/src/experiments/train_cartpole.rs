use super::discrete_env::{
    self, DiscreteEnvConfig, DqnHyperparameters, PpoHyperparameters, RndHyperparameters,
    SacHyperparameters,
};

fn config() -> DiscreteEnvConfig {
    DiscreteEnvConfig {
        label: "CartPole-v1",
        gym_id: "CartPole-v1",
        state_count: 4,
        action_count: 2,
        one_hot_state: false,
        episodes: 10_000,
        max_steps: 500,
        log_interval: 100,
        dqn: DqnHyperparameters {
            hidden_channels: 64,
            replay_capacity: 50_000,
            epsilon_decay_steps: 20_000,
            ..Default::default()
        },
        ppo: PpoHyperparameters {
            hidden_channels: 64,
            update_interval: 500,
            minibatch_size: 32,
            entropy_coefficient: 0.0,
            ..Default::default()
        },
        sac: SacHyperparameters {
            replay_capacity: 300_000,
            batch_size: 32,
            ..Default::default()
        },
        rnd: RndHyperparameters::default(),
    }
}

pub fn train_cartpole_with_dqn(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_dqn(config(), parallel, save, load);
}

pub fn train_cartpole_with_ppo(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_ppo(config(), parallel, save, load);
}

pub fn train_cartpole_with_sac(parallel: usize, save: Option<String>, load: Option<String>) {
    discrete_env::train_sac(config(), parallel, save, load);
}
