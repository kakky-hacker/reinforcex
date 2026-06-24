mod continuous_env;
mod discrete_env;
mod train_ant;
mod train_bipedal_walker;
mod train_cartpole;
mod train_frozen_lake;
mod train_half_cheetah;
mod train_lunar_lander;
mod train_pusher;
mod train_taxi;

pub use train_ant::{
    train_ant_with_ppo, train_ant_with_ppo_rnd, train_ant_with_ppo_shared_rnd, train_ant_with_sac,
};
pub use train_bipedal_walker::{
    train_bipedal_walker_with_ppo, train_bipedal_walker_with_ppo_rnd,
    train_bipedal_walker_with_ppo_shared_rnd, train_bipedal_walker_with_sac,
};
pub use train_cartpole::{
    train_cartpole_with_dqn, train_cartpole_with_ppo, train_cartpole_with_sac,
};
pub use train_frozen_lake::{
    train_frozen_lake_with_dqn, train_frozen_lake_with_ppo, train_frozen_lake_with_sac,
};
pub use train_half_cheetah::{
    train_half_cheetah_with_ppo, train_half_cheetah_with_ppo_rnd,
    train_half_cheetah_with_ppo_shared_rnd, train_half_cheetah_with_sac,
};
pub use train_lunar_lander::{
    train_lunar_lander_with_dqn, train_lunar_lander_with_ppo, train_lunar_lander_with_ppo_rnd,
    train_lunar_lander_with_ppo_shared_rnd, train_lunar_lander_with_sac,
};
pub use train_pusher::{
    train_pusher_with_ppo, train_pusher_with_ppo_rnd, train_pusher_with_ppo_shared_rnd,
    train_pusher_with_sac,
};
pub use train_taxi::{train_taxi_with_dqn, train_taxi_with_ppo, train_taxi_with_sac};

fn path_for_agent(path: &Option<String>, agent_id: usize) -> Option<String> {
    path.as_ref()
        .map(|path| path.replace("{agent_id}", &agent_id.to_string()))
}

fn environment_ports(parallel_count: usize) -> Vec<u16> {
    assert!(parallel_count > 0, "parallel count must be positive");

    (0..parallel_count)
        .map(|i| {
            8001u16
                .checked_add(u16::try_from(i).expect("parallel count is too large"))
                .expect("parallel count is too large")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::environment_ports;

    #[test]
    fn environment_ports_start_at_8001() {
        assert_eq!(environment_ports(4), vec![8001, 8002, 8003, 8004]);
    }

    #[test]
    #[should_panic(expected = "parallel count must be positive")]
    fn environment_ports_rejects_zero_workers() {
        environment_ports(0);
    }
}
