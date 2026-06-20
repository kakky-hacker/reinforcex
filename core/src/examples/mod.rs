mod train_ant_with_ppo;
mod train_cartpole_with_dqn;
mod train_cartpole_with_ppo;
mod train_cartpole_with_sac;
mod train_lunar_lander_with_dqn;
mod train_lunar_lander_with_ppo_rnd;
mod train_lunar_lander_with_ppo_shared_rnd;
mod train_lunar_lander_with_sac;

pub use train_ant_with_ppo::train_ant_with_ppo;
pub use train_cartpole_with_dqn::train_cartpole_with_dqn;
pub use train_cartpole_with_ppo::train_cartpole_with_ppo;
pub use train_cartpole_with_sac::train_cartpole_with_sac;
pub use train_lunar_lander_with_dqn::train_lunar_lander_with_dqn;
pub use train_lunar_lander_with_ppo_rnd::train_lunar_lander_with_ppo_rnd;
pub use train_lunar_lander_with_ppo_shared_rnd::train_lunar_lander_with_ppo_shared_rnd;
pub use train_lunar_lander_with_sac::train_lunar_lander_with_sac;

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
