mod train_ant_with_ppo;
mod train_cartpole_with_ppo;
mod train_lunar_lander_with_ppo_rnd;
mod train_lunar_lander_with_ppo_shared_rnd;
mod train_web_LunarLander_with_dqn;
mod train_web_cartpole_with_dqn;
mod train_web_cartpole_with_sac;
mod train_web_lunar_lander_with_sac;

pub use train_ant_with_ppo::train_ant_with_ppo;
pub use train_cartpole_with_ppo::train_cartpole_with_ppo;
pub use train_lunar_lander_with_ppo_rnd::train_lunar_lander_with_ppo_rnd;
pub use train_lunar_lander_with_ppo_shared_rnd::train_lunar_lander_with_ppo_shared_rnd;
pub use train_web_LunarLander_with_dqn::train_web_LunarLander_with_dqn;
pub use train_web_cartpole_with_dqn::train_web_cartpole_with_dqn;
pub use train_web_cartpole_with_sac::train_web_cartpole_with_sac;
pub use train_web_lunar_lander_with_sac::train_web_lunar_lander_with_sac;

fn path_for_agent(path: &Option<String>, agent_id: usize) -> Option<String> {
    path.as_ref()
        .map(|path| path.replace("{agent_id}", &agent_id.to_string()))
}
