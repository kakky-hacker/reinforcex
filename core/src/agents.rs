mod base_agent;
mod dqn;
mod ppo;
mod sac;

pub use base_agent::BaseAgent;
pub use dqn::DQN;
pub use ppo::PPO;
pub use sac::SAC;

#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    #[test]
    fn agents_derive_send_from_their_components() {
        fn assert_send<T: Send>() {}
        assert_send::<DQN>();
        assert_send::<PPO>();
        assert_send::<SAC>();
    }
}
