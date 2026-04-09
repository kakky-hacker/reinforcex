use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use dashmap::DashMap;

use reinforcex::agents::{BaseAgent, DQN, PPO};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::{FCGaussianPolicyWithValue, FCQNetwork};

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

static AGENTS: LazyLock<DashMap<u64, Arc<std::sync::Mutex<AgentWrapper>>>> =
    LazyLock::new(|| DashMap::new());

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
pub struct AgentConfig {
    pub agent_type: u32,

    pub obs_size: u64,
    pub action_size: u64,
    pub learning_rate: f64,
    pub gamma: f64,

    pub batch_size: u64,
    pub buffer_size: u64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: u64,

    pub lambda: f64,
    pub update_interval: u64,
    pub epoch: u64,
    pub minibatch_size: u64,
    pub clip_eps: f64,
}

enum AgentEnum {
    DQN(DQN),
    PPO(PPO),
}

struct AgentWrapper {
    agent: AgentEnum,
    device: Device,
}

fn create_agent(config: &AgentConfig) -> Option<AgentWrapper> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let opt = nn::Adam::default().build(&vs, config.learning_rate).ok()?;

    match config.agent_type {
        0 => {
            let model = FCQNetwork::new(
                vs,
                config.obs_size as i64,
                config.action_size as i64,
                2,
                128,
            );

            let explorer = EpsilonGreedy::new(
                config.epsilon_start,
                config.epsilon_end,
                config.epsilon_decay as usize,
            );

            let buffer = ReplayBuffer::new(config.buffer_size as usize, 1);

            let agent = DQN::new(
                Box::new(model),
                Arc::new(buffer),
                opt,
                config.action_size as usize,
                config.batch_size as usize,
                1,
                100,
                Box::new(explorer),
                None,
                config.gamma,
            );

            Some(AgentWrapper {
                agent: AgentEnum::DQN(agent),
                device,
            })
        }

        _ => {
            let model = FCGaussianPolicyWithValue::new(
                vs,
                config.obs_size as i64,
                config.action_size as i64,
                2,
                128,
                Some(-1.0),
                Some(1.0),
                true,
                "spherical",
                0.1,
            );

            let agent = PPO::new(
                Box::new(model),
                opt,
                config.gamma,
                config.lambda,
                config.update_interval as usize,
                config.epoch as usize,
                config.minibatch_size as usize,
                config.clip_eps,
                0.2,
                0.5,
                0.01,
                true,
            );

            Some(AgentWrapper {
                agent: AgentEnum::PPO(agent),
                device,
            })
        }
    }
}

fn get_agent(id: u64) -> Option<Arc<std::sync::Mutex<AgentWrapper>>> {
    AGENTS.get(&id).map(|entry| entry.value().clone())
}

#[no_mangle]
pub extern "C" fn rx_agent_create(config: *const AgentConfig) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        if config.is_null() {
            return 0;
        }

        let config = unsafe { &*config };

        let wrapper = match create_agent(config) {
            Some(w) => w,
            None => return 0,
        };

        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        AGENTS.insert(id, Arc::new(std::sync::Mutex::new(wrapper)));

        id
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn rx_agent_act_and_train(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    reward: f32,
    out: *mut f32,
    out_len: u64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if obs.is_null() || out.is_null() || obs_len == 0 || out_len == 0 {
            return;
        }

        let wrapper = match get_agent(id) {
            Some(w) => w,
            None => return,
        };

        let mut guard = match wrapper.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        let obs_len = obs_len as usize;
        let out_len = out_len as usize;

        let obs_slice = unsafe { std::slice::from_raw_parts(obs, obs_len) };

        let obs_tensor = Tensor::from_slice(obs_slice)
            .to_kind(Kind::Float)
            .to_device(guard.device);

        match &mut guard.agent {
            AgentEnum::DQN(agent) => {
                if out_len < 1 {
                    return;
                }

                let action = agent.act_and_train(&obs_tensor, reward as f64);

                unsafe {
                    *out = action.int64_value(&[]) as f32;
                }
            }

            AgentEnum::PPO(agent) => {
                let action_tensor = agent
                    .act_and_train(&obs_tensor, reward as f64)
                    .flatten(0, -1);

                let action_tensor = action_tensor.to_device(Device::Cpu).to_kind(Kind::Float);

                let numel = action_tensor.numel();
                let write_len = numel.min(out_len);

                let mut vec = vec![0f32; numel];

                action_tensor.copy_data(&mut vec, numel);

                unsafe {
                    std::ptr::copy_nonoverlapping(vec.as_ptr(), out, write_len);
                }
            }
        }
    }));
}

#[no_mangle]
pub extern "C" fn rx_agent_stop_episode(id: u64, obs: *const f32, obs_len: u64, reward: f32) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if obs.is_null() || obs_len == 0 {
            return;
        }

        let wrapper = match get_agent(id) {
            Some(w) => w,
            None => return,
        };

        let mut guard = match wrapper.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        let obs_len = obs_len as usize;

        let obs_slice = unsafe { std::slice::from_raw_parts(obs, obs_len) };

        let obs_tensor = Tensor::from_slice(obs_slice)
            .to_kind(Kind::Float)
            .to_device(guard.device);

        match &mut guard.agent {
            AgentEnum::DQN(agent) => {
                agent.stop_episode_and_train(&obs_tensor, reward as f64);
            }
            AgentEnum::PPO(agent) => {
                agent.stop_episode_and_train(&obs_tensor, reward as f64);
            }
        }
    }));
}

#[no_mangle]
pub extern "C" fn rx_agent_destroy(id: u64) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        AGENTS.remove(&id);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dqn_agent() {
        let config = AgentConfig {
            agent_type: 0,

            obs_size: 4,
            action_size: 2,
            learning_rate: 1e-3,
            gamma: 0.99,

            batch_size: 32,
            buffer_size: 1000,
            epsilon_start: 1.0,
            epsilon_end: 0.1,
            epsilon_decay: 1000,

            lambda: 0.95,
            update_interval: 128,
            epoch: 4,
            minibatch_size: 32,
            clip_eps: 0.2,
        };

        let id = rx_agent_create(&config as *const _);
        assert!(id != 0);

        let obs = vec![0.1f32; config.obs_size as usize];
        let mut out = vec![0.0f32; 8];

        for _ in 0..10 {
            rx_agent_act_and_train(
                id,
                obs.as_ptr(),
                obs.len() as u64,
                0.5,
                out.as_mut_ptr(),
                out.len() as u64,
            );
        }

        rx_agent_stop_episode(id, obs.as_ptr(), obs.len() as u64, 1.0);
        rx_agent_destroy(id);
    }

    #[test]
    fn test_ppo_agent() {
        let config = AgentConfig {
            agent_type: 1, // PPO

            obs_size: 8,
            action_size: 3,
            learning_rate: 1e-4,
            gamma: 0.99,

            batch_size: 32,
            buffer_size: 1000,
            epsilon_start: 1.0,
            epsilon_end: 0.1,
            epsilon_decay: 1000,

            lambda: 0.95,
            update_interval: 128,
            epoch: 4,
            minibatch_size: 32,
            clip_eps: 0.2,
        };

        // --- create ---
        let id = rx_agent_create(&config as *const _);
        assert!(id != 0);

        let obs = vec![0.2f32; config.obs_size as usize];

        let mut out = vec![0.0f32; config.action_size as usize];

        for _ in 0..10 {
            rx_agent_act_and_train(
                id,
                obs.as_ptr(),
                obs.len() as u64,
                0.3,
                out.as_mut_ptr(),
                out.len() as u64,
            );

            assert!(out.iter().any(|&v| v != 0.0));
        }

        rx_agent_stop_episode(id, obs.as_ptr(), obs.len() as u64, 1.0);

        rx_agent_destroy(id);
    }
}
