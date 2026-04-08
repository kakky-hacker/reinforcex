use std::ffi::c_void;

use reinforcex::agents::{BaseAgent, DQN, PPO};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::{FCGaussianPolicyWithValue, FCQNetwork};
use std::sync::Arc;

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

#[repr(C)]
pub enum AgentType {
    DQN = 0,
    PPO = 1,
}

#[repr(C)]
pub struct AgentConfig {
    pub agent_type: usize,

    // common
    pub obs_size: usize,
    pub action_size: usize,
    pub learning_rate: f64,
    pub gamma: f64,

    // DQN
    pub batch_size: usize,
    pub buffer_size: usize,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: usize,

    // PPO
    pub lambda: f64,
    pub update_interval: usize,
    pub epoch: usize,
    pub minibatch_size: usize,
    pub clip_eps: f64,
}

enum AgentEnum {
    DQN(DQN),
    PPO(PPO),
}

struct AgentWrapper {
    agent: AgentEnum,
}

fn create_agent(config: &AgentConfig) -> AgentWrapper {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    match config.agent_type {
        x if x == AgentType::DQN as usize => {
            let opt = nn::Adam::default()
                .build(&vs, config.learning_rate)
                .unwrap();

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
                config.epsilon_decay,
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

            AgentWrapper {
                agent: AgentEnum::DQN(agent),
            }
        }

        _ => {
            let opt = nn::Adam::default()
                .build(&vs, config.learning_rate)
                .unwrap();

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
                config.update_interval,
                config.epoch,
                config.minibatch_size,
                config.clip_eps,
                0.2,
                0.5,
                0.01,
                true,
            );

            AgentWrapper {
                agent: AgentEnum::PPO(agent),
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn rx_agent_create(config: *const AgentConfig) -> *mut c_void {
    if config.is_null() {
        return std::ptr::null_mut();
    }

    let config = unsafe { &*config };

    let wrapper = create_agent(config);

    Box::into_raw(Box::new(wrapper)) as *mut c_void
}

#[no_mangle]
pub extern "C" fn rx_agent_act_and_train(
    ptr: *mut c_void,
    obs: *const f64,
    obs_len: usize,
    reward: f64,
) -> *mut f64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &mut *(ptr as *mut AgentWrapper) };

    let obs_slice = unsafe { std::slice::from_raw_parts(obs, obs_len as usize) };

    let obs_tensor = Tensor::from_slice(obs_slice).to_kind(Kind::Float);

    match &mut wrapper.agent {
        AgentEnum::DQN(agent) => {
            let action = agent.act_and_train(&obs_tensor, reward);

            let val = action.int64_value(&[]) as f64;
            let boxed = Box::new(val);
            Box::into_raw(boxed)
        }

        AgentEnum::PPO(agent) => {
            let action_tensor = agent.act_and_train(&obs_tensor, reward).flatten(0, -1);

            let size = action_tensor.size()[0];

            let mut vec = Vec::with_capacity(size as usize);
            for i in 0..size {
                vec.push(action_tensor.double_value(&[i]) as f64);
            }

            let mut boxed = vec.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            ptr
        }
    }
}

#[no_mangle]
pub extern "C" fn rx_agent_stop_episode(
    ptr: *mut c_void,
    obs: *const f64,
    obs_len: usize,
    reward: f64,
) {
    if ptr.is_null() {
        return;
    }

    let wrapper = unsafe { &mut *(ptr as *mut AgentWrapper) };

    let obs_slice = unsafe { std::slice::from_raw_parts(obs, obs_len as usize) };

    let obs_tensor = Tensor::from_slice(obs_slice).to_kind(Kind::Float);

    match &mut wrapper.agent {
        AgentEnum::DQN(agent) => {
            agent.stop_episode_and_train(&obs_tensor, reward);
        }
        AgentEnum::PPO(agent) => {
            agent.stop_episode_and_train(&obs_tensor, reward);
        }
    }
}

#[no_mangle]
pub extern "C" fn rx_agent_destroy(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        Box::from_raw(ptr as *mut AgentWrapper);
    }
}
