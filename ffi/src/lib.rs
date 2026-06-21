use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use dashmap::DashMap;
use reinforcex::agents::{BaseAgent, DQN, PPO, SAC};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::{
    BasePolicy, BaseQFunction, FCGaussianPolicy, FCGaussianPolicyWithValue, FCQNetwork,
    FCSoftmaxPolicy, FCSoftmaxPolicyWithValue,
};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

pub const RX_OK: i32 = 0;
pub const RX_ERROR_NULL_POINTER: i32 = -1;
pub const RX_ERROR_INVALID_ARGUMENT: i32 = -2;
pub const RX_ERROR_NOT_FOUND: i32 = -3;
pub const RX_ERROR_BUFFER_TOO_SMALL: i32 = -4;
pub const RX_ERROR_PANIC: i32 = -5;
pub const RX_ERROR_INTERNAL: i32 = -6;

pub const RX_ACTION_DISCRETE: u32 = 0;
pub const RX_ACTION_CONTINUOUS: u32 = 1;

static AGENTS: LazyLock<DashMap<u64, Arc<Mutex<AgentWrapper>>>> = LazyLock::new(DashMap::new);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Settings shared by every built-in agent.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxAgentConfig {
    pub obs_size: u64,
    pub action_size: u64,
    pub hidden_layers: u64,
    pub hidden_size: u64,
    pub gamma: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxDqnConfig {
    pub agent: RxAgentConfig,
    pub learning_rate: f64,
    pub batch_size: u64,
    pub replay_capacity: u64,
    pub replay_n_steps: u64,
    pub update_interval: u64,
    pub target_update_interval: u64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay_steps: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxPpoConfig {
    pub agent: RxAgentConfig,
    pub action_space: u32,
    pub learning_rate: f64,
    pub gae_lambda: f64,
    pub update_interval: u64,
    pub epochs: u64,
    pub minibatch_size: u64,
    pub policy_clip_epsilon: f64,
    pub value_clip_range: f64,
    pub value_loss_coefficient: f64,
    pub entropy_coefficient: f64,
    pub standardize_gae: u32,
    pub min_action: f64,
    pub max_action: f64,
    pub min_variance: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxSacConfig {
    pub agent: RxAgentConfig,
    pub action_space: u32,
    pub actor_learning_rate: f64,
    pub critic_learning_rate: f64,
    pub replay_capacity: u64,
    pub replay_start_size: u64,
    pub batch_size: u64,
    pub replay_n_steps: u64,
    pub update_interval: u64,
    pub target_update_interval: u64,
    pub tau: f64,
    pub alpha: f64,
    pub min_variance: f64,
    pub squash_action: u32,
}

struct AgentWrapper {
    agent: Box<dyn BaseAgent + Send>,
    device: Device,
    obs_size: usize,
    output_size: usize,
}

fn default_agent_config(obs_size: u64, action_size: u64, hidden_size: u64) -> RxAgentConfig {
    RxAgentConfig {
        obs_size,
        action_size,
        hidden_layers: 2,
        hidden_size,
        gamma: 0.99,
    }
}

fn default_dqn_config(obs_size: u64, action_size: u64) -> RxDqnConfig {
    RxDqnConfig {
        agent: default_agent_config(obs_size, action_size, 128),
        learning_rate: 3e-4,
        batch_size: 64,
        replay_capacity: 100_000,
        replay_n_steps: 1,
        update_interval: 1,
        target_update_interval: 200,
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay_steps: 50_000,
    }
}

fn default_ppo_config(obs_size: u64, action_size: u64) -> RxPpoConfig {
    RxPpoConfig {
        agent: default_agent_config(obs_size, action_size, 256),
        action_space: RX_ACTION_CONTINUOUS,
        learning_rate: 3e-4,
        gae_lambda: 0.95,
        update_interval: 2_048,
        epochs: 10,
        minibatch_size: 64,
        policy_clip_epsilon: 0.2,
        value_clip_range: 0.2,
        value_loss_coefficient: 0.5,
        entropy_coefficient: 0.01,
        standardize_gae: 1,
        min_action: -1.0,
        max_action: 1.0,
        min_variance: 0.1,
    }
}

fn default_sac_config(obs_size: u64, action_size: u64) -> RxSacConfig {
    RxSacConfig {
        agent: default_agent_config(obs_size, action_size, 256),
        action_space: RX_ACTION_CONTINUOUS,
        actor_learning_rate: 3e-4,
        critic_learning_rate: 3e-4,
        replay_capacity: 1_000_000,
        replay_start_size: 10_000,
        batch_size: 256,
        replay_n_steps: 1,
        update_interval: 1,
        target_update_interval: 1,
        tau: 0.005,
        alpha: 0.2,
        min_variance: 1e-3,
        squash_action: 1,
    }
}

fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn is_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn is_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn valid_flag(value: u32) -> bool {
    value <= 1
}

fn valid_action_space(value: u32) -> bool {
    matches!(value, RX_ACTION_DISCRETE | RX_ACTION_CONTINUOUS)
}

fn to_usize(value: u64) -> Result<usize, i32> {
    usize::try_from(value).map_err(|_| RX_ERROR_INVALID_ARGUMENT)
}

fn to_i64(value: u64) -> Result<i64, i32> {
    i64::try_from(value).map_err(|_| RX_ERROR_INVALID_ARGUMENT)
}

fn validate_agent(config: &RxAgentConfig) -> Result<(), i32> {
    if config.obs_size == 0
        || config.action_size == 0
        || config.hidden_size == 0
        || to_i64(config.obs_size).is_err()
        || to_i64(config.action_size).is_err()
        || to_i64(config.hidden_size).is_err()
        || to_usize(config.hidden_layers).is_err()
        || !is_probability(config.gamma)
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn validate_dqn(config: &RxDqnConfig) -> Result<(), i32> {
    validate_agent(&config.agent)?;
    if !is_positive(config.learning_rate)
        || config.batch_size == 0
        || config.replay_capacity == 0
        || config.batch_size > config.replay_capacity
        || config.replay_n_steps == 0
        || config.update_interval == 0
        || config.target_update_interval == 0
        || !is_probability(config.epsilon_start)
        || !is_probability(config.epsilon_end)
        || config.epsilon_decay_steps == 0
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn validate_ppo(config: &RxPpoConfig) -> Result<(), i32> {
    validate_agent(&config.agent)?;
    if !valid_action_space(config.action_space)
        || !valid_flag(config.standardize_gae)
        || !is_positive(config.learning_rate)
        || !is_probability(config.gae_lambda)
        || config.update_interval == 0
        || config.epochs == 0
        || config.minibatch_size == 0
        || config.minibatch_size > config.update_interval
        || !is_probability(config.policy_clip_epsilon)
        || !is_non_negative(config.value_clip_range)
        || !is_non_negative(config.value_loss_coefficient)
        || !is_non_negative(config.entropy_coefficient)
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    if config.action_space == RX_ACTION_CONTINUOUS
        && (!config.min_action.is_finite()
            || !config.max_action.is_finite()
            || config.min_action >= config.max_action
            || !is_positive(config.min_variance))
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn validate_sac(config: &RxSacConfig) -> Result<(), i32> {
    validate_agent(&config.agent)?;
    if !valid_action_space(config.action_space)
        || !valid_flag(config.squash_action)
        || !is_positive(config.actor_learning_rate)
        || !is_positive(config.critic_learning_rate)
        || config.replay_capacity == 0
        || config.replay_start_size == 0
        || config.replay_start_size > config.replay_capacity
        || config.batch_size == 0
        || config.batch_size > config.replay_capacity
        || config.replay_n_steps == 0
        || config.update_interval == 0
        || config.target_update_interval == 0
        || !is_probability(config.tau)
        || !is_non_negative(config.alpha)
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    if config.action_space == RX_ACTION_CONTINUOUS && !is_positive(config.min_variance) {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn create_dqn(config: &RxDqnConfig) -> Result<AgentWrapper, i32> {
    validate_dqn(config)?;
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let optimizer = nn::Adam::default()
        .build(&vs, config.learning_rate)
        .map_err(|_| RX_ERROR_INTERNAL)?;
    let model = FCQNetwork::new(
        vs,
        to_i64(config.agent.obs_size)?,
        to_i64(config.agent.action_size)?,
        to_usize(config.agent.hidden_layers)?,
        to_i64(config.agent.hidden_size)?,
    );
    let replay = Arc::new(ReplayBuffer::new(
        to_usize(config.replay_capacity)?,
        to_usize(config.replay_n_steps)?,
    ));
    let explorer = EpsilonGreedy::new(
        config.epsilon_start,
        config.epsilon_end,
        to_usize(config.epsilon_decay_steps)?,
    );
    let agent = DQN::new(
        Box::new(model),
        replay,
        optimizer,
        to_usize(config.agent.action_size)?,
        to_usize(config.batch_size)?,
        to_usize(config.update_interval)?,
        to_usize(config.target_update_interval)?,
        Box::new(explorer),
        None,
        config.agent.gamma,
        None,
        None,
    );

    Ok(AgentWrapper {
        agent: Box::new(agent),
        device,
        obs_size: to_usize(config.agent.obs_size)?,
        output_size: 1,
    })
}

fn create_ppo(config: &RxPpoConfig) -> Result<AgentWrapper, i32> {
    validate_ppo(config)?;
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let optimizer = nn::Adam::default()
        .build(&vs, config.learning_rate)
        .map_err(|_| RX_ERROR_INTERNAL)?;
    let obs_size = to_i64(config.agent.obs_size)?;
    let action_size = to_i64(config.agent.action_size)?;
    let hidden_layers = to_usize(config.agent.hidden_layers)?;
    let hidden_size = to_i64(config.agent.hidden_size)?;

    let model: Box<dyn BasePolicy> = match config.action_space {
        RX_ACTION_DISCRETE => Box::new(FCSoftmaxPolicyWithValue::new(
            vs,
            obs_size,
            action_size,
            hidden_layers,
            hidden_size,
            0.0,
        )),
        RX_ACTION_CONTINUOUS => Box::new(FCGaussianPolicyWithValue::new(
            vs,
            obs_size,
            action_size,
            hidden_layers,
            hidden_size,
            Some(config.min_action),
            Some(config.max_action),
            true,
            "spherical",
            config.min_variance,
        )),
        _ => return Err(RX_ERROR_INVALID_ARGUMENT),
    };
    let agent = PPO::new(
        model,
        optimizer,
        config.agent.gamma,
        config.gae_lambda,
        to_usize(config.update_interval)?,
        to_usize(config.epochs)?,
        to_usize(config.minibatch_size)?,
        config.policy_clip_epsilon,
        config.value_clip_range,
        config.value_loss_coefficient,
        config.entropy_coefficient,
        config.standardize_gae != 0,
        None,
        None,
    );

    Ok(AgentWrapper {
        agent: Box::new(agent),
        device,
        obs_size: to_usize(config.agent.obs_size)?,
        output_size: if config.action_space == RX_ACTION_DISCRETE {
            1
        } else {
            to_usize(config.agent.action_size)?
        },
    })
}

fn build_critic(
    device: Device,
    learning_rate: f64,
    input_size: i64,
    output_size: i64,
    hidden_layers: usize,
    hidden_size: i64,
) -> Result<(Box<dyn BaseQFunction>, nn::Optimizer), i32> {
    let vs = nn::VarStore::new(device);
    let optimizer = nn::Adam::default()
        .build(&vs, learning_rate)
        .map_err(|_| RX_ERROR_INTERNAL)?;
    let critic = FCQNetwork::new(vs, input_size, output_size, hidden_layers, hidden_size);
    Ok((Box::new(critic), optimizer))
}

fn create_sac(config: &RxSacConfig) -> Result<AgentWrapper, i32> {
    validate_sac(config)?;
    let device = Device::cuda_if_available();
    let obs_size = to_i64(config.agent.obs_size)?;
    let action_size = to_i64(config.agent.action_size)?;
    let hidden_layers = to_usize(config.agent.hidden_layers)?;
    let hidden_size = to_i64(config.agent.hidden_size)?;

    let actor_vs = nn::VarStore::new(device);
    let actor_optimizer = nn::Adam::default()
        .build(&actor_vs, config.actor_learning_rate)
        .map_err(|_| RX_ERROR_INTERNAL)?;

    let (actor, critic_input_size, critic_output_size): (Box<dyn BasePolicy>, i64, i64) =
        match config.action_space {
            RX_ACTION_DISCRETE => (
                Box::new(FCSoftmaxPolicy::new(
                    actor_vs,
                    obs_size,
                    action_size,
                    hidden_layers,
                    hidden_size,
                    0.0,
                )),
                obs_size,
                action_size,
            ),
            RX_ACTION_CONTINUOUS => (
                Box::new(FCGaussianPolicy::new(
                    actor_vs,
                    obs_size,
                    action_size,
                    hidden_layers,
                    hidden_size,
                    None,
                    None,
                    false,
                    "diagonal",
                    config.min_variance,
                )),
                obs_size
                    .checked_add(action_size)
                    .ok_or(RX_ERROR_INVALID_ARGUMENT)?,
                1,
            ),
            _ => return Err(RX_ERROR_INVALID_ARGUMENT),
        };

    let (critic1, critic1_optimizer) = build_critic(
        device,
        config.critic_learning_rate,
        critic_input_size,
        critic_output_size,
        hidden_layers,
        hidden_size,
    )?;
    let (critic2, critic2_optimizer) = build_critic(
        device,
        config.critic_learning_rate,
        critic_input_size,
        critic_output_size,
        hidden_layers,
        hidden_size,
    )?;
    let replay = Arc::new(ReplayBuffer::new(
        to_usize(config.replay_capacity)?,
        to_usize(config.replay_n_steps)?,
    ));
    let agent = SAC::new(
        actor,
        actor_optimizer,
        critic1,
        critic1_optimizer,
        critic2,
        critic2_optimizer,
        replay,
        to_usize(config.replay_start_size)?,
        to_usize(config.batch_size)?,
        to_usize(config.update_interval)?,
        to_usize(config.target_update_interval)?,
        config.agent.gamma,
        config.tau,
        config.alpha,
        config.action_space == RX_ACTION_CONTINUOUS && config.squash_action != 0,
    );

    Ok(AgentWrapper {
        agent: Box::new(agent),
        device,
        obs_size: to_usize(config.agent.obs_size)?,
        output_size: if config.action_space == RX_ACTION_DISCRETE {
            1
        } else {
            to_usize(config.agent.action_size)?
        },
    })
}

fn insert_agent(wrapper: AgentWrapper) -> Result<u64, i32> {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        return Err(RX_ERROR_INTERNAL);
    }
    AGENTS.insert(id, Arc::new(Mutex::new(wrapper)));
    Ok(id)
}

fn get_agent(id: u64) -> Result<Arc<Mutex<AgentWrapper>>, i32> {
    if id == 0 {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    AGENTS
        .get(&id)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or(RX_ERROR_NOT_FOUND)
}

fn create_from_config<T: Copy>(
    config: *const T,
    out_id: *mut u64,
    create: impl FnOnce(&T) -> Result<AgentWrapper, i32>,
) -> i32 {
    if config.is_null() || out_id.is_null() {
        return RX_ERROR_NULL_POINTER;
    }
    unsafe {
        *out_id = 0;
    }
    let config = unsafe { *config };
    match create(&config).and_then(insert_agent) {
        Ok(id) => {
            unsafe {
                *out_id = id;
            }
            RX_OK
        }
        Err(status) => status,
    }
}

fn write_default<T>(out_config: *mut T, config: T) -> i32 {
    if out_config.is_null() {
        return RX_ERROR_NULL_POINTER;
    }
    unsafe {
        *out_config = config;
    }
    RX_OK
}

fn make_observation(
    obs: *const f32,
    obs_len: u64,
    expected_len: usize,
    device: Device,
) -> Result<Tensor, i32> {
    if obs.is_null() {
        return Err(RX_ERROR_NULL_POINTER);
    }
    let obs_len = to_usize(obs_len)?;
    if obs_len != expected_len {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    let obs = unsafe { std::slice::from_raw_parts(obs, obs_len) };
    if !obs.iter().all(|value| value.is_finite()) {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(Tensor::from_slice(obs)
        .to_kind(Kind::Float)
        .to_device(device))
}

fn write_action(action: Tensor, expected_len: usize, out: *mut f32, out_len: u64) -> i64 {
    if out.is_null() {
        return i64::from(RX_ERROR_NULL_POINTER);
    }
    let out_len = match to_usize(out_len) {
        Ok(value) => value,
        Err(status) => return i64::from(status),
    };
    if out_len < expected_len {
        return i64::from(RX_ERROR_BUFFER_TOO_SMALL);
    }

    let action = action
        .flatten(0, -1)
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);
    let numel = action.numel();
    if numel != expected_len {
        return i64::from(RX_ERROR_INTERNAL);
    }
    let mut values = vec![0.0f32; numel];
    action.copy_data(&mut values, numel);
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), out, numel);
    }
    i64::try_from(numel).unwrap_or(i64::from(RX_ERROR_INTERNAL))
}

fn act_impl(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    reward: Option<f32>,
    out: *mut f32,
    out_len: u64,
) -> i64 {
    if let Some(reward) = reward {
        if !reward.is_finite() {
            return i64::from(RX_ERROR_INVALID_ARGUMENT);
        }
    }
    let wrapper = match get_agent(id) {
        Ok(wrapper) => wrapper,
        Err(status) => return i64::from(status),
    };
    let mut guard = match wrapper.lock() {
        Ok(guard) => guard,
        Err(_) => return i64::from(RX_ERROR_INTERNAL),
    };
    if out.is_null() {
        return i64::from(RX_ERROR_NULL_POINTER);
    }
    let out_len_usize = match to_usize(out_len) {
        Ok(value) => value,
        Err(status) => return i64::from(status),
    };
    if out_len_usize < guard.output_size {
        return i64::from(RX_ERROR_BUFFER_TOO_SMALL);
    }
    let obs = match make_observation(obs, obs_len, guard.obs_size, guard.device) {
        Ok(obs) => obs,
        Err(status) => return i64::from(status),
    };
    let action = match reward {
        Some(reward) => guard.agent.act_and_train(&obs, f64::from(reward)),
        None => guard.agent.act(&obs),
    };
    write_action(action, guard.output_size, out, out_len)
}

fn stop_episode_impl(id: u64, obs: *const f32, obs_len: u64, reward: f32) -> i32 {
    if !reward.is_finite() {
        return RX_ERROR_INVALID_ARGUMENT;
    }
    let wrapper = match get_agent(id) {
        Ok(wrapper) => wrapper,
        Err(status) => return status,
    };
    let mut guard = match wrapper.lock() {
        Ok(guard) => guard,
        Err(_) => return RX_ERROR_INTERNAL,
    };
    let obs = match make_observation(obs, obs_len, guard.obs_size, guard.device) {
        Ok(obs) => obs,
        Err(status) => return status,
    };
    guard.agent.stop_episode_and_train(&obs, f64::from(reward));
    RX_OK
}

#[no_mangle]
pub extern "C" fn rx_dqn_config_default(
    out_config: *mut RxDqnConfig,
    obs_size: u64,
    action_size: u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        write_default(out_config, default_dqn_config(obs_size, action_size))
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_ppo_config_default(
    out_config: *mut RxPpoConfig,
    obs_size: u64,
    action_size: u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        write_default(out_config, default_ppo_config(obs_size, action_size))
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_sac_config_default(
    out_config: *mut RxSacConfig,
    obs_size: u64,
    action_size: u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        write_default(out_config, default_sac_config(obs_size, action_size))
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_dqn_create(config: *const RxDqnConfig, out_id: *mut u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        create_from_config(config, out_id, create_dqn)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_ppo_create(config: *const RxPpoConfig, out_id: *mut u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        create_from_config(config, out_id, create_ppo)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_sac_create(config: *const RxSacConfig, out_id: *mut u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        create_from_config(config, out_id, create_sac)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

/// Selects an action without adding a transition or updating the agent.
/// Returns the number of floats written, or a negative error code.
#[no_mangle]
pub extern "C" fn rx_agent_act(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    out: *mut f32,
    out_len: u64,
) -> i64 {
    catch_unwind(AssertUnwindSafe(|| {
        act_impl(id, obs, obs_len, None, out, out_len)
    }))
    .unwrap_or(i64::from(RX_ERROR_PANIC))
}

/// Selects an action, records the previous transition, and updates when due.
/// Returns the number of floats written, or a negative error code.
#[no_mangle]
pub extern "C" fn rx_agent_act_and_train(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    reward: f32,
    out: *mut f32,
    out_len: u64,
) -> i64 {
    catch_unwind(AssertUnwindSafe(|| {
        act_impl(id, obs, obs_len, Some(reward), out, out_len)
    }))
    .unwrap_or(i64::from(RX_ERROR_PANIC))
}

#[no_mangle]
pub extern "C" fn rx_agent_stop_episode(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    reward: f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        stop_episode_impl(id, obs, obs_len, reward)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_agent_destroy(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if id == 0 {
            return RX_ERROR_INVALID_ARGUMENT;
        }
        if AGENTS.remove(&id).is_some() {
            RX_OK
        } else {
            RX_ERROR_NOT_FOUND
        }
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dqn_for_test() -> (u64, RxDqnConfig) {
        let mut config = default_dqn_config(4, 2);
        config.batch_size = 4;
        config.replay_capacity = 32;
        let mut id = 0;
        assert_eq!(rx_dqn_create(&config, &mut id), RX_OK);
        assert_ne!(id, 0);
        (id, config)
    }

    #[test]
    fn dqn_lifecycle_and_validation() {
        let (id, config) = create_dqn_for_test();
        let obs = vec![0.1f32; config.agent.obs_size as usize];
        let mut out = [0.0f32; 1];

        assert_eq!(
            rx_agent_act_and_train(id, obs.as_ptr(), obs.len() as u64, 0.5, out.as_mut_ptr(), 1),
            1
        );
        assert_eq!(
            rx_agent_act(id, obs.as_ptr(), obs.len() as u64, out.as_mut_ptr(), 1),
            1
        );
        assert_eq!(
            rx_agent_act(id, obs.as_ptr(), 3, out.as_mut_ptr(), 1),
            i64::from(RX_ERROR_INVALID_ARGUMENT)
        );
        assert_eq!(rx_agent_stop_episode(id, obs.as_ptr(), 4, 1.0), RX_OK);
        assert_eq!(rx_agent_destroy(id), RX_OK);
        assert_eq!(rx_agent_destroy(id), RX_ERROR_NOT_FOUND);
    }

    #[test]
    fn ppo_supports_discrete_and_continuous_actions() {
        for action_space in [RX_ACTION_DISCRETE, RX_ACTION_CONTINUOUS] {
            let mut config = default_ppo_config(4, 2);
            config.action_space = action_space;
            config.update_interval = 8;
            config.minibatch_size = 4;
            let mut id = 0;
            assert_eq!(rx_ppo_create(&config, &mut id), RX_OK);

            let obs = [0.1f32; 4];
            let mut out = [0.0f32; 2];
            let expected = if action_space == RX_ACTION_DISCRETE {
                1
            } else {
                2
            };
            assert_eq!(
                rx_agent_act_and_train(id, obs.as_ptr(), 4, 0.0, out.as_mut_ptr(), 2),
                expected
            );
            assert_eq!(rx_agent_stop_episode(id, obs.as_ptr(), 4, 1.0), RX_OK);
            assert_eq!(rx_agent_destroy(id), RX_OK);
        }
    }

    #[test]
    fn sac_supports_discrete_and_continuous_actions() {
        for action_space in [RX_ACTION_DISCRETE, RX_ACTION_CONTINUOUS] {
            let mut config = default_sac_config(4, 2);
            config.action_space = action_space;
            config.replay_capacity = 32;
            config.replay_start_size = 4;
            config.batch_size = 4;
            let mut id = 0;
            assert_eq!(rx_sac_create(&config, &mut id), RX_OK);

            let obs = [0.1f32; 4];
            let mut out = [0.0f32; 2];
            let expected = if action_space == RX_ACTION_DISCRETE {
                1
            } else {
                2
            };
            for _ in 0..6 {
                assert_eq!(
                    rx_agent_act_and_train(id, obs.as_ptr(), 4, 0.0, out.as_mut_ptr(), 2),
                    expected
                );
            }
            assert_eq!(rx_agent_stop_episode(id, obs.as_ptr(), 4, 1.0), RX_OK);
            assert_eq!(rx_agent_destroy(id), RX_OK);
        }
    }

    #[test]
    fn rejects_invalid_configs_and_small_output_buffers() {
        let mut invalid = default_sac_config(4, 2);
        invalid.tau = 2.0;
        let mut id = 123;
        assert_eq!(rx_sac_create(&invalid, &mut id), RX_ERROR_INVALID_ARGUMENT);
        assert_eq!(id, 0);

        let mut config = default_ppo_config(4, 2);
        config.update_interval = 8;
        config.minibatch_size = 4;
        assert_eq!(rx_ppo_create(&config, &mut id), RX_OK);
        let obs = [0.0f32; 4];
        let mut out = [0.0f32; 1];
        assert_eq!(
            rx_agent_act(id, obs.as_ptr(), 4, out.as_mut_ptr(), 1),
            i64::from(RX_ERROR_BUFFER_TOO_SMALL)
        );
        assert_eq!(rx_agent_destroy(id), RX_OK);
    }
}
