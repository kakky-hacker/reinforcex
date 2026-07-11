use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use dashmap::DashMap;
use reinforcex::agents::{BaseAgent, DQN, PPO, SAC};
use reinforcex::curiosity::{Basecuriosity, RND};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::{Experience, ReplayBuffer};
use reinforcex::models::{
    BasePolicy, BaseQFunction, FCGaussianPolicy, FCGaussianPolicyWithValue, FCQNetwork, FCRNDModel,
    FCSoftmaxPolicy, FCSoftmaxPolicyWithValue,
};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use ulid::Ulid;

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
static REPLAY_BUFFERS: LazyLock<DashMap<u64, Arc<ReplayBufferWrapper>>> =
    LazyLock::new(DashMap::new);
static REPLAY_WRITERS: LazyLock<DashMap<u64, Arc<Mutex<ReplayWriterWrapper>>>> =
    LazyLock::new(DashMap::new);
static RNDS: LazyLock<DashMap<u64, Arc<Mutex<RndWrapper>>>> = LazyLock::new(DashMap::new);
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxReplayBufferConfig {
    pub capacity: u64,
    pub n_steps: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxReplayWriterConfig {
    pub obs_size: u64,
    pub action_len: u64,
    pub gamma: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxRndConfig {
    pub obs_size: u64,
    pub feature_size: u64,
    pub hidden_layers: u64,
    pub hidden_size: u64,
    pub learning_rate: f64,
    pub update_interval: u64,
}

pub const RX_STAT_NAME_LEN: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RxStatistic {
    pub name: [c_char; RX_STAT_NAME_LEN],
    pub value: f64,
}

struct AgentWrapper {
    agent: Box<dyn BaseAgent + Send>,
    device: Device,
    obs_size: usize,
    output_size: usize,
}

struct ReplayBufferWrapper {
    buffer: Arc<ReplayBuffer>,
    capacity: usize,
    n_steps: usize,
}

struct ReplayWriterWrapper {
    buffer: Arc<ReplayBuffer>,
    agent_id: Ulid,
    current_episode_id: Ulid,
    device: Device,
    obs_size: usize,
    action_len: usize,
    gamma: f64,
}

struct RndWrapper {
    rnd: RND,
    device: Device,
    obs_size: usize,
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

fn default_replay_buffer_config(capacity: u64, n_steps: u64) -> RxReplayBufferConfig {
    RxReplayBufferConfig { capacity, n_steps }
}

fn default_replay_writer_config(obs_size: u64, action_len: u64) -> RxReplayWriterConfig {
    RxReplayWriterConfig {
        obs_size,
        action_len,
        gamma: 0.99,
    }
}

fn default_rnd_config(obs_size: u64) -> RxRndConfig {
    RxRndConfig {
        obs_size,
        feature_size: 128,
        hidden_layers: 2,
        hidden_size: 256,
        learning_rate: 1e-4,
        update_interval: 128,
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

fn validate_replay_buffer(config: &RxReplayBufferConfig) -> Result<(), i32> {
    if config.capacity == 0
        || config.n_steps == 0
        || to_usize(config.capacity).is_err()
        || to_usize(config.n_steps).is_err()
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn validate_replay_writer(config: &RxReplayWriterConfig) -> Result<(), i32> {
    if config.obs_size == 0
        || config.action_len == 0
        || to_i64(config.obs_size).is_err()
        || to_i64(config.action_len).is_err()
        || !is_probability(config.gamma)
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn validate_rnd(config: &RxRndConfig) -> Result<(), i32> {
    if config.obs_size == 0
        || config.feature_size == 0
        || config.hidden_size == 0
        || config.update_interval == 0
        || to_i64(config.obs_size).is_err()
        || to_i64(config.feature_size).is_err()
        || to_i64(config.hidden_size).is_err()
        || to_usize(config.hidden_layers).is_err()
        || !is_positive(config.learning_rate)
    {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn optional_c_string(ptr: *const c_char) -> Result<Option<String>, i32> {
    if ptr.is_null() {
        return Ok(None);
    }
    let string = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| RX_ERROR_INVALID_ARGUMENT)?;
    if string.is_empty() {
        Ok(None)
    } else {
        Ok(Some(string.to_string()))
    }
}

fn create_dqn_with_paths(
    config: &RxDqnConfig,
    save_path: Option<String>,
    load_path: Option<String>,
) -> Result<AgentWrapper, i32> {
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
        save_path,
        load_path,
    );

    Ok(AgentWrapper {
        agent: Box::new(agent),
        device,
        obs_size: to_usize(config.agent.obs_size)?,
        output_size: 1,
    })
}

fn create_dqn(config: &RxDqnConfig) -> Result<AgentWrapper, i32> {
    create_dqn_with_paths(config, None, None)
}

fn create_ppo_with_paths(
    config: &RxPpoConfig,
    save_path: Option<String>,
    load_path: Option<String>,
) -> Result<AgentWrapper, i32> {
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
        save_path,
        load_path,
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

fn create_ppo(config: &RxPpoConfig) -> Result<AgentWrapper, i32> {
    create_ppo_with_paths(config, None, None)
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

fn create_sac_with_replay_and_paths(
    config: &RxSacConfig,
    replay: Arc<ReplayBuffer>,
    save_path: Option<String>,
    load_path: Option<String>,
) -> Result<AgentWrapper, i32> {
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
    let agent = SAC::new_with_save_load(
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
        save_path,
        load_path,
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

fn create_sac_with_paths(
    config: &RxSacConfig,
    save_path: Option<String>,
    load_path: Option<String>,
) -> Result<AgentWrapper, i32> {
    validate_sac(config)?;
    let replay = Arc::new(ReplayBuffer::new(
        to_usize(config.replay_capacity)?,
        to_usize(config.replay_n_steps)?,
    ));
    create_sac_with_replay_and_paths(config, replay, save_path, load_path)
}

fn create_sac(config: &RxSacConfig) -> Result<AgentWrapper, i32> {
    create_sac_with_paths(config, None, None)
}

fn create_rnd_with_paths(
    config: &RxRndConfig,
    save_path: Option<String>,
    load_path: Option<String>,
) -> Result<RndWrapper, i32> {
    validate_rnd(config)?;
    let device = Device::cuda_if_available();
    let model = FCRNDModel::new(
        nn::VarStore::new(device),
        nn::VarStore::new(device),
        to_i64(config.obs_size)?,
        to_i64(config.feature_size)?,
        to_usize(config.hidden_layers)?,
        to_i64(config.hidden_size)?,
    );
    let optimizer = nn::Adam::default()
        .build(model.predictor_var_store(), config.learning_rate)
        .map_err(|_| RX_ERROR_INTERNAL)?;
    let rnd = RND::new(
        Box::new(model),
        optimizer,
        to_usize(config.update_interval)?,
        save_path,
        load_path,
    );

    Ok(RndWrapper {
        rnd,
        device,
        obs_size: to_usize(config.obs_size)?,
    })
}

fn next_id() -> Result<u64, i32> {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        return Err(RX_ERROR_INTERNAL);
    }
    Ok(id)
}

fn insert_agent(wrapper: AgentWrapper) -> Result<u64, i32> {
    let id = next_id()?;
    AGENTS.insert(id, Arc::new(Mutex::new(wrapper)));
    Ok(id)
}

fn insert_replay_buffer(wrapper: ReplayBufferWrapper) -> Result<u64, i32> {
    let id = next_id()?;
    REPLAY_BUFFERS.insert(id, Arc::new(wrapper));
    Ok(id)
}

fn insert_replay_writer(wrapper: ReplayWriterWrapper) -> Result<u64, i32> {
    let id = next_id()?;
    REPLAY_WRITERS.insert(id, Arc::new(Mutex::new(wrapper)));
    Ok(id)
}

fn insert_rnd(wrapper: RndWrapper) -> Result<u64, i32> {
    let id = next_id()?;
    RNDS.insert(id, Arc::new(Mutex::new(wrapper)));
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

fn get_replay_buffer(id: u64) -> Result<Arc<ReplayBufferWrapper>, i32> {
    if id == 0 {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    REPLAY_BUFFERS
        .get(&id)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or(RX_ERROR_NOT_FOUND)
}

fn get_replay_writer(id: u64) -> Result<Arc<Mutex<ReplayWriterWrapper>>, i32> {
    if id == 0 {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    REPLAY_WRITERS
        .get(&id)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or(RX_ERROR_NOT_FOUND)
}

fn get_rnd(id: u64) -> Result<Arc<Mutex<RndWrapper>>, i32> {
    if id == 0 {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    RNDS.get(&id)
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

fn create_from_config_with_paths<T: Copy>(
    config: *const T,
    save_path: *const c_char,
    load_path: *const c_char,
    out_id: *mut u64,
    create: impl FnOnce(&T, Option<String>, Option<String>) -> Result<AgentWrapper, i32>,
) -> i32 {
    if config.is_null() || out_id.is_null() {
        return RX_ERROR_NULL_POINTER;
    }
    unsafe {
        *out_id = 0;
    }
    let config = unsafe { *config };
    let save_path = match optional_c_string(save_path) {
        Ok(path) => path,
        Err(status) => return status,
    };
    let load_path = match optional_c_string(load_path) {
        Ok(path) => path,
        Err(status) => return status,
    };
    match create(&config, save_path, load_path).and_then(insert_agent) {
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

fn make_action(
    action: *const f32,
    action_len: u64,
    expected_len: usize,
    device: Device,
) -> Result<Tensor, i32> {
    if action.is_null() {
        return Err(RX_ERROR_NULL_POINTER);
    }
    let action_len = to_usize(action_len)?;
    if action_len != expected_len {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    let action = unsafe { std::slice::from_raw_parts(action, action_len) };
    if !action.iter().all(|value| value.is_finite()) {
        return Err(RX_ERROR_INVALID_ARGUMENT);
    }
    Ok(Tensor::from_slice(action)
        .to_kind(Kind::Float)
        .to_device(device))
}

fn curiosity_experience(state: Tensor, is_episode_terminal: bool) -> Arc<Experience> {
    Arc::new(Experience::new(
        Ulid::new(),
        Ulid::new(),
        state,
        None,
        None,
        0.0,
        is_episode_terminal,
    ))
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

fn write_stat_name(name: &str, out: &mut [c_char; RX_STAT_NAME_LEN]) {
    out.fill(0);
    let bytes = name.as_bytes();
    let n = bytes.len().min(RX_STAT_NAME_LEN - 1);
    for i in 0..n {
        out[i] = bytes[i] as c_char;
    }
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

fn replay_writer_append_impl(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    action: *const f32,
    action_len: u64,
    reward: f32,
) -> i32 {
    if !reward.is_finite() {
        return RX_ERROR_INVALID_ARGUMENT;
    }
    let writer = match get_replay_writer(id) {
        Ok(writer) => writer,
        Err(status) => return status,
    };
    let guard = match writer.lock() {
        Ok(guard) => guard,
        Err(_) => return RX_ERROR_INTERNAL,
    };
    let state = match make_observation(obs, obs_len, guard.obs_size, guard.device) {
        Ok(obs) => obs,
        Err(status) => return status,
    };
    let action = match make_action(action, action_len, guard.action_len, guard.device) {
        Ok(action) => action,
        Err(status) => return status,
    };
    let experience = Arc::new(Experience::new(
        guard.agent_id,
        guard.current_episode_id,
        state,
        Some(action),
        None,
        f64::from(reward),
        false,
    ));
    guard.buffer.append(experience, guard.gamma);
    RX_OK
}

fn replay_writer_stop_episode_impl(id: u64, obs: *const f32, obs_len: u64, reward: f32) -> i32 {
    if !reward.is_finite() {
        return RX_ERROR_INVALID_ARGUMENT;
    }
    let writer = match get_replay_writer(id) {
        Ok(writer) => writer,
        Err(status) => return status,
    };
    let mut guard = match writer.lock() {
        Ok(guard) => guard,
        Err(_) => return RX_ERROR_INTERNAL,
    };
    let state = match make_observation(obs, obs_len, guard.obs_size, guard.device) {
        Ok(obs) => obs,
        Err(status) => return status,
    };
    let experience = Arc::new(Experience::new(
        guard.agent_id,
        guard.current_episode_id,
        state,
        None,
        None,
        f64::from(reward),
        true,
    ));
    guard.buffer.append(experience, guard.gamma);
    guard.current_episode_id = Ulid::new();
    RX_OK
}

fn rnd_calc_reward_impl(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    is_episode_terminal: u32,
    observe: bool,
    out_reward: *mut f64,
) -> i32 {
    if !valid_flag(is_episode_terminal) {
        return RX_ERROR_INVALID_ARGUMENT;
    }
    if out_reward.is_null() {
        return RX_ERROR_NULL_POINTER;
    }
    let rnd = match get_rnd(id) {
        Ok(rnd) => rnd,
        Err(status) => return status,
    };
    let mut guard = match rnd.lock() {
        Ok(guard) => guard,
        Err(_) => return RX_ERROR_INTERNAL,
    };
    let state = match make_observation(obs, obs_len, guard.obs_size, guard.device) {
        Ok(obs) => obs,
        Err(status) => return status,
    };
    let experience = curiosity_experience(state, is_episode_terminal != 0);
    let reward = guard.rnd.calc_reward(Arc::clone(&experience));
    let reward = reward
        .to_device(Device::Cpu)
        .mean(Kind::Float)
        .double_value(&[]);
    if observe {
        guard.rnd.observe(experience);
    }
    unsafe {
        *out_reward = reward;
    }
    RX_OK
}

fn rnd_observe_impl(id: u64, obs: *const f32, obs_len: u64, is_episode_terminal: u32) -> i32 {
    if !valid_flag(is_episode_terminal) {
        return RX_ERROR_INVALID_ARGUMENT;
    }
    let rnd = match get_rnd(id) {
        Ok(rnd) => rnd,
        Err(status) => return status,
    };
    let mut guard = match rnd.lock() {
        Ok(guard) => guard,
        Err(_) => return RX_ERROR_INTERNAL,
    };
    let state = match make_observation(obs, obs_len, guard.obs_size, guard.device) {
        Ok(obs) => obs,
        Err(status) => return status,
    };
    guard
        .rnd
        .observe(curiosity_experience(state, is_episode_terminal != 0));
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
pub extern "C" fn rx_replay_buffer_config_default(
    out_config: *mut RxReplayBufferConfig,
    capacity: u64,
    n_steps: u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        write_default(out_config, default_replay_buffer_config(capacity, n_steps))
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_writer_config_default(
    out_config: *mut RxReplayWriterConfig,
    obs_size: u64,
    action_len: u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        write_default(
            out_config,
            default_replay_writer_config(obs_size, action_len),
        )
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_config_default(out_config: *mut RxRndConfig, obs_size: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        write_default(out_config, default_rnd_config(obs_size))
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

#[no_mangle]
pub extern "C" fn rx_dqn_create_with_paths(
    config: *const RxDqnConfig,
    save_path: *const c_char,
    load_path: *const c_char,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        create_from_config_with_paths(
            config,
            save_path,
            load_path,
            out_id,
            |config, save, load| create_dqn_with_paths(config, save, load),
        )
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_ppo_create_with_paths(
    config: *const RxPpoConfig,
    save_path: *const c_char,
    load_path: *const c_char,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        create_from_config_with_paths(
            config,
            save_path,
            load_path,
            out_id,
            |config, save, load| create_ppo_with_paths(config, save, load),
        )
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_sac_create_with_paths(
    config: *const RxSacConfig,
    save_path: *const c_char,
    load_path: *const c_char,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        create_from_config_with_paths(
            config,
            save_path,
            load_path,
            out_id,
            |config, save, load| create_sac_with_paths(config, save, load),
        )
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_buffer_create(
    config: *const RxReplayBufferConfig,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if config.is_null() || out_id.is_null() {
            return RX_ERROR_NULL_POINTER;
        }
        unsafe {
            *out_id = 0;
        }
        let config = unsafe { *config };
        if let Err(status) = validate_replay_buffer(&config) {
            return status;
        }
        let capacity = match to_usize(config.capacity) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let n_steps = match to_usize(config.n_steps) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let wrapper = ReplayBufferWrapper {
            buffer: Arc::new(ReplayBuffer::new(capacity, n_steps)),
            capacity,
            n_steps,
        };
        match insert_replay_buffer(wrapper) {
            Ok(id) => {
                unsafe {
                    *out_id = id;
                }
                RX_OK
            }
            Err(status) => status,
        }
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_sac_create_with_replay(
    config: *const RxSacConfig,
    replay_id: u64,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        rx_sac_create_with_replay_and_paths(
            config,
            replay_id,
            std::ptr::null(),
            std::ptr::null(),
            out_id,
        )
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_sac_create_with_replay_and_paths(
    config: *const RxSacConfig,
    replay_id: u64,
    save_path: *const c_char,
    load_path: *const c_char,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if config.is_null() || out_id.is_null() {
            return RX_ERROR_NULL_POINTER;
        }
        unsafe {
            *out_id = 0;
        }
        let config = unsafe { *config };
        if let Err(status) = validate_sac(&config) {
            return status;
        }
        let replay = match get_replay_buffer(replay_id) {
            Ok(replay) => replay,
            Err(status) => return status,
        };
        let replay_n_steps = match to_usize(config.replay_n_steps) {
            Ok(value) => value,
            Err(status) => return status,
        };
        if replay.n_steps != replay_n_steps
            || replay.capacity < to_usize(config.batch_size).unwrap_or(usize::MAX)
            || replay.capacity < to_usize(config.replay_start_size).unwrap_or(usize::MAX)
        {
            return RX_ERROR_INVALID_ARGUMENT;
        }
        let save_path = match optional_c_string(save_path) {
            Ok(path) => path,
            Err(status) => return status,
        };
        let load_path = match optional_c_string(load_path) {
            Ok(path) => path,
            Err(status) => return status,
        };
        match create_sac_with_replay_and_paths(
            &config,
            Arc::clone(&replay.buffer),
            save_path,
            load_path,
        )
        .and_then(insert_agent)
        {
            Ok(id) => {
                unsafe {
                    *out_id = id;
                }
                RX_OK
            }
            Err(status) => status,
        }
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_writer_create(
    replay_id: u64,
    config: *const RxReplayWriterConfig,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if config.is_null() || out_id.is_null() {
            return RX_ERROR_NULL_POINTER;
        }
        unsafe {
            *out_id = 0;
        }
        let config = unsafe { *config };
        if let Err(status) = validate_replay_writer(&config) {
            return status;
        }
        let replay = match get_replay_buffer(replay_id) {
            Ok(replay) => replay,
            Err(status) => return status,
        };
        let wrapper = ReplayWriterWrapper {
            buffer: Arc::clone(&replay.buffer),
            agent_id: Ulid::new(),
            current_episode_id: Ulid::new(),
            device: Device::cuda_if_available(),
            obs_size: match to_usize(config.obs_size) {
                Ok(value) => value,
                Err(status) => return status,
            },
            action_len: match to_usize(config.action_len) {
                Ok(value) => value,
                Err(status) => return status,
            },
            gamma: config.gamma,
        };
        match insert_replay_writer(wrapper) {
            Ok(id) => {
                unsafe {
                    *out_id = id;
                }
                RX_OK
            }
            Err(status) => status,
        }
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_create(config: *const RxRndConfig, out_id: *mut u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        rx_rnd_create_with_paths(config, std::ptr::null(), std::ptr::null(), out_id)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_create_with_paths(
    config: *const RxRndConfig,
    save_path: *const c_char,
    load_path: *const c_char,
    out_id: *mut u64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if config.is_null() || out_id.is_null() {
            return RX_ERROR_NULL_POINTER;
        }
        unsafe {
            *out_id = 0;
        }
        let config = unsafe { *config };
        let save_path = match optional_c_string(save_path) {
            Ok(path) => path,
            Err(status) => return status,
        };
        let load_path = match optional_c_string(load_path) {
            Ok(path) => path,
            Err(status) => return status,
        };
        match create_rnd_with_paths(&config, save_path, load_path).and_then(insert_rnd) {
            Ok(id) => {
                unsafe {
                    *out_id = id;
                }
                RX_OK
            }
            Err(status) => status,
        }
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
pub extern "C" fn rx_agent_statistics_len(id: u64, out_len: *mut u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() {
            return RX_ERROR_NULL_POINTER;
        }
        let wrapper = match get_agent(id) {
            Ok(wrapper) => wrapper,
            Err(status) => return status,
        };
        let guard = match wrapper.lock() {
            Ok(guard) => guard,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        let len = guard.agent.get_statistics().len();
        let len = match u64::try_from(len) {
            Ok(len) => len,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        unsafe {
            *out_len = len;
        }
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_agent_statistics(id: u64, out_stats: *mut RxStatistic, out_len: u64) -> i64 {
    catch_unwind(AssertUnwindSafe(|| {
        if out_stats.is_null() {
            return i64::from(RX_ERROR_NULL_POINTER);
        }
        let out_len = match to_usize(out_len) {
            Ok(value) => value,
            Err(status) => return i64::from(status),
        };
        let wrapper = match get_agent(id) {
            Ok(wrapper) => wrapper,
            Err(status) => return i64::from(status),
        };
        let guard = match wrapper.lock() {
            Ok(guard) => guard,
            Err(_) => return i64::from(RX_ERROR_INTERNAL),
        };
        let statistics = guard.agent.get_statistics();
        if out_len < statistics.len() {
            return i64::from(RX_ERROR_BUFFER_TOO_SMALL);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_stats, statistics.len()) };
        for (dst, (name, value)) in out.iter_mut().zip(statistics.iter()) {
            write_stat_name(name, &mut dst.name);
            dst.value = *value;
        }
        i64::try_from(statistics.len()).unwrap_or(i64::from(RX_ERROR_INTERNAL))
    }))
    .unwrap_or(i64::from(RX_ERROR_PANIC))
}

#[no_mangle]
pub extern "C" fn rx_agent_save(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let wrapper = match get_agent(id) {
            Ok(wrapper) => wrapper,
            Err(status) => return status,
        };
        let guard = match wrapper.lock() {
            Ok(guard) => guard,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        guard.agent.save();
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_agent_load(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let wrapper = match get_agent(id) {
            Ok(wrapper) => wrapper,
            Err(status) => return status,
        };
        let mut guard = match wrapper.lock() {
            Ok(guard) => guard,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        guard.agent.load();
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_buffer_len(id: u64, out_len: *mut u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() {
            return RX_ERROR_NULL_POINTER;
        }
        let replay = match get_replay_buffer(id) {
            Ok(replay) => replay,
            Err(status) => return status,
        };
        let len = match u64::try_from(replay.buffer.len()) {
            Ok(len) => len,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        unsafe {
            *out_len = len;
        }
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_buffer_clear(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let replay = match get_replay_buffer(id) {
            Ok(replay) => replay,
            Err(status) => return status,
        };
        replay.buffer.clear();
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_buffer_destroy(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if id == 0 {
            return RX_ERROR_INVALID_ARGUMENT;
        }
        if REPLAY_BUFFERS.remove(&id).is_some() {
            RX_OK
        } else {
            RX_ERROR_NOT_FOUND
        }
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_writer_append(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    action: *const f32,
    action_len: u64,
    reward: f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        replay_writer_append_impl(id, obs, obs_len, action, action_len, reward)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_writer_stop_episode(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    reward: f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        replay_writer_stop_episode_impl(id, obs, obs_len, reward)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_replay_writer_destroy(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if id == 0 {
            return RX_ERROR_INVALID_ARGUMENT;
        }
        if REPLAY_WRITERS.remove(&id).is_some() {
            RX_OK
        } else {
            RX_ERROR_NOT_FOUND
        }
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_calc_reward(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    out_reward: *mut f64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        rnd_calc_reward_impl(id, obs, obs_len, 0, false, out_reward)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_calc_reward_and_observe(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    is_episode_terminal: u32,
    out_reward: *mut f64,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        rnd_calc_reward_impl(id, obs, obs_len, is_episode_terminal, true, out_reward)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_observe(
    id: u64,
    obs: *const f32,
    obs_len: u64,
    is_episode_terminal: u32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        rnd_observe_impl(id, obs, obs_len, is_episode_terminal)
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_save(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let rnd = match get_rnd(id) {
            Ok(rnd) => rnd,
            Err(status) => return status,
        };
        let guard = match rnd.lock() {
            Ok(guard) => guard,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        guard.rnd.save();
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_load(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let rnd = match get_rnd(id) {
            Ok(rnd) => rnd,
            Err(status) => return status,
        };
        let mut guard = match rnd.lock() {
            Ok(guard) => guard,
            Err(_) => return RX_ERROR_INTERNAL,
        };
        guard.rnd.load();
        RX_OK
    }))
    .unwrap_or(RX_ERROR_PANIC)
}

#[no_mangle]
pub extern "C" fn rx_rnd_destroy(id: u64) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if id == 0 {
            return RX_ERROR_INVALID_ARGUMENT;
        }
        if RNDS.remove(&id).is_some() {
            RX_OK
        } else {
            RX_ERROR_NOT_FOUND
        }
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
    use std::ffi::CString;

    fn stat_name(stat: &RxStatistic) -> String {
        let bytes = stat
            .name
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8)
            .collect::<Vec<_>>();
        String::from_utf8(bytes).unwrap()
    }

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

    #[test]
    fn sac_agents_can_share_replay_buffer_and_report_statistics() {
        let replay_config = default_replay_buffer_config(64, 1);
        let mut replay_id = 0;
        assert_eq!(
            rx_replay_buffer_create(&replay_config, &mut replay_id),
            RX_OK
        );
        assert_ne!(replay_id, 0);

        let mut config = default_sac_config(4, 2);
        config.action_space = RX_ACTION_DISCRETE;
        config.replay_capacity = replay_config.capacity;
        config.replay_start_size = 4;
        config.batch_size = 4;
        config.replay_n_steps = replay_config.n_steps;

        let mut agent1 = 0;
        let mut agent2 = 0;
        assert_eq!(
            rx_sac_create_with_replay(&config, replay_id, &mut agent1),
            RX_OK
        );
        assert_eq!(
            rx_sac_create_with_replay(&config, replay_id, &mut agent2),
            RX_OK
        );
        assert_ne!(agent1, agent2);

        let obs = [0.1f32; 4];
        let mut out = [0.0f32; 1];
        for _ in 0..3 {
            assert_eq!(
                rx_agent_act_and_train(agent1, obs.as_ptr(), 4, 1.0, out.as_mut_ptr(), 1),
                1
            );
            assert_eq!(
                rx_agent_act_and_train(agent2, obs.as_ptr(), 4, 1.0, out.as_mut_ptr(), 1),
                1
            );
        }

        let mut replay_len = 0;
        assert_eq!(rx_replay_buffer_len(replay_id, &mut replay_len), RX_OK);
        assert!(replay_len >= 2);

        let mut stats_len = 0;
        assert_eq!(rx_agent_statistics_len(agent1, &mut stats_len), RX_OK);
        assert!(stats_len >= 4);

        let mut stats = vec![
            RxStatistic {
                name: [0; RX_STAT_NAME_LEN],
                value: 0.0,
            };
            stats_len as usize
        ];
        assert_eq!(
            rx_agent_statistics(agent1, stats.as_mut_ptr(), stats.len() as u64),
            stats_len as i64
        );
        assert!(stats
            .iter()
            .any(|stat| stat_name(stat) == "replay_buffer_len" && stat.value >= replay_len as f64));

        assert_eq!(rx_agent_destroy(agent1), RX_OK);
        assert_eq!(rx_agent_destroy(agent2), RX_OK);
        assert_eq!(rx_replay_buffer_destroy(replay_id), RX_OK);
    }

    #[test]
    fn replay_writer_feeds_external_policy_experience() {
        let replay_config = default_replay_buffer_config(16, 1);
        let mut replay_id = 0;
        assert_eq!(
            rx_replay_buffer_create(&replay_config, &mut replay_id),
            RX_OK
        );

        let writer_config = default_replay_writer_config(4, 2);
        let mut writer_id = 0;
        assert_eq!(
            rx_replay_writer_create(replay_id, &writer_config, &mut writer_id),
            RX_OK
        );
        assert_ne!(writer_id, 0);

        let obs = [0.1f32, 0.2, 0.3, 0.4];
        let action = [0.5f32, -0.5];
        assert_eq!(
            rx_replay_writer_append(
                writer_id,
                obs.as_ptr(),
                obs.len() as u64,
                action.as_ptr(),
                action.len() as u64,
                0.0,
            ),
            RX_OK
        );
        assert_eq!(
            rx_replay_writer_append(
                writer_id,
                obs.as_ptr(),
                obs.len() as u64,
                action.as_ptr(),
                action.len() as u64,
                1.0,
            ),
            RX_OK
        );
        assert_eq!(
            rx_replay_writer_stop_episode(writer_id, obs.as_ptr(), obs.len() as u64, 2.0),
            RX_OK
        );

        let mut replay_len = 0;
        assert_eq!(rx_replay_buffer_len(replay_id, &mut replay_len), RX_OK);
        assert!(replay_len >= 2);

        assert_eq!(rx_replay_buffer_clear(replay_id), RX_OK);
        assert_eq!(rx_replay_buffer_len(replay_id, &mut replay_len), RX_OK);
        assert_eq!(replay_len, 0);

        assert_eq!(rx_replay_writer_destroy(writer_id), RX_OK);
        assert_eq!(rx_replay_buffer_destroy(replay_id), RX_OK);
    }

    #[test]
    fn rnd_lifecycle_reward_observe_and_validation() {
        let mut config = default_rnd_config(4);
        config.feature_size = 8;
        config.hidden_layers = 1;
        config.hidden_size = 16;
        config.update_interval = 2;

        let mut rnd_id = 0;
        assert_eq!(rx_rnd_create(&config, &mut rnd_id), RX_OK);
        assert_ne!(rnd_id, 0);

        let obs = [0.1f32, 0.2, 0.3, 0.4];
        let mut reward = 0.0;
        assert_eq!(
            rx_rnd_calc_reward(rnd_id, obs.as_ptr(), obs.len() as u64, &mut reward),
            RX_OK
        );
        assert!(reward.is_finite());

        assert_eq!(
            rx_rnd_observe(rnd_id, obs.as_ptr(), obs.len() as u64, 0),
            RX_OK
        );
        assert_eq!(
            rx_rnd_calc_reward_and_observe(rnd_id, obs.as_ptr(), obs.len() as u64, 1, &mut reward),
            RX_OK
        );
        assert!(reward.is_finite());
        assert_eq!(
            rx_rnd_calc_reward_and_observe(rnd_id, obs.as_ptr(), obs.len() as u64, 2, &mut reward),
            RX_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(rx_rnd_save(rnd_id), RX_OK);
        assert_eq!(rx_rnd_load(rnd_id), RX_OK);
        assert_eq!(rx_rnd_destroy(rnd_id), RX_OK);
        assert_eq!(rx_rnd_destroy(rnd_id), RX_ERROR_NOT_FOUND);
    }

    #[test]
    fn agents_can_be_created_with_save_and_load_paths() {
        let mut config = default_dqn_config(4, 2);
        config.batch_size = 4;
        config.replay_capacity = 32;
        let path = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("ffi-tests")
            .join(format!("dqn-{}.ot", Ulid::new()));
        let path = path.to_string_lossy().into_owned();
        let c_path = CString::new(path.clone()).unwrap();

        let mut id = 0;
        assert_eq!(
            rx_dqn_create_with_paths(&config, c_path.as_ptr(), std::ptr::null(), &mut id),
            RX_OK
        );
        assert_eq!(rx_agent_save(id), RX_OK);
        assert!(std::path::Path::new(&path).exists());
        assert_eq!(rx_agent_destroy(id), RX_OK);

        let mut loaded_id = 0;
        assert_eq!(
            rx_dqn_create_with_paths(&config, std::ptr::null(), c_path.as_ptr(), &mut loaded_id),
            RX_OK
        );
        assert_eq!(rx_agent_load(loaded_id), RX_OK);
        assert_eq!(rx_agent_destroy(loaded_id), RX_OK);

        let _ = std::fs::remove_file(path);
    }
}
