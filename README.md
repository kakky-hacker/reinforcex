# ReinforceX - Intelligence embedded in the environment.
ReinforceX (ReX) is an early-stage deep reinforcement learning framework built
in Rust. It is designed as a Rust-first playground for implementing,
experimenting with, and eventually productionizing reinforcement learning
agents without making Python the core runtime.

The project currently focuses on:

- a small, readable core for value-based, policy-based, and actor-critic
  algorithms;
- neural-network policies and Q-functions backed by `tch` / libtorch;
- intrinsic-motivation modules for curiosity-driven exploration;
- replay and on-policy buffers that can be shared across training workers;
- sample Gymnasium environments exposed through a simple HTTP server;
- an optional C ABI for embedding agents from C, C++, C#, Unity, or other
  runtimes.

Advantages of Rust for this project:

- ownership and RAII make long-running training jobs easier to reason about;
- `Send` / `Sync` boundaries make parallel training explicit;
- native binaries are a good fit for simulators, games, robotics, and embedded
  integrations;
- Rust can still use libtorch through `tch`, so the project can combine systems
  programming ergonomics with modern tensor operations.

ReinforceX is not yet a stable 1.0 API. Contributions are welcome, especially
around algorithms, documentation, benchmark environments, test coverage, and
safe public API design.

# Package
crates.io: https://crates.io/crates/reinforcex

```sh
cargo add reinforcex
```

The default `cpu` feature enables `torch-sys` with `download-libtorch`.

```toml
[dependencies]
reinforcex = "0.0.5"
```

For CUDA experiments, build with the `cuda` feature and make sure your local
libtorch / CUDA runtime is visible to `tch`. On Windows, `load_cuda_dlls()` also
checks `TORCH_CUDA_DLL` when the `cuda` feature is enabled.

# Algorithms
Implemented agents and exploration modules:

- DQN ([Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)):
  Double-DQN style target network, n-step replay, epsilon-greedy exploration,
  optional reward-based selector, shared replay buffer support.
- PPO ([Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)):
  clipped policy objective, GAE, value clipping, entropy regularization,
  discrete, multi-branch discrete, and Gaussian policies.
- SAC ([Soft Actor-Critic](https://arxiv.org/abs/1801.01290),
  [Discrete SAC](https://arxiv.org/abs/1910.07207)): continuous and discrete
  Soft Actor-Critic, twin critics, soft target updates, automatic temperature
  updates for discrete policies, and component checkpointing.
- RND ([Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)):
  Random Network Distillation with a fixed random target network, a trainable
  predictor, batched predictor updates, and predictor/target checkpointing.

Core building blocks:

- Models: `FCQNetwork`, `FCSoftmaxPolicy`, `FCSoftmaxPolicyWithValue`,
  `FCGaussianPolicy`, `FCGaussianPolicyWithValue`, `FCRNDModel`.
- Distributions: `SoftmaxDistribution`, `MultiSoftmaxDistribution`,
  `GaussianDistribution`.
- Memory: `ReplayBuffer` with n-step transitions, `OnPolicyBuffer`.
- Exploration and selection: `EpsilonGreedy`, `RewardBasedSelector`.
- Curiosity: `RND` computes intrinsic rewards and periodically trains its
  predictor from buffered observations.
- FFI: DQN, PPO, and SAC can be created and trained through a C-compatible API.

# API
Instantiate a DQN agent.

```rust
use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::memory::ReplayBuffer;
use reinforcex::models::FCQNetwork;
use std::sync::Arc;
use tch::{nn, nn::OptimizerConfig, Device};

let device = Device::cuda_if_available();
let vs = nn::VarStore::new(device);
let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();

let n_input_channels = 4;
let action_size = 2;
let n_hidden_layers = 2;
let n_hidden_channels = 128;

let model = Box::new(FCQNetwork::new(
    vs,
    n_input_channels,
    action_size,
    n_hidden_layers,
    n_hidden_channels,
));

let gamma = 0.97;
let n_steps = 3;
let batch_size = 16;
let update_interval = 8;
let target_update_interval = 100;
let replay_buffer_capacity = 2_000;

let explorer = EpsilonGreedy::new(0.5, 0.1, 50_000);
let replay_buffer = Arc::new(ReplayBuffer::new(replay_buffer_capacity, n_steps));

let mut agent = DQN::new(
    model,
    replay_buffer,
    optimizer,
    action_size as usize,
    batch_size,
    update_interval,
    target_update_interval,
    Box::new(explorer),
    None,
    gamma,
    Some("models/dqn_latest.ot".to_string()),
    None,
);
```

Common agent methods are provided by `BaseAgent`.

```rust
fn act(&self, obs: &Tensor) -> Tensor;
fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;
fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64);
fn get_statistics(&self) -> Vec<(String, f64)>;
fn save(&self);
fn load(&mut self);
```

Pseudo code for training:

```rust
for episode in 0..max_episode {
    let mut reward = 0.0;

    for step in 0..max_step {
        let action = agent.act_and_train(&obs, reward);
        let (next_obs, next_reward, done) = env.step(action);

        obs = next_obs;
        reward = next_reward;

        if done {
            agent.stop_episode_and_train(&obs, reward);
            break;
        }
    }
}
```

Pseudo code for parallel learning:

```rust
use rayon::prelude::*;
use std::sync::Arc;

let buffer = Arc::new(ReplayBuffer::new(1_000, 1));

(0..n_threads).into_par_iter().for_each(|agent_id| {
    let (model, optimizer, explorer) = build_agent_components();

    let mut agent = DQN::new(
        model,
        Arc::clone(&buffer),
        optimizer,
        action_size,
        batch_size,
        update_interval,
        target_update_interval,
        Box::new(explorer),
        None,
        gamma,
        Some(format!("models/dqn_{agent_id}.ot")),
        None,
    );

    for episode in 0..max_episode {
        // Run the same training loop as above.
    }
});
```

`build_agent_components()` is a placeholder for creating a separate model,
optimizer, and explorer per worker. Share only the replay buffer or other
explicitly thread-safe state.

## Random Network Distillation

RND assigns a larger intrinsic reward to observations for which a trainable
predictor does not yet match the output of a fixed, randomly initialized target
network. As observations become familiar, predictor error decreases and so does
their curiosity reward.

Create an RND module with separate predictor and target variable stores. The
optimizer must be built from the predictor variable store after the model has
registered its layers.

```rust
use reinforcex::curiosity::RND;
use reinforcex::models::FCRNDModel;
use tch::{nn, nn::OptimizerConfig, Device};

let device = Device::cuda_if_available();
let observation_size = 8;

let rnd_model = FCRNDModel::new(
    nn::VarStore::new(device),
    nn::VarStore::new(device),
    observation_size,
    128, // feature size
    2,   // hidden layers
    256, // hidden channels
);
let rnd_optimizer = nn::Adam::default()
    .build(rnd_model.predictor_var_store(), 1e-4)
    .unwrap();

let mut curiosity = RND::new(
    Box::new(rnd_model),
    rnd_optimizer,
    128, // observations per predictor update
    Some("models/lunar_rnd".to_string()),
    None,
);
```

`RND::calc_reward` evaluates predictor error without gradients.
`RND::observe` buffers the state and updates the predictor whenever
`update_interval` observations have accumulated. Both methods are provided by
the `Basecuriosity` trait. RND checkpoints contain `rnd_predictor.ot` and
`rnd_target.ot` in the configured directory.

# Sample experiments
The sample experiments call Gymnasium environments through FastAPI servers.
Docker Compose starts ten environment servers on ports `8001` to `8010`.

```sh
docker compose -f sample_env/docker-compose.yml up -d --build
```

Run CartPole with DQN using four parallel environments and one shared replay
buffer:

```sh
cargo run -p reinforcex --features cpu -- --env cartpole --algo dqn --parallel 4
```

Run CartPole with four independent PPO workers:

```sh
cargo run -p reinforcex --features cpu -- --env cartpole --algo ppo --parallel 4
```

Run CartPole with discrete SAC using four parallel environment servers:

```sh
cargo run -p reinforcex --features cpu -- --env cartpole --algo sac --parallel 4
```

Run LunarLanderContinuous with continuous SAC:

```sh
cargo run -p reinforcex --features cpu -- --env lunar --algo sac --parallel 4
```

Run discrete LunarLander with DQN using four parallel environments and one
shared replay buffer:

```sh
cargo run -p reinforcex --features cpu -- --env lunar --algo dqn --parallel 4
```

Run discrete LunarLander with PPO and RND curiosity using four parallel
environment servers:

```sh
cargo run -p reinforcex --features cpu -- \
  --env lunar \
  --algo ppo-rnd \
  --parallel 4 \
  --save-path "models/lunar_ppo_rnd_{agent_id}.ot"
```

Each worker owns an independent PPO agent and RND predictor. Port `8001` is
used by agent 0, `8002` by agent 1, and so on. Make sure the corresponding
environment servers are running before increasing `--parallel`.

To share one RND predictor across all PPO workers, use `ppo-shared-rnd`:

```sh
cargo run -p reinforcex --features cpu -- \
  --env lunar \
  --algo ppo-shared-rnd \
  --parallel 4 \
  --save-path "models/lunar_ppo_shared_rnd_{agent_id}.ot"
```

The PPO agents remain independent, while intrinsic-reward calculation and RND
predictor updates use one `Arc<Mutex<RND>>`. This lets observations from every
environment train the same predictor. RND access is serialized by the mutex;
environment stepping and PPO updates still run in parallel.

Run Ant with four independent PPO workers:

```sh
cargo run -p reinforcex --features cpu -- --env ant --algo ppo --parallel 4
```

Use `--save-path` and `--load-path` to persist models. Multi-agent samples can
include `{agent_id}` in the path.

```sh
cargo run -p reinforcex --features cpu -- \
  --env cartpole \
  --algo dqn \
  --save-path "models/cartpole_dqn_{agent_id}.ot" \
  --load-path "models/cartpole_dqn_{agent_id}.ot"
```

For SAC, a single save path expands into component checkpoints such as actor,
critic1, critic2, and temperature files.

For PPO+RND, the PPO model uses the configured agent path. Its RND checkpoint is
stored beside it using the same path with `.rnd` appended. For example,
`models/lunar_ppo_rnd_0.ot` is paired with the directory
`models/lunar_ppo_rnd_0.ot.rnd`. Pass the same base path through `--load-path`
to restore both components.

The shared-RND example replaces `{agent_id}` with `shared` for the RND
checkpoint. With the command above, PPO checkpoints use agent-specific paths
and the shared predictor is stored in
`models/lunar_ppo_shared_rnd_shared.ot.rnd`.

Run the hybrid discrete LunarLander example with four PPO explorers and two
SAC learners/actors:

```sh
cargo run -p reinforcex --features cpu -- \
  --env lunar \
  --algo ppo-sac-rnd \
  --ppo-agents 4 \
  --sac-agents 2 \
  --save-path "models/lunar_hybrid_{role}_{agent_id}.ot"
```

This starts one environment per agent: PPO uses ports `8001` through `8004`
and SAC uses `8005` and `8006`. All workers write extrinsic-reward transitions
to one `Arc<ReplayBuffer>`, and every SAC instance samples from it. PPO alone
adds RND intrinsic reward to its on-policy update; intrinsic rewards are never
stored in the SAC replay buffer. The PPO workers share one `Arc<Mutex<RND>>`,
but each has a deterministic random binary mask over the 128 RND error
features. This gives each explorer a different novelty projection while all of
them continue training the same predictor. Use at most ten total agents with
the supplied Docker Compose file.

`{role}` expands to `ppo`, `sac`, or `rnd`; `{agent_id}` expands to the worker
index. If the placeholders are omitted, role and agent suffixes are added
automatically to avoid checkpoint collisions.

The same hybrid topology is available for continuous `Ant-v5`. PPO and SAC
both use its eight-dimensional `[-1, 1]` action space; SAC applies the tanh
transform internally:

```sh
cargo run -p reinforcex --features cpu -- \
  --env ant \
  --algo ppo-sac-rnd \
  --ppo-agents 4 \
  --sac-agents 2 \
  --save-path "models/ant_hybrid_{role}_{agent_id}.ot"
```

As in the LunarLander example, PPO workers share one RND model but use distinct
fixed random feature masks. Only raw Ant environment rewards enter the shared
SAC replay buffer; masked intrinsic rewards affect PPO updates only.

Stop the sample environment servers:

```sh
docker compose -f sample_env/docker-compose.yml down
```

<img width="597" alt="CartPole training sample" src="https://github.com/user-attachments/assets/b8c0606b-ec11-4b5a-b7fc-3070ad327d72" />

# Unit test
Run all Rust unit tests from the workspace root:

```sh
cargo test --workspace
```

The core unit tests exercise agents, models, curiosity modules, probability
distributions, memory buffers, selectors, and the FFI wrapper. The Docker-based
Gymnasium server is only required for the sample experiments above.

# FFI
ReinforceX provides a C-compatible API for embedding DQN, PPO, SAC, shared replay
buffers, and RND curiosity modules from C, C++, C#, Unity, Python `ctypes`/CFFI,
and other runtimes. The canonical declarations are in
[`ffi/include/reinforcex.h`](ffi/include/reinforcex.h).

Build the dynamic library:

```sh
cargo build -p reinforcex_ffi --release
```

The generated library is named `reinforcex` with the platform-specific dynamic
library extension, for example `reinforcex.dll`, `libreinforcex.so`, or
`libreinforcex.dylib`.

## FFI design

- All long-lived objects are owned by the Rust library and referenced through a
  non-zero `uint64_t` handle.
- DQN, PPO, SAC, replay buffers, replay writers, and RND modules each have their
  own create/destroy lifecycle.
- PPO and SAC support `RX_ACTION_DISCRETE` and `RX_ACTION_CONTINUOUS`.
- Public functions catch Rust panics and return `RX_ERROR_PANIC` instead of
  unwinding across the ABI.
- Calls for one agent, replay writer, or RND handle are serialized internally.
  Different handles may be used from different host threads.
- The caller owns input and output buffer allocation.
- Observation buffers must contain exactly `obs_size` finite `float` values.
- Discrete agents write one action value. Continuous agents write `action_size`
  values.
- `const char *save_path` and `const char *load_path` are optional. Pass `NULL`
  or an empty string to disable that path. Non-null paths must be valid UTF-8.

This API replaces the old catch-all `AgentConfig` and `rx_agent_create` API. Use
a typed config and its matching create function instead.

## Constants and status codes

```c
enum {
    RX_OK = 0,
    RX_ERROR_NULL_POINTER = -1,
    RX_ERROR_INVALID_ARGUMENT = -2,
    RX_ERROR_NOT_FOUND = -3,
    RX_ERROR_BUFFER_TOO_SMALL = -4,
    RX_ERROR_PANIC = -5,
    RX_ERROR_INTERNAL = -6
};

enum {
    RX_ACTION_DISCRETE = 0,
    RX_ACTION_CONTINUOUS = 1
};

enum {
    RX_STAT_NAME_LEN = 64
};
```

| Status | Meaning |
|---|---|
| `RX_OK` (`0`) | Success |
| `RX_ERROR_NULL_POINTER` (`-1`) | A required pointer was null |
| `RX_ERROR_INVALID_ARGUMENT` (`-2`) | A size, enum, flag, path string, or numeric setting was invalid |
| `RX_ERROR_NOT_FOUND` (`-3`) | The requested handle does not exist |
| `RX_ERROR_BUFFER_TOO_SMALL` (`-4`) | The caller-provided output buffer is too small |
| `RX_ERROR_PANIC` (`-5`) | A Rust panic was caught at the ABI boundary |
| `RX_ERROR_INTERNAL` (`-6`) | An optimizer, mutex, tensor shape, conversion, or handle operation failed |

Functions returning `int32_t` return one of the status codes above. Functions
returning `int64_t` return a non-negative count on success, or a negative
`RX_ERROR_*` value on failure.

## FFI data structures

All typed agent configs begin with the common network and environment settings:

```c
typedef struct RxAgentConfig {
    uint64_t obs_size;
    uint64_t action_size;
    uint64_t hidden_layers;
    uint64_t hidden_size;
    double gamma;
} RxAgentConfig;
```

Algorithm configs:

```c
typedef struct RxDqnConfig {
    RxAgentConfig agent;
    double learning_rate;
    uint64_t batch_size;
    uint64_t replay_capacity;
    uint64_t replay_n_steps;
    uint64_t update_interval;
    uint64_t target_update_interval;
    double epsilon_start;
    double epsilon_end;
    uint64_t epsilon_decay_steps;
} RxDqnConfig;

typedef struct RxPpoConfig {
    RxAgentConfig agent;
    uint32_t action_space;
    double learning_rate;
    double gae_lambda;
    uint64_t update_interval;
    uint64_t epochs;
    uint64_t minibatch_size;
    double policy_clip_epsilon;
    double value_clip_range;
    double value_loss_coefficient;
    double entropy_coefficient;
    uint32_t standardize_gae;
    double min_action;
    double max_action;
    double min_variance;
} RxPpoConfig;

typedef struct RxSacConfig {
    RxAgentConfig agent;
    uint32_t action_space;
    double actor_learning_rate;
    double critic_learning_rate;
    uint64_t replay_capacity;
    uint64_t replay_start_size;
    uint64_t batch_size;
    uint64_t replay_n_steps;
    uint64_t update_interval;
    uint64_t target_update_interval;
    double tau;
    double alpha;
    double min_variance;
    uint32_t squash_action;
} RxSacConfig;
```

Replay, RND, and statistics structs:

```c
typedef struct RxReplayBufferConfig {
    uint64_t capacity;
    uint64_t n_steps;
} RxReplayBufferConfig;

typedef struct RxReplayWriterConfig {
    uint64_t obs_size;
    uint64_t action_len;
    double gamma;
} RxReplayWriterConfig;

typedef struct RxRndConfig {
    uint64_t obs_size;
    uint64_t feature_size;
    uint64_t hidden_layers;
    uint64_t hidden_size;
    double learning_rate;
    uint64_t update_interval;
} RxRndConfig;

typedef struct RxStatistic {
    char name[RX_STAT_NAME_LEN];
    double value;
} RxStatistic;
```

Notes:

- A `uint32_t` flag must be `0` or `1`.
- `RxPpoConfig.action_space` and `RxSacConfig.action_space` must be
  `RX_ACTION_DISCRETE` or `RX_ACTION_CONTINUOUS`.
- For continuous PPO, `min_action`, `max_action`, and `min_variance` configure
  the Gaussian policy. They are ignored for discrete PPO.
- Continuous SAC uses a diagonal Gaussian policy. If `squash_action` is `1`, the
  action is tanh-squashed to `[-1, 1]`. `min_variance` is ignored for discrete
  SAC.
- `RxReplayWriterConfig.action_len` is the action tensor length written into a
  replay buffer. Use `1` for discrete actions and the action dimension for
  continuous actions.
- `RxStatistic.name` is null-terminated when shorter than `RX_STAT_NAME_LEN`.
  Longer names are truncated to fit.

## Default config helpers

Use default helpers to initialize every field, then override only what your
application needs.

```c
int32_t rx_dqn_config_default(
    RxDqnConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_ppo_config_default(
    RxPpoConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_sac_config_default(
    RxSacConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_replay_buffer_config_default(
    RxReplayBufferConfig *out_config,
    uint64_t capacity,
    uint64_t n_steps);

int32_t rx_replay_writer_config_default(
    RxReplayWriterConfig *out_config,
    uint64_t obs_size,
    uint64_t action_len);

int32_t rx_rnd_config_default(
    RxRndConfig *out_config,
    uint64_t obs_size);
```

Example:

```c
RxSacConfig config;
int32_t status = rx_sac_config_default(&config, 8, 2);
config.action_space = RX_ACTION_CONTINUOUS;
config.replay_start_size = 5000;

uint64_t agent_id = 0;
if (status == RX_OK) {
    status = rx_sac_create(&config, &agent_id);
}
```

## Agent creation APIs

```c
int32_t rx_dqn_create(const RxDqnConfig *config, uint64_t *out_id);
int32_t rx_ppo_create(const RxPpoConfig *config, uint64_t *out_id);
int32_t rx_sac_create(const RxSacConfig *config, uint64_t *out_id);

int32_t rx_dqn_create_with_paths(
    const RxDqnConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_paths(
    const RxPpoConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_sac_create_with_paths(
    const RxSacConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_sac_create_with_replay(
    const RxSacConfig *config,
    uint64_t replay_id,
    uint64_t *out_id);

int32_t rx_sac_create_with_replay_and_paths(
    const RxSacConfig *config,
    uint64_t replay_id,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);
```

| Function | Purpose |
|---|---|
| `rx_dqn_create` | Creates a DQN agent with its own replay buffer. |
| `rx_ppo_create` | Creates a PPO agent with its own on-policy buffer. |
| `rx_sac_create` | Creates a SAC agent with its own replay buffer. |
| `rx_*_create_with_paths` | Creates an agent with optional save/load checkpoint paths. |
| `rx_sac_create_with_replay` | Creates a SAC agent that uses an existing shared replay buffer. |
| `rx_sac_create_with_replay_and_paths` | Same as above, with optional save/load checkpoint paths. |

On success, create functions return `RX_OK` and write a non-zero handle to
`out_id`. On failure, `out_id` is set to zero. Shared SAC replay creation checks
that the shared replay buffer has the same `n_steps` as the SAC config and is
large enough for `batch_size` and `replay_start_size`.

## Agent action, training, statistics, and lifecycle APIs

```c
int64_t rx_agent_act(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float *out,
    uint64_t out_len);

int64_t rx_agent_act_and_train(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float reward,
    float *out,
    uint64_t out_len);

int32_t rx_agent_stop_episode(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float reward);

int32_t rx_agent_statistics_len(uint64_t id, uint64_t *out_len);

int64_t rx_agent_statistics(
    uint64_t id,
    RxStatistic *out_stats,
    uint64_t out_len);

int32_t rx_agent_save(uint64_t id);
int32_t rx_agent_load(uint64_t id);
int32_t rx_agent_destroy(uint64_t id);
```

| Function | Purpose |
|---|---|
| `rx_agent_act` | Selects an action without adding a transition or updating the agent. |
| `rx_agent_act_and_train` | Selects an action, records the previous transition, and updates when due. |
| `rx_agent_stop_episode` | Records the terminal observation/reward and ends the current episode. |
| `rx_agent_statistics_len` | Returns how many statistics entries are currently available. |
| `rx_agent_statistics` | Writes `RxStatistic` entries into the caller-provided buffer. |
| `rx_agent_save` | Saves the agent using the `save_path` supplied at creation. No-ops if no path was supplied. |
| `rx_agent_load` | Loads the agent using the `load_path` supplied at creation. No-ops if no path was supplied. |
| `rx_agent_destroy` | Releases the agent handle from the Rust registry. |

`rx_agent_act` and `rx_agent_act_and_train` return the number of `float` values
written, or a negative `RX_ERROR_*` code. An undersized output buffer returns
`RX_ERROR_BUFFER_TOO_SMALL` without a partial write.

As in the Rust `BaseAgent` API, the `reward` passed to
`rx_agent_act_and_train` is the reward received after the previously selected
action. At the end of an episode, pass the final observation and final reward to
`rx_agent_stop_episode`.

## Replay buffer APIs

```c
int32_t rx_replay_buffer_create(
    const RxReplayBufferConfig *config,
    uint64_t *out_id);

int32_t rx_replay_buffer_len(uint64_t id, uint64_t *out_len);
int32_t rx_replay_buffer_clear(uint64_t id);
int32_t rx_replay_buffer_destroy(uint64_t id);
```

| Function | Purpose |
|---|---|
| `rx_replay_buffer_create` | Creates a replay buffer handle that can be shared by SAC agents or replay writers. |
| `rx_replay_buffer_len` | Reads the number of fully linkable experiences currently stored. |
| `rx_replay_buffer_clear` | Clears stored experiences and per-episode n-step state. |
| `rx_replay_buffer_destroy` | Removes the replay buffer handle from the registry. Existing agents/writers keep their `Arc` reference alive. |

Use shared replay buffers for multi-agent SAC training or for feeding
experiences collected by another policy into SAC.

## Replay writer APIs

Replay writers let an external policy, such as a Python-managed PPO agent, write
experience into a shared replay buffer. This is the intended bridge for
experiments where PPO explores while SAC learns from a shared replay buffer.

```c
int32_t rx_replay_writer_create(
    uint64_t replay_id,
    const RxReplayWriterConfig *config,
    uint64_t *out_id);

int32_t rx_replay_writer_append(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    const float *action,
    uint64_t action_len,
    float reward);

int32_t rx_replay_writer_stop_episode(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float reward);

int32_t rx_replay_writer_destroy(uint64_t id);
```

| Function | Purpose |
|---|---|
| `rx_replay_writer_create` | Creates a writer attached to an existing replay buffer. |
| `rx_replay_writer_append` | Appends a non-terminal decision point: observation, action, and reward. |
| `rx_replay_writer_stop_episode` | Appends the terminal observation/reward and starts a new episode for that writer. |
| `rx_replay_writer_destroy` | Releases the writer handle. |

The writer follows the same reward convention as `rx_agent_act_and_train`: pass
the reward received after the previously selected action. A common host loop is:

1. Choose an action for the initial observation.
2. Call `rx_replay_writer_append(..., initial_obs, action, 0.0)`.
3. Step the environment.
4. Choose the next action and call `rx_replay_writer_append(..., next_obs,
   next_action, reward_from_previous_step)`.
5. On terminal, call `rx_replay_writer_stop_episode(..., final_obs,
   final_reward)`.

For discrete actions, write one float containing the action index. For continuous
actions, write one float per action dimension.

## RND APIs

```c
int32_t rx_rnd_create(const RxRndConfig *config, uint64_t *out_id);

int32_t rx_rnd_create_with_paths(
    const RxRndConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_rnd_calc_reward(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    double *out_reward);

int32_t rx_rnd_calc_reward_and_observe(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    uint32_t is_episode_terminal,
    double *out_reward);

int32_t rx_rnd_observe(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    uint32_t is_episode_terminal);

int32_t rx_rnd_save(uint64_t id);
int32_t rx_rnd_load(uint64_t id);
int32_t rx_rnd_destroy(uint64_t id);
```

| Function | Purpose |
|---|---|
| `rx_rnd_create` | Creates an RND curiosity module. |
| `rx_rnd_create_with_paths` | Creates an RND module with optional save/load paths. |
| `rx_rnd_calc_reward` | Computes intrinsic reward for an observation without training RND. |
| `rx_rnd_calc_reward_and_observe` | Computes intrinsic reward and then observes the same state for periodic RND updates. |
| `rx_rnd_observe` | Observes a state for periodic RND updates without returning reward. |
| `rx_rnd_save` | Saves RND using the `save_path` supplied at creation. No-ops if no path was supplied. |
| `rx_rnd_load` | Loads RND using the `load_path` supplied at creation. No-ops if no path was supplied. |
| `rx_rnd_destroy` | Releases the RND handle. |

`is_episode_terminal` must be `0` or `1`. To share RND across multiple PPO
workers, create one RND handle and call it from each worker. To use independent
RND modules, create one RND handle per worker. Per-worker curiosity masks can be
applied in the host runtime by scaling or dropping the returned intrinsic reward
before adding it to the PPO reward.

# Contributing
ReinforceX is a good place to contribute if you are interested in Rust,
reinforcement learning, libtorch bindings, simulator integration, or FFI.

Useful contribution areas:

- algorithm implementations and correctness tests;
- benchmark scripts and reproducible training results;
- safer public APIs around tensor shapes, device placement, and errors;
- documentation for model construction and environment integration;
- CI for Rust tests, formatting, and platform-specific FFI builds.

Before opening a pull request, please run:

```sh
cargo fmt --all -- --check
cargo test --workspace
```

# License
MIT License (https://github.com/kakky-hacker/reinforcex/blob/master/LICENSE)
