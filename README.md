# About ReinforceX
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

- DQN: Double-DQN style target network, n-step replay, epsilon-greedy
  exploration, optional reward-based selector, shared replay buffer support.
- PPO: clipped policy objective, GAE, value clipping, entropy regularization,
  discrete, multi-branch discrete, and Gaussian policies.
- SAC: continuous and discrete Soft Actor-Critic, twin critics, soft target
  updates, automatic temperature updates for discrete policies, and component
  checkpointing.
- RND: Random Network Distillation with a fixed random target network, a
  trainable predictor, batched predictor updates, and predictor/target
  checkpointing.

Core building blocks:

- Models: `FCQNetwork`, `FCSoftmaxPolicy`, `FCSoftmaxPolicyWithValue`,
  `FCGaussianPolicy`, `FCGaussianPolicyWithValue`, `FCRNDModel`.
- Distributions: `SoftmaxDistribution`, `MultiSoftmaxDistribution`,
  `GaussianDistribution`.
- Memory: `ReplayBuffer` with n-step transitions, `OnPolicyBuffer`.
- Exploration and selection: `EpsilonGreedy`, `RewardBasedSelector`.
- Curiosity: `RND` computes intrinsic rewards and periodically trains its
  predictor from buffered observations.
- FFI: DQN and PPO can be created and trained through a C-compatible API.

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
let transition_buffer = Arc::new(ReplayBuffer::new(replay_buffer_capacity, n_steps));

let mut agent = DQN::new(
    model,
    transition_buffer,
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
use reinforcex::curiousity::RND;
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
the `BaseCuriousity` trait. RND checkpoints contain `rnd_predictor.ot` and
`rnd_target.ot` in the configured directory.

# Sample experiments
The sample experiments call Gymnasium environments through FastAPI servers.
Docker Compose starts ten environment servers on ports `8001` to `8010`.

```sh
docker compose -f sample_env/docker-compose.yml up -d --build
```

Run CartPole with DQN:

```sh
cargo run -p reinforcex --features cpu -- --env cartpole --algo dqn
```

Run CartPole with PPO:

```sh
cargo run -p reinforcex --features cpu -- --env cartpole --algo ppo
```

Run CartPole with discrete SAC using four parallel environment servers:

```sh
cargo run -p reinforcex --features cpu -- --env cartpole --algo sac --parallel 4
```

Run LunarLanderContinuous with continuous SAC:

```sh
cargo run -p reinforcex --features cpu -- --env lunar --algo sac --parallel 4
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

Run Ant with PPO:

```sh
cargo run -p reinforcex --features cpu -- --env ant --algo ppo
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
ReinforceX also provides a small Foreign Function Interface (FFI) crate for
embedding agents from external runtimes such as C, C++, C#, or Unity.

Build the dynamic library:

```sh
cargo build -p reinforcex_ffi --release
```

The generated library is named `reinforcex` with the platform-specific dynamic
library extension, for example `reinforcex.dll`, `libreinforcex.so`, or
`libreinforcex.dylib`.

## Overview

- All agents are managed internally and referenced through a `u64` ID.
- The public FFI functions catch panics and return silently on invalid inputs.
- All sizes use `u64` for ABI-friendly boundaries.
- The caller owns input and output buffer allocation.
- `agent_type = 0` creates DQN; any other value creates PPO.

## Data Structures

### AgentConfig

```c
typedef struct {
    uint32_t agent_type;

    uint64_t obs_size;
    uint64_t action_size;
    double learning_rate;
    double gamma;

    uint64_t batch_size;
    uint64_t buffer_size;
    double epsilon_start;
    double epsilon_end;
    uint64_t epsilon_decay;

    double lambda;
    uint64_t update_interval;
    uint64_t epoch;
    uint64_t minibatch_size;
    double clip_eps;
} AgentConfig;
```

| Field | Description |
|------|-------------|
| `agent_type` | `0 = DQN`, otherwise PPO |
| `obs_size` | Observation vector size |
| `action_size` | Action space size |
| `learning_rate` | Optimizer learning rate |
| `gamma` | Discount factor |
| `batch_size` | DQN batch size |
| `buffer_size` | DQN replay buffer size |
| `epsilon_start` | Initial epsilon for DQN |
| `epsilon_end` | Final epsilon for DQN |
| `epsilon_decay` | Epsilon decay steps for DQN |
| `lambda` | PPO GAE lambda |
| `update_interval` | PPO update interval |
| `epoch` | PPO training epochs |
| `minibatch_size` | PPO minibatch size |
| `clip_eps` | PPO clipping epsilon |

## Functions

### rx_agent_create

```c
uint64_t rx_agent_create(const AgentConfig* config);
```

Creates a new agent and returns its ID. Returns `0` on failure.

### rx_agent_act_and_train

```c
void rx_agent_act_and_train(
    uint64_t id,
    const float* obs,
    uint64_t obs_len,
    float reward,
    float* out,
    uint64_t out_len
);
```

Performs action selection and one training step. DQN writes one scalar action.
PPO writes a vector action and truncates to `out_len` if the output buffer is
smaller than the action tensor.

### rx_agent_stop_episode

```c
void rx_agent_stop_episode(
    uint64_t id,
    const float* obs,
    uint64_t obs_len,
    float reward
);
```

Signals the end of an episode and performs the final training step.

### rx_agent_destroy

```c
void rx_agent_destroy(uint64_t id);
```

Destroys the agent for the given ID. Calling it with an unknown ID is a no-op.

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
