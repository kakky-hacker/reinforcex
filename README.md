# About ReinforceX
ReinforceX (ReX) is a deep reinforcement learning framework built entirely from scratch in Rust.
We plan to implement various reinforcement learning algorithms, including value-based, policy-based, and actor-critic methods.

Advantages of Rust:
Efficient memory management prevents memory leaks that often trouble data scientists when using Python.
Enables thread-safe execution for parallel training.
Offers overall faster training speeds compared to Python-based frameworks.

# Package
crates.io: https://crates.io/crates/reinforcex
```
cargo add reinforcex
```

# Algorithms
DQN,
PPO

# API
Instantiate the agent.
```Rust
use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::models::FCQNetwork;
use reinforcex::memory::ReplayBuffer;
use std::sync::Arc;

let device = Device::cuda_if_available();
let vs = nn::VarStore::new(device);
let n_input_channels = 4;
let action_size = 2;
let n_hidden_layers = 2;
let n_hidden_channels = 128;

let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
let model = Box::new(FCQNetwork::new(
    vs,
    n_input_channels,
    action_size,
    n_hidden_layers,
    n_hidden_channels,
));

let gamma = 0.97;
let n_steps = 3;
let batchsize = 16;
let update_interval = 8;
let target_update_interval = 100;
let replay_buffer_capacity = 2000

let explorer = EpsilonGreedy::new(0.5, 0.1, 50000);
let transition_buffer = Arc::new(ReplayBuffer::new(replay_buffer_capacity, n_steps));

let mut agent = DQN::new(
    model,
    optimizer,
    transition_buffer,
    action_size as usize,
    batchsize,
    update_interval,
    target_update_interval,
    Box::new(explorer),
    gamma,
);
```

Methods of agent.
```Rust
fn act(&self, obs: &Tensor) -> Tensor;
fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;
fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64);
```

Pseudo code for training.
```Rust
for episode in 0..max_episode {
  for step in 0..max_step {
    action = agent.act_and_train(&mut self, obs: &Tensor, reward: f64)
    obs, reward = env.step(action)
  }
  agent.stop_episode_and_train(&mut self, obs: &Tensor, reward: f64)
}
```

This is a pseudo code for parallel learning.
ReplayBuffer is shared by all agents.
```Rust
use rayon::prelude::*;

let buffer = Arc::new(ReplayBuffer::new(1000, 1));

(0..n_threads).into_par_iter().for_each(|_| {
    let mut dqn = DQN::new(
        transition_buffer: Arc::clone(&buffer),
        ...(other params)...
    );

    for episode in 0..max_episode {
        for step in 0..max_step {
            action = agent.act_and_train(&mut self, obs: &Tensor, reward: f64)
            obs, reward = env.step(action)
        }
        agent.stop_episode_and_train(&mut self, obs: &Tensor, reward: f64)
    }
});
```

# Sample experiments
Run sample environment server in Docker.
```
docker-compose -f sample_env/docker-compose.yml up -d
```

```
cargo run --features cpu -- --env cartpole --algo dqn
```

<img width="597" alt="image" src="https://github.com/user-attachments/assets/b8c0606b-ec11-4b5a-b7fc-3070ad327d72" />

# Unit test
The experimental environment is built using OpenAI Gym. Since Gym is a Python framework, set up a Python environment and run the following pip command:
```
pip install gymnasium==0.26.3
```
We use Gym as the environment by calling Python from Rust.

```
cargo test
```

# FFI

This document describes the Foreign Function Interface (FFI) for interacting with ReinforceX agents from external languages such as C, C++, or C# (Unity).

---

## Overview

- All agents are managed internally and referenced via a `u64` ID.
- The API is **panic-safe**: all functions fail silently on error.
- All sizes use `u64` (ABI-safe across platforms).
- The caller is responsible for memory allocation of input/output buffers.

---

## Data Structures

### AgentConfig

Configuration used to create an agent.

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

### Fields

| Field | Description |
|------|------------|
| agent_type | 0 = DQN, otherwise PPO |
| obs_size | Size of observation vector |
| action_size | Size of action space |
| learning_rate | Optimizer learning rate |
| gamma | Discount factor |
| batch_size | Batch size (DQN) |
| buffer_size | Replay buffer size (DQN) |
| epsilon_start | Initial epsilon (DQN) |
| epsilon_end | Final epsilon (DQN) |
| epsilon_decay | Epsilon decay steps (DQN) |
| lambda | GAE lambda (PPO) |
| update_interval | PPO update interval |
| epoch | PPO training epochs |
| minibatch_size | PPO minibatch size |
| clip_eps | PPO clipping epsilon |

---

## Functions

### rx_agent_create

```c
uint64_t rx_agent_create(const AgentConfig* config);
```

#### Description
Creates a new agent and returns its unique ID.

#### Parameters
- `config`: Pointer to a valid AgentConfig struct

#### Returns
- `>= 1`: Agent ID  
- `0`: Failure (invalid config or internal error)

---

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

#### Description
Performs action selection and training step.

- For DQN: outputs a single scalar action  
- For PPO: outputs a vector action  

#### Parameters
- `id`: Agent ID  
- `obs`: Pointer to observation array  
- `obs_len`: Length of observation array  
- `reward`: Reward from previous step  
- `out`: Output buffer (pre-allocated)  
- `out_len`: Capacity of output buffer  

#### Output
- Writes action(s) into `out`  
- Writes up to `out_len` elements  

#### Notes
- If `out_len` is too small, output will be truncated  
- If pointers are null, function returns silently  

---

### rx_agent_stop_episode

```c
void rx_agent_stop_episode(
    uint64_t id,
    const float* obs,
    uint64_t obs_len,
    float reward
);
```

#### Description
Signals the end of an episode and performs a final training step.

#### Parameters
- `id`: Agent ID  
- `obs`: Final observation  
- `obs_len`: Length of observation  
- `reward`: Final reward  

---

### rx_agent_destroy

```c
void rx_agent_destroy(uint64_t id);
```

#### Description
Destroys the agent associated with the given ID.

#### Parameters
- `id`: Agent ID  

#### Notes
- Safe to call multiple times  
- If the agent does not exist, this is a no-op  

---

# License
MIT License (https://github.com/kakky-hacker/reinforcex/blob/master/LICENSE)
