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
| Name | Status | Parallel |
| --- | --- | --- |
| DQN | Done | Done |
| PPO | Done | WIP |
| SAC | WIP | WIP |

# API
Instantiate the agent.
```Rust
use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::models::FCQNetwork;
use reinforcex::memory::TransitionBuffer;
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
let transition_buffer = Arc::new(TransitionBuffer::new(replay_buffer_capacity, n_steps));

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
TransitionBuffer is shared by all agents.
```Rust
use rayon::prelude::*;

let buffer = Arc::new(TransitionBuffer::new(1000, 1));

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
The experimental environment is built using OpenAI Gym. Since Gym is a Python framework, set up a Python environment and run the following pip command:
```
pip install gymnasium==0.26.3
```
We use Gym as the environment by calling Python from Rust.

```
cargo run --features dev -- --env cartpole --algo dqn
```

<img width="597" alt="image" src="https://github.com/user-attachments/assets/b8c0606b-ec11-4b5a-b7fc-3070ad327d72" />

# Unit test
```
cargo test --features dev
```

# License
MIT License (https://github.com/kakky-hacker/reinforcex/blob/master/LICENSE)
