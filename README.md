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
| Name | Status |
| --- | --- |
| REINFORCE | Done |
| DQN | Done |
| PPO | Not yet |
| SAC | Not yet |

# API
Instantiate the agent.
```
use reinforcex::agents::{BaseAgent, DQN};
use reinforcex::explorers::EpsilonGreedy;
use reinforcex::models::FCQNetwork;

let device = Device::cuda_if_available();
let vs = nn::VarStore::new(device);
let n_input_channels = 4;
let action_size = 2;
let n_hidden_layers = 2;
let n_hidden_channels = Some(128);

let model = Box::new(FCQNetwork::new(
    &vs,
    n_input_channels,
    action_size,
    n_hidden_layers,
    n_hidden_channels,
));

let optimizer = nn::Adam::default().build(&vs, 3e-4).unwrap();
let explorer = EpsilonGreedy::new(0.5, 0.1, 50000);
let gamma = 0.97;
let n_steps = 3;
let batchsize = 16;
let update_interval = 8;
let target_update_interval = 100;

let mut agent = DQN::new(
    model,
    optimizer,
    action_size as usize,
    batchsize,
    2000,
    update_interval,
    target_update_interval,
    Box::new(explorer),
    gamma,
    n_steps,
);
```

Methods of agent.
```
fn act(&self, obs: &Tensor) -> Tensor;
fn act_and_train(&mut self, obs: &Tensor, reward: f64) -> Tensor;
fn stop_episode_and_train(&mut self, obs: &Tensor, reward: f64);
```

Pseudo code for training.
```
for episode in 0..max_episode {
  for step in 0..max_step {
    action = agent.act_and_train(&mut self, obs: &Tensor, reward: f64)
    obs, reward = env.step(action)
  }
  agent.stop_episode_and_train(&mut self, obs: &Tensor, reward: f64)
}
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
