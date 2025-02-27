# About ReinforceX
ReinforceX (ReX) is a deep reinforcement learning framework built entirely from scratch in Rust.
We plan to implement various reinforcement learning algorithms, including value-based, policy-based, and actor-critic methods.

Advantages of Rust:
Efficient memory management prevents memory leaks that often trouble data scientists when using Python.
Enables thread-safe execution for parallel training.
Offers overall faster training speeds compared to Python-based frameworks.

# Algorithms
| Name | Status |
| --- | --- |
| REINFORCE | Done |
| DQN | Done |
| PPO | Not yet |
| SAC | Not yet |

# Sample experiments
The experimental environment is built using OpenAI Gym. Since Gym is a Python framework, set up a Python environment and run the following pip command:
```
pip install gymnasium==0.26.3
```
We use Gym as the environment by calling Python from Rust.

```
cargo run -- --env cartpole --algo dqn
```

<img width="597" alt="image" src="https://github.com/user-attachments/assets/b8c0606b-ec11-4b5a-b7fc-3070ad327d72" />

# Unit test
```
cargo test
```

# License
MIT License (https://github.com/kakky-hacker/reinforcex/blob/master/LICENSE)
