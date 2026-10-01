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
- replay buffers shared across training workers, with a separate on-policy
  rollout buffer for each PPO learner;
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
The committed lockfile requires Rust 1.88 or newer. `tch` 0.20 uses LibTorch
2.7.0; an older installed Python `torch` is not a compatible substitute.

```toml
[dependencies]
reinforcex = "0.0.5"
```

For CUDA experiments, build with the `cuda` feature and make sure your local
libtorch / CUDA runtime is visible to `tch`. On Windows, `try_load_cuda_dlls()`
loads `TORCH_CUDA_DLL` when the `cuda` feature is enabled and returns a `Result`
on missing/invalid configuration. The legacy `load_cuda_dlls()` remains a
best-effort wrapper. Successful loads are reused; failed loads can be retried.

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
  updates for continuous and discrete policies, and component checkpointing.
- RND ([Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)):
  Random Network Distillation with a fixed random target network, a trainable
  predictor, batched predictor updates, and predictor/target checkpointing.

Core building blocks:

- Models: `FCQNetwork`, `FCSoftmaxPolicy`, `FCSoftmaxPolicyWithValue`,
  `FCGaussianPolicy`, `FCGaussianPolicyWithValue`, `FCPpoPolicy`, `FCRNDModel`.
- Distributions: `SoftmaxDistribution`, `MultiSoftmaxDistribution`,
  `GaussianDistribution`.
- Memory: `ReplayBuffer` with n-step transitions, `OnPolicyBuffer`.
- Exploration and selection: `EpsilonGreedy`, `RewardBasedSelector`.
- Curiosity: `RND` computes intrinsic rewards and trains its predictor from
  PPO rollout batches.
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
fn supports_learning_rate_update(&self) -> bool;
fn set_learning_rate(&mut self, learning_rate: f64);
fn get_statistics(&self) -> Vec<(String, f64)>;
fn save(&self);
fn load(&mut self);
```

DQN and PPO support changing the existing optimizer's learning rate without
resetting Adam state. Check `supports_learning_rate_update()` first and pass a
finite, strictly positive rate. SAC does not support this setter. A schedule is
optional and supplied by the caller; existing constructor rates stay constant
unless the setter is used.

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

RND uses prediction error against a fixed, randomly initialized target network
as an intrinsic reward. Predictor training aims to reduce this error on the
observations it receives. Rollout-average error can rise as visited states
change, and shared RND also depends on worker update order. A decreasing error
alone does not establish improved exploration or higher external rewards.

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
    128, // maximum predictor minibatch size
    Some("models/lunar_rnd".to_string()),
    None,
);
```

`RND::calc_internal_reward` evaluates predictor error for a batch of
experiences without gradients. During each PPO update, PPO calculates these
rewards first and then updates the predictor using the same next-state batch.
`calc_internal_reward_and_update` keeps both operations under one lock when
RND is shared by multiple workers.
RND splits that batch into predictor minibatches of at most the configured
size. RND checkpoints contain `rnd_predictor.ot` and `rnd_target.ot` in the
configured directory.
The target is frozen. The RND module itself does not provide running observation
or intrinsic-reward normalization. Host-side observation preprocessing is
separate; it does not normalize the RND error automatically. Tune observation
units and the curiosity coefficient for each environment. Checkpoints save
network weights, not optimizer state.
The [RND comparison scope](reports/oss_benchmarks/RND_SCOPE.ja.md) distinguishes
module correctness checks from measured changes in external rewards.

# Python sample experiments

The runnable examples use Gymnasium directly and call ReinforceX exclusively
through its C FFI. Build the dynamic library and install the Python runtime
dependencies first:

```sh
cargo build -p reinforcex_ffi --release
python -m pip install numpy "gymnasium[classic-control,box2d,mujoco]"
```

Each former Rust experiment has a Python counterpart in [`examples`](examples):

| Environment | Algorithm | Command |
| --- | --- | --- |
| CartPole | DQN | `python examples/train_cartpole_dqn_ffi.py` |
| CartPole | PPO | `python examples/train_cartpole_ppo_ffi.py` |
| CartPole | discrete SAC | `python examples/train_cartpole_sac_ffi.py` |
| Ant | continuous PPO | `python examples/train_ant_ppo_ffi.py` |
| Ant | PPO + RND / PPO + SAC shared replay comparison | `python examples/train_ant_ppo_rnd_sac_shared_ffi.py` |
| Hopper | continuous SAC | `python examples/train_hopper_sac_ffi.py` |
| Walker2d | continuous PPO | `python examples/train_walker2d_ppo_ffi.py` |
| HalfCheetah | PPO + SAC shared replay | `python examples/train_half_cheetah_hybrid_ffi.py` |
| LunarLander | DQN | `python examples/train_lunar_lander_dqn_ffi.py` |
| LunarLander | PPO + RND | `python examples/train_lunar_lander_ppo_rnd_ffi.py` |
| LunarLanderContinuous | SAC | `python examples/train_lunar_lander_sac_ffi.py` |

The scripts accept `--episodes`, `--max-steps`, `--seed`, `--log-interval`, and
`--parallel`. CartPole PPO now defaults to a fixed step budget, described below;
its explicit episode mode requires a schedule horizon for the new model.
DQN and SAC workers share one FFI replay buffer. PPO workers own
separate policies, optimizers, and rollout buffers, with a separate RND module
per worker in the LunarLander RND example. The validation matrix also covers
several PPO learners attached to one RND handle. Intrinsic-reward calculation
and predictor updates on that shared RND are serialized; each PPO learner
continues to train its own policy from its own rollout.

Use `--save-path` and `--load-path` for checkpoints. In a parallel run,
`{agent_id}` is replaced with the worker index. Saving multiple workers to the
same resolved path is rejected:

```sh
python examples/train_cartpole_dqn_ffi.py \
  --parallel 4 \
  --save-path "models/cartpole_dqn_{agent_id}.ot" \
  --load-path "models/cartpole_dqn_{agent_id}.ot"
```

PPO+RND stores each predictor beside its PPO checkpoint with `.rnd` appended.
Set `REINFORCEX_LIB` when the dynamic library is not in `target/release` or the
platform library search path.
Gymnasium time limits are sent as `terminated=False` to preserve value
bootstrapping. Continuous model actions are mapped to each Gymnasium Box bound;
replay stores model-space actions. Keep those coordinate conventions and reward
semantics identical for agents sharing replay. A worker exception cooperatively
stops the example training loops and closes their environments.

The older episode-based examples use a full 100-episode training moving average
for their training thresholds. The Ant sample saves
the checkpoint with the best full-window mean and can reload it for deterministic
evaluation:

```sh
python examples/train_ant_ppo_ffi.py \
  --save-path "models/ant_ppo_best.ot" \
  --eval-episodes 20

python examples/train_ant_ppo_ffi.py \
  --eval-only \
  --load-path "models/ant_ppo_best.ot" \
  --eval-episodes 20
```

Pass `--preset baseline` to the Ant sample to reproduce its original
hyperparameters and clipped reward for controlled comparisons; `tuned` is the
default.

## CartPole PPO: fixed steps and final evaluation

The [CartPole PPO example](examples/train_cartpole_ppo_ffi.py) defaults to the
separate actor/value Tanh model, two hidden layers of width 64 per network,
and a linear learning rate from `5e-4` to `2.5e-5`. Adam epsilon is explicitly
`1e-8`, rollout length is 256, and each update uses six epochs and minibatches
of 64. Training retains the example's reward shaping; evaluation uses the
original Gymnasium reward.

```sh
python examples/train_cartpole_ppo_ffi.py \
  --steps-per-agent 204800 \
  --save-path "models/cartpole_ppo.ot" \
  --results-path "results/cartpole_ppo.json"

python examples/train_cartpole_ppo_ffi.py \
  --eval-only --load-path "models/cartpole_ppo.ot" \
  --eval-episodes 100 --eval-seed 900000
```

`--steps-per-agent` is the budget for **each** worker: `--parallel 2` at the
default budget collects 409,600 environment steps in total. Every worker trains
to its budget, sends its last transition, and saves its final model if requested.
There is no training-score early stop or best-checkpoint selection. All training
workers finish before deterministic raw-return evaluation begins (100 episodes
per model by default). A budget cut bootstraps and is logged separately from
complete episodes. Only the first training reset is seeded; evaluation resets
use `eval_seed + episode_index`. The default evaluation block is public and
reused, so an example run is not a new held-out confirmation experiment.

For an explicit episode budget, use, for example,
`--episodes 500 --schedule-steps 204800`. Episode mode is not equivalent to
204,800 actual steps: `--schedule-steps` only defines the LR horizon and is never
estimated from episode count. The two budget options are mutually exclusive.
`--max-steps` remains the episode limit (500 by default).

Saving `models/cartpole_ppo.ot` also writes
`models/cartpole_ppo.ot.learning_rate.json`: the suffix is **appended** to the
entire weight filename. Use `{agent_id}` in parallel save paths. With no
`--save-path`, neither weights nor schedule state are saved. `--results-path`
writes per-episode training records, every final evaluation return, per-worker
results and statistics, library SHA256, and source hashes before/after the run.
An existing results JSON is refused; choose a new result path for another run.

Training with `--load-path` is a **weights-only warm start**: optimizer, rollout
and schedule counter start fresh, and the old schedule sidecar is not restored.
For inference, `--eval-only` requires `--load-path` and performs no training,
saving or LR updates. Add `--legacy-model` to evaluate an old shared-ReLU PPO
checkpoint; the new separate model cannot load that architecture. Legacy-model
training uses a constant LR and is not the validated default. Rebuild the native
library for PPO V2 and runtime LR support; missing capabilities produce an error.

## Optional Python normalization and schedules

[`NormalizedAgent`](examples/reinforcex_normalization.py) wraps an already-created
agent with independently enabled observation and reward normalization. Its
float64 running moments update only on training calls, including the final
observation. It excludes the initial dummy reward, resets discounted returns
on both termination and truncation, and freezes statistics during `act()`.
Observations are centered and scaled; rewards are divided by the discounted
return standard deviation without subtracting their mean. Both clips default
to 10. Use the learner's gamma and keep each worker's moments separate.

For a normalized PPO exporting to raw-data shared replay, explicitly use
`preserve_replay_inputs=True` and a library supporting the separate replay-input
API below. Otherwise normalized and unnormalized transitions would mix in one
buffer. Here replay inputs are those received before this wrapper, which may
already include example reward shaping. This option does not establish a shared
normalization contract for multiple RND learners.

[`LinearLearningRateAgent`](examples/reinforcex_schedules.py) wraps DQN or PPO
and uses successful training action calls as its step counter. Supply the
initial rate, `total_steps`, and `final_fraction` (default `.05`). Episode-end
training applies the current rate without incrementing the counter; evaluation
does neither. The final stop at the step budget applies the exact LR floor.

Both wrappers take explicit JSON `save_path` / `load_path` arguments distinct
from native weight paths, delegate native saving, and validate saved options on
restore. Constructing with a wrapper `load_path` restores wrapper state only;
the supplied native agent must already have loaded its weights. Weight and
sidecar files are not a single atomic transaction. They support restoring a
policy and its preprocessing for inference, not exact training resume:
optimizer moments, replay/rollout buffers, RNG and live environment state are
not restored. The CartPole PPO CLI deliberately uses the warm-start behavior
described above when loading for training.

## Other environment examples

The Hopper SAC sample removes the environment's constant healthy bonus only
from the training target so the policy is rewarded for forward progress instead
of learning to stand still. Raw Gymnasium returns are still used for logging,
checkpoint selection, and evaluation:

```sh
python examples/train_hopper_sac_ffi.py \
  --save-path "artifacts/hopper_sac_best.ot" \
  --eval-episodes 20

python examples/train_hopper_sac_ffi.py \
  --eval-only \
  --load-path "artifacts/hopper_sac_best.ot" \
  --eval-episodes 20 \
  --render
```

SAC checkpoints use the supplied path as a base name and store actor, two
critics, and temperature in four component files.

The Walker2d PPO sample supports CUDA detection, deterministic evaluation, and
configurable rollout and training-reward settings. A saved model can be run
with a MuJoCo window using:

```sh
py examples/train_walker2d_ppo_ffi.py \
  --eval-only \
  --load-path "artifacts/walker2d_ppo_best.ot" \
  --eval-episodes 10 \
  --render
```

A historical tuning example, documented in commit `69b1cd0` on 2026-08-16,
uses three stages: 512-step rollouts, a forward-only reward stage, and
2048-step rollouts. The following settings illustrate the long-rollout stage
for training from saved weights. The fixed-budget CPU benchmark uses its own
frozen configuration and evaluates the final checkpoint; this historical
recipe does not establish its performance results or a best configuration.

```sh
py examples/train_walker2d_ppo_ffi.py \
  --episodes 1000 \
  --load-path "artifacts/walker2d_ppo_checkpoint.ot" \
  --save-path "artifacts/walker2d_ppo_candidate.ot" \
  --reward-mode survival \
  --learning-rate 0.00005 \
  --update-interval 2048 \
  --minibatch-size 128 \
  --entropy-coefficient 0.01 \
  --min-variance 0.02
```

The HalfCheetah hybrid sample runs PPO and SAC workers concurrently on CPU or CUDA.
Every PPO transition is copied into the same replay buffer used by SAC, so PPO
also acts as an off-policy data collector. PPO keeps its own on-policy rollout
data separately; shared transitions are stored on CPU without policy
distribution tensors to keep long CUDA runs memory bounded. The best PPO and
SAC candidates are selected by deterministic evaluation:

```sh
python examples/train_half_cheetah_hybrid_ffi.py \
  --ppo-episodes 300 \
  --sac-episodes 150 \
  --ppo-workers 2 \
  --sac-workers 1 \
  --eval-episodes 30

python examples/train_half_cheetah_hybrid_ffi.py \
  --eval-only \
  --eval-algorithm both \
  --load-ppo-path "artifacts/half_cheetah_ppo_best.ot" \
  --load-sac-path "artifacts/half_cheetah_sac_best.ot" \
  --eval-episodes 10 \
  --render
```

The Ant comparison sample runs three controlled conditions: PPO+RND with SAC
and shared replay, PPO without RND with SAC and shared replay, and standalone
SAC. Hybrid runs start PPO and SAC concurrently. All conditions use the same
initialization and evaluation seeds; shared replay contains transformed
extrinsic rewards only, so RND affects PPO exploration without changing SAC's
reward target. The default 400/400 hybrid and 800-episode standalone budgets
match the maximum number of collected transitions, while SAC update intervals
approximately match the optimizer-update budget:

```sh
python examples/train_ant_ppo_rnd_sac_shared_ffi.py

python examples/train_ant_ppo_rnd_sac_shared_ffi.py \
  --eval-only \
  --eval-condition rnd_shared \
  --eval-algorithm ppo \
  --load-path "artifacts/ant_rnd_shared_ppo_best.ot" \
  --eval-episodes 10 \
  --render
```

On Windows, build the CUDA FFI before training with:

```powershell
Get-Content .env | Invoke-Expression
cargo build -p reinforcex_ffi --release --no-default-features --features cuda
```

<img width="597" alt="CartPole training sample" src="https://github.com/user-attachments/assets/b8c0606b-ec11-4b5a-b7fc-3070ad327d72" />

# Unit test
Run all Rust unit tests from the workspace root:

```sh
cargo test --workspace
```

The core unit tests exercise agents, models, curiosity modules, probability
distributions, memory buffers, selectors, and the FFI wrapper. Gymnasium is
only required when running the Python sample experiments.

For DLL-level legacy ABI and process-safety regressions (Python standard library
only), build the FFI library and run:

```sh
python ffi/tests/test_ffi_regressions.py --library target/debug/reinforcex.dll
```

Use the platform's `.so`/`.dylib` name on Linux/macOS. For a Windows CUDA-feature
build, also pass `--expect-cuda-loader` and set `TORCH_CUDA_DLL` to an existing
CUDA DLL. This checks missing/invalid paths and retries in isolated processes;
it does not require an available GPU or measure GPU training performance.

For CPU learning, shared replay/RND, lifecycle and checkpoint validation:

```sh
cargo build -p reinforcex_ffi
python -m unittest discover -s ffi/tests -p test_example_helpers.py -v
OMP_NUM_THREADS=1 REINFORCEX_LIB=target/debug/libreinforcex.so \
  python ffi/tests/run_cpu_validation.py --suite full \
  --output reports/cpu_validation/results.json
```

Use `.dylib` on macOS or `reinforcex.dll` on Windows and configure the LibTorch
runtime loader path as needed. `--suite smoke --seeds 42` runs a smaller matrix.
The [CPU validation report](reports/cpu_validation/REPORT.ja.md) includes measured
results, reproduction commands and limits of the tests. A successful execution
check is not a claim that every environment has converged.

# CPU learning benchmarks

The fixed-protocol benchmark covers seven Gymnasium environments, 20 native
configurations and three training seeds per configuration. It runs entirely on
CPU with the core, FFI and examples frozen. Each final model is evaluated on
100 held-out episodes, without selecting the best checkpoint or worker.
Matched Stable-Baselines3 comparisons and supplementary Tianshou controls use
the same total environment-step budgets, summed across parallel workers.

The completed 2026-09-28 campaign contains 99 runs and 119,193,600 training
steps. CartPole SAC and the single-learner Lunar configurations passed the
registered thresholds in every tested seed; several other conditions did not.
The report also documents continuous PPO performance gaps and mixed RND results.

The [executive report](reports/oss_benchmarks/EXECUTIVE.ja.md) and
[learning curves](reports/oss_benchmarks/REPORT.ja.md), including episode versus
raw reward, disclose attained and missed thresholds and variation across seeds
and workers. See the [protocol](reports/oss_benchmarks/PROTOCOL.ja.md),
[comparison limitations](reports/oss_benchmarks/comparison_limitations.ja.md)
and [reproduction guide](benchmarks/REPRODUCTION.md) for conditions and methods.
Check the executive report, [handoff](reports/oss_benchmarks/HANDOFF.ja.md) and
[finalization status](reports/oss_benchmarks/finalization.json) for the current
completion state. Completing a run or passing an integrity audit does not
establish convergence in every environment.

## Subsequent core improvements

The separate 2026-10-01 improvement study preserves the completed campaign
above. With `core_v3_schedule`, the revised CartPole PPO configuration passed
all five previously unused training seeds: each final 100-episode raw mean was
500 (threshold 475). Reloading all five saved models in fresh CPU processes
reproduced all 500 returns and episode lengths exactly. See the
[current results](reports/core_improvements_20261001/RESULTS.ja.md) and
[checkpoint verification](reports/core_improvements_20261001/verification_batch3/README.ja.md).
These results validate the tested configuration and seeds, not every future
seed or the other three targets. CartPole DQN, Hopper SAC and HalfCheetah hybrid
PPO settings are still being investigated; their existing examples should not
be interpreted as having passed this new study.

The [core change notes](reports/core_improvements_20261001/CORE_CHANGES.ja.md)
separate implementation changes, model choices and training settings. Changes
already present at the study's start are recorded in
[`baseline/starting_worktree.patch`](reports/core_improvements_20261001/baseline/starting_worktree.patch);
the later core changes through `core_v3_schedule` are recorded in
[`changes_from_start.patch`](reports/core_improvements_20261001/builds/core_v3_schedule/changes_from_start.patch).
The old 99-run frozen reports remain evidence for their original source and
configuration, rather than results for the revised core or examples.

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
- DQN, PPO, SAC, shared replay buffers, and RND modules each have their own
  create/destroy lifecycle.
- PPO and SAC support `RX_ACTION_DISCRETE` and `RX_ACTION_CONTINUOUS`.
- Public functions catch Rust panics and return `RX_ERROR_PANIC` instead of
  unwinding across the ABI.
- Calls for one agent or RND handle are serialized internally. Different
  handles may be used from different host threads.
  Use a separate agent handle for each environment trajectory; serialized calls
  do not make interleaved observations from multiple environments one trajectory.
- The caller owns input and output buffer allocation.
- Observation buffers must contain exactly `obs_size` finite `float` values.
- Discrete agents write one action value. Continuous agents write `action_size`
  values.
- `const char *save_path` and `const char *load_path` are optional. Pass `NULL`
  or an empty string to disable that path. Non-null paths must be valid UTF-8.

Typed configs replaced the former catch-all `AgentConfig` and `rx_agent_create`
API. The subsequent PPO V2, replay-input and runtime LR additions preserve the
existing typed config layouts and legacy constructor/default entry points.
New functionality is opt-in; preserving the ABI and old model formats does not
promise identical training trajectories across core revisions.

Finite-value checks are not a transaction or rollback mechanism. A failure can
occur after an earlier critic/minibatch update or after consuming a rollout.
A panic can also poison an agent mutex; discard and recreate an unusable agent.
Input rejection before training and failure during training are distinct
guarantees; see the [change notes](reports/core_improvements_20261001/CORE_CHANGES.ja.md).

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

typedef struct RxPpoConfigV2 {
    RxPpoConfig base;
    uint32_t model;
    uint32_t activation;
    double initial_log_std;
    double adam_epsilon;
    double target_kl;
} RxPpoConfigV2;

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

typedef struct RxSacConfigV2 {
    RxSacConfig base;
    double discrete_target_entropy_ratio;
} RxSacConfigV2;
```

Replay, RND, and statistics structs:

```c
typedef struct RxReplayBufferConfig {
    uint64_t capacity;
    uint64_t n_steps;
} RxReplayBufferConfig;

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
  the Gaussian policy. They are ignored for discrete PPO. The policy retains
  the sampled, unclipped action for PPO likelihood ratios and clips only the
  action returned to the environment or exported to shared replay.
- Legacy Gaussian PPO variance is `softplus(raw_variance) + min_variance`, without an
  implicit upper bound. Rust callers can opt into a sigmoid-parameterized bound
  with `FCGaussianPolicyWithValue::with_max_variance(max_variance)`; the upper
  bound must be finite and strictly greater than `min_variance`. Loading a
  stochastic-policy checkpoint requires the same variance parameterization as
  training. Checkpoints trained with this branch's former implicit `0.1` cap
  need `.with_max_variance(0.1)` to reproduce that variance (for minima `< 0.1`).
  Deterministic evaluation still uses the unchanged action mean.
- `RxPpoConfigV2.model` selects `RX_PPO_MODEL_LEGACY` (0) or
  `RX_PPO_MODEL_SEPARATE` (1). The latter uses `FCPpoPolicy`: separate actor/value
  MLPs, orthogonal initialization, and Tanh (activation 0) or ReLU (1). Continuous
  separate policies have linear means and one learned, observation-independent
  `log_std` per action, with a `min_variance` floor. `initial_log_std` is the
  natural log of standard deviation, not variance. New-model checkpoints record
  model configuration and reject incompatible loads, including activation
  mismatches. Legacy and separate checkpoints are not interchangeable.
- V2 defaults select the separate Tanh model, `initial_log_std=0`,
  `adam_epsilon=1e-5`, and `target_kl=0` (disabled). The CartPole PPO example
  explicitly uses Adam epsilon `1e-8`. A positive target KL stops further
  minibatch updates when the approximate reverse KL exceeds `1.5 * target_kl`;
  it does not undo updates already applied. In the updated core,
  `value_clip_range=0` disables value clipping for either PPO model.
- `GaussianDistribution::sample()` and `most_probable()` operate in Gaussian
  policy space even for `new_bounded`. Direct distribution users should call
  `to_env_action()` before sending these actions to the environment; do not
  replace PPO's stored raw action with this mapped action.
- `RxRndConfig.update_interval` is retained as the ABI field name and configures
  the maximum RND predictor minibatch size.
- Continuous SAC uses a diagonal Gaussian policy. If `squash_action` is `1`, the
  action is tanh-squashed to `[-1, 1]`. Continuous SAC automatically tunes
  `alpha` toward target entropy `-action_size`. `min_variance` is ignored for
  discrete SAC.
- Discrete SAC automatically tunes `alpha` toward
  `log(action_size) * discrete_target_entropy_ratio`. Legacy `RxSacConfig` and
  the original SAC creation functions retain their binary layout and use the
  default ratio `0.98`. To customize it, use `RxSacConfigV2` with the `_v2`
  default/creation functions. Set `alpha` to `0` to
  disable automatic entropy tuning for either action-space type. The ratio is
  ignored for continuous SAC.
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

int32_t rx_ppo_config_default_v2(
    RxPpoConfigV2 *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_sac_config_default(
    RxSacConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_sac_config_default_v2(
    RxSacConfigV2 *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_replay_buffer_config_default(
    RxReplayBufferConfig *out_config,
    uint64_t capacity,
    uint64_t n_steps);

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

int32_t rx_ppo_create_v2(
    const RxPpoConfigV2 *config,
    uint64_t rnd_id,
    uint64_t replay_id,
    double curiosity_reward_coefficient,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_dqn_create_with_paths(
    const RxDqnConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_dqn_create_with_replay(
    const RxDqnConfig *config,
    uint64_t replay_id,
    uint64_t *out_id);

int32_t rx_dqn_create_with_replay_and_paths(
    const RxDqnConfig *config,
    uint64_t replay_id,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_paths(
    const RxPpoConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_rnd(
    const RxPpoConfig *config,
    uint64_t rnd_id,
    double curiosity_reward_coefficient,
    uint64_t *out_id);

int32_t rx_ppo_create_with_rnd_and_paths(
    const RxPpoConfig *config,
    uint64_t rnd_id,
    double curiosity_reward_coefficient,
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
| `rx_ppo_create_v2` | Selects PPO model/optimizer options, with optional RND, shared replay and paths. |
| `rx_sac_create` | Creates a SAC agent with its own replay buffer. |
| `rx_*_create_with_paths` | Creates an agent with optional save/load checkpoint paths. |
| `rx_ppo_create_with_rnd` | Creates a PPO agent with an existing RND curiosity module. |
| `rx_ppo_create_with_rnd_and_paths` | Same as above, with optional PPO save/load paths. |
| `rx_ppo_create_with_rnd_and_replay` | Creates a PPO agent that uses RND and also exports transitions to shared replay. |
| `rx_ppo_create_with_rnd_and_replay_and_paths` | Same as above, with optional PPO save/load paths. |
| `rx_ppo_create_with_replay` | Creates a PPO agent that exports transitions to an existing shared replay buffer. |
| `rx_ppo_create_with_replay_and_paths` | Same as above, with optional PPO save/load paths. |
| `rx_dqn_create_with_replay` | Creates a DQN agent that uses an existing shared replay buffer. |
| `rx_dqn_create_with_replay_and_paths` | Same as above, with optional save/load checkpoint paths. |
| `rx_sac_create_with_replay` | Creates a SAC agent that uses an existing shared replay buffer. |
| `rx_sac_create_with_replay_and_paths` | Same as above, with optional save/load checkpoint paths. |

For the unified PPO V2 constructor, `rnd_id=0` and `replay_id=0` mean absent;
both can also be supplied together. Paths may be `NULL`. Initialize with
`rx_ppo_config_default_v2` and access common C fields through `config.base`.
Python `RxPpoConfigV2` exposes those fields directly, and `create_ppo` dispatches
by config type. Old `rx_ppo_create*` entry points continue to use the legacy
shared model. New Python symbol bindings are optional, allowing an old library
to serve old APIs; requesting a missing new feature raises an explicit error.

SAC also provides `rx_sac_create_v2`, `rx_sac_create_with_paths_v2`,
`rx_sac_create_with_replay_v2`, and `rx_sac_create_with_replay_and_paths_v2`.
They take `const RxSacConfigV2 *` and otherwise have the same arguments as their
legacy counterparts. Initialize with `rx_sac_config_default_v2`, configure
common fields through `config.base`, and set `config.discrete_target_entropy_ratio`.
In the Python wrapper, `RxSacConfigV2` exposes common fields directly (for
example, `config.agent`); `create_sac` dispatches to the matching API version.

On success, create functions return `RX_OK` and write a non-zero handle to
`out_id`. On failure, `out_id` is set to zero. Shared DQN and SAC replay
creation checks that the replay buffer has the same `n_steps` as the agent
config and is large enough for its batches. SAC also checks
`replay_start_size`.
The first successful attachment binds the shared replay's observation size,
action size/type and gamma. Incompatible later attachments fail before training,
including simultaneous first attachments. Failed checkpoint loads do not bind
or poison this contract. Matching dimensions alone cannot verify observation,
reward or action meanings; callers must use compatible environments.

`rx_ppo_create_with_rnd*` requires the RND and PPO observation sizes to match.
The PPO agent retains the RND internally, calculates intrinsic rewards, updates
the RND predictor during PPO updates, and saves it together with the PPO agent.
The combined RND-and-replay constructors keep intrinsic rewards inside PPO;
only the caller-supplied extrinsic rewards are exported to shared replay.

## Agent action, training, statistics, and lifecycle APIs

```c
uint32_t rx_cuda_is_available(void);
int32_t rx_manual_seed(int64_t seed);

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

int32_t rx_agent_stop_episode_with_terminal(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float reward,
    uint32_t terminated);

int32_t rx_agent_set_learning_rate(uint64_t id, double learning_rate);

int32_t rx_agent_statistics_len(uint64_t id, uint64_t *out_len);

int64_t rx_agent_statistics(
    uint64_t id,
    RxStatistic *out_stats,
    uint64_t out_len);

int32_t rx_agent_save(uint64_t id);
int32_t rx_agent_load(uint64_t id);
int32_t rx_agent_destroy(uint64_t id);
```

`rx_cuda_is_available` reports libtorch CUDA availability. On Windows CUDA
builds, missing/invalid `TORCH_CUDA_DLL` returns `0` without terminating the host
process. Correcting the environment allows a later call to retry initialization.
Call `rx_manual_seed` before agent construction to make libtorch parameter
initialization reproducible in controlled comparisons.
It does not seed Rust `thread_rng` or determine parallel execution order, so
complete training trajectories are not guaranteed to be reproducible.

| Function | Purpose |
|---|---|
| `rx_agent_act` | Selects an action without adding a transition or updating the agent. |
| `rx_agent_act_and_train` | Selects an action, records the previous transition, and updates when due. |
| `rx_agent_stop_episode` | Records the terminal observation/reward and ends the current episode. |
| `rx_agent_stop_episode_with_terminal` | Ends the episode; `terminated=0` bootstraps through a time limit, `1` is a true terminal state. |
| `rx_agent_set_learning_rate` | Changes a DQN/PPO optimizer's LR while preserving Adam state, parameters and counters. |
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
For Gymnasium use the new function with the environment's `terminated` flag,
including `0` when `truncated` or a caller-imposed step budget ends the episode.
The legacy function keeps its original true-terminal behavior.

`rx_agent_set_learning_rate` accepts only finite, strictly positive values.
Zero, negative/non-finite rates and unsupported agents such as SAC return
`RX_ERROR_INVALID_ARGUMENT` without applying a change. Python exposes this as
`Agent.set_learning_rate(lr)`. It changes the existing optimizer, supplies no
schedule itself, and does not add LR or optimizer state to native checkpoints.

## Separate PPO learner and shared replay inputs

Normalized PPO can export unnormalized transitions for SAC without rewriting
its own on-policy data:

```c
int64_t rx_agent_act_and_train_with_replay_input(
    uint64_t id, const float *obs, uint64_t obs_len, float reward,
    const float *replay_obs, uint64_t replay_obs_len, float replay_reward,
    float *out, uint64_t out_len);

int32_t rx_agent_stop_episode_with_replay_input(
    uint64_t id, const float *obs, uint64_t obs_len, float reward,
    uint32_t terminated, const float *replay_obs, uint64_t replay_obs_len,
    float replay_reward);
```

These entry points support PPO only; DQN and SAC are rejected. Both observation
buffers describe the same current environment state, and both rewards belong
to the previous action. PPO rollout/GAE uses `obs` / `reward`; shared replay uses
`replay_obs` / `replay_reward` and the bounded action in policy coordinates,
before the host's Gym action mapping. Both streams retain the same episode and
terminal/truncation boundaries; shared replay applies its configured n-step
aggregation. Existing training calls pass the same inputs to both streams.

The FFI checks lengths, finite values, flags, output capacity and agent support
before invoking training. This cannot check the semantic meaning of host data
and does not make subsequent optimizer updates transactional. Replay reward
means the caller's separate extrinsic reward, potentially after reward shaping;
it does not necessarily mean the original environment reward. RND bonuses stay
inside PPO and are not exported.

Python provides `Agent.act_and_train_with_replay_input(observation, reward,
replay_observation, replay_reward)` and the corresponding
`stop_episode_with_replay_input(..., terminated=...)`. Use the latter for the
last transition with the actual terminal flag, including `False` at a time
limit. `NormalizedAgent(..., preserve_replay_inputs=True)` uses this path and
fails explicitly if the loaded library lacks it.

## Replay buffer APIs

```c
int32_t rx_replay_buffer_create(
    const RxReplayBufferConfig *config,
    uint64_t *out_id);

int32_t rx_replay_buffer_destroy(uint64_t id);
int32_t rx_replay_buffer_len(uint64_t id, uint64_t *out_len);
```

| Function | Purpose |
|---|---|
| `rx_replay_buffer_create` | Creates a replay buffer handle that can be shared by DQN, PPO, or SAC agents. |
| `rx_replay_buffer_destroy` | Removes the replay buffer handle from the registry. Existing agents keep their `Arc` reference alive. |
| `rx_replay_buffer_len` | Returns the number of sampleable transitions currently stored. |

Use shared replay buffers for multi-agent DQN or SAC training, or attach PPO as
an additional experience producer for an off-policy consumer. Experience
collection and replay insertion remain inside each agent.

## RND APIs

```c
int32_t rx_rnd_create(const RxRndConfig *config, uint64_t *out_id);

int32_t rx_rnd_create_with_paths(
    const RxRndConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_rnd_save(uint64_t id);
int32_t rx_rnd_load(uint64_t id);
int32_t rx_rnd_destroy(uint64_t id);
```

| Function | Purpose |
|---|---|
| `rx_rnd_create` | Creates an RND curiosity module. |
| `rx_rnd_create_with_paths` | Creates an RND module with optional save/load paths. |
| `rx_rnd_save` | Saves RND using the `save_path` supplied at creation. No-ops if no path was supplied. |
| `rx_rnd_load` | Loads RND using the `load_path` supplied at creation. No-ops if no path was supplied. |
| `rx_rnd_destroy` | Releases the RND handle. |

Attach an RND to PPO through `rx_ppo_create_with_rnd*`; the host does not
calculate intrinsic reward or train the RND directly. A shared RND handle can
be attached to multiple PPO agents. Its access is serialized internally.

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
