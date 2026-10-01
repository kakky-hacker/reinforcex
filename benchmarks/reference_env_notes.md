# Stable-Baselines3 reference environment

Prepared on 2026-09-18 in `/private/tmp/reinforcex-reference-bench`. The existing
`/private/tmp/reinforcex-gym-smoke` and global Python installations were not changed.
The benchmark selects `device="cpu"`; numerical library thread limits, PyTorch
intra-op threads, and inter-op threads are all set to one.

| Package | Installed version |
|---|---|
| Python | 3.11.5, macOS arm64 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.14.0 |
| Gymnasium | 1.3.0 |
| MuJoCo | 3.13.0 |
| Box2D | 2.3.10 |
| NumPy | 2.4.6 |

`pip check` passed. All seven environment IDs reset successfully:
`CartPole-v1`, `LunarLander-v3`, `LunarLanderContinuous-v3`, `Ant-v5`,
`Hopper-v5`, `Walker2d-v5`, and `HalfCheetah-v5`. Their default limits are 500 steps
for CartPole and 1,000 for the others. Rendering is disabled. Both CUDA and MPS
reported unavailable in this runtime; neither is requested by the runner.

The [official SB3 2.9.0 package declaration](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/setup.py)
requires Python >=3.10, Gymnasium >=0.29.1,<2.0, NumPy >=1.20,<3.0, and
PyTorch >=2.8,<3.0. These pinned packages satisfy that contract.
`reference_requirements.txt` contains the complete environment lock, including
the packaging tools. Recreate in another isolated directory with:

```sh
python3 -m venv /private/tmp/reinforcex-reference-bench-copy
/private/tmp/reinforcex-reference-bench-copy/bin/python -m pip install -r benchmarks/reference_requirements.txt
```

Version selection was checked against the official older releases too.
[SB3 2.8.0](https://github.com/DLR-RM/stable-baselines3/blob/v2.8.0/setup.py)
and 2.7.x support PyTorch 2.7 and DQN n-step returns, but require
Gymnasium **<1.3.0**. The [official changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html)
records that Gymnasium 1.3 support and the PyTorch >=2.8 requirement both arrived
in 2.9.0. Keeping the same Gymnasium environment implementation as the native
runs avoids introducing a new task-version difference or forcing unsupported
dependencies. Therefore the comparison keeps SB3 2.9.0/PyTorch 2.14.0 and explicitly
reports the tensor-runtime difference from native LibTorch 2.7.0.

## Runner API

`run_sb3.py` uses official `DQN`, `PPO`, `SAC`, `DummyVecEnv`, `BaseCallback`,
`model.learn()`, `model.predict(deterministic=True)`, and `model.save()` APIs.
[PPO](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html),
[DQN](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/dqn.html), and
[SAC](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/sac.html) document
the exposed constructor arguments. SB3 2.9.0 DQN supports `n_steps=3`, as used by
the CartPole native configuration. Official SB3 SAC supports continuous actions;
it cannot serve as a same-algorithm control for the discrete CartPole SAC example.

```sh
/private/tmp/reinforcex-reference-bench/bin/python benchmarks/run_sb3.py \
  --config reports/oss_benchmarks/configs/cartpole_dqn.json \
  --seed 42 --steps 204800 \
  --output reports/oss_benchmarks/sb3/cartpole_dqn/seed_42
```

Alternatively specify `--env` and `--algo` directly. `--config` reads the JSON
export of `native_configs.as_dict(config(...))` without importing the native
library into the PyTorch process. Exactly one agent is accepted; RND configurations
and discrete SAC are rejected. Individual CLI hyperparameters or `--hyperparams`
JSON can override the translated constructor, with the resolved settings saved.
Existing result files are never overwritten.

The initial training reset uses `--seed`; subsequent episode resets continue
the environment RNG naturally. A separate evaluation environment runs ten
deterministic episodes initially and at each of ten equally spaced budget
thresholds, with fixed seeds 800000–800009. The final model is evaluated over
100 held-out episodes with seeds 900000–900099, and only that final result can
receive a pass/fail label via `--success-threshold`. Evaluation preserves the
Python/NumPy/PyTorch RNG state and restores the policy's training mode. The
progress callback observes the model before that collection's pending optimizer
step; final evaluation occurs after `learn()` has completed all updates.

As documented in [the official callback guide](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/callbacks.html),
the callback's step count is environment transitions. The runner has one
environment. SB3's requested total is a lower bound: PPO completes a rollout and
off-policy algorithms finish a collection interval. The runner records actual
steps and never inserts a false terminal or resets the active training episode
at the budget boundary. A partial last episode is recorded separately and is
excluded from complete-episode reward averages.

Output files:

- `train_episodes.jsonl`: every completed training episode's raw return, learning
  return, length, total steps, and separate terminated/truncated flags.
- `eval_episodes.jsonl`: every evaluation episode, its seed and raw return.
- `evaluations.jsonl`: initial/progress/final summaries and actual model steps.
- `config.json`: source configuration, resolved constructor, policy architecture,
  parameter count, and Gymnasium environment spec.
- `metadata.json`: exact packages, device, thread limits, source hashes,
  comparison caveats, and run completion/failure status.
- `final.json`: final held-out evaluation, actual steps, update count, timing,
  peak process RSS in bytes, complete episode count and excluded partial episode.
- `final_model.zip`: final policy, rather than the best validation checkpoint.

All plotted/comparable returns are raw environment rewards. Optional learning
transforms match the examples: CartPole uses 0.01*reward except -1 on true
termination before the time limit; `scale` is 0.1*reward; Hopper subtracts one;
Ant shared subtracts one then scales by 0.1. Evaluation applies no shaping.

## Mapping and remaining implementation differences

Native FC models create an input-to-hidden layer and then `hidden_layers`
additional layers. Consequently SB3 uses `[hidden_size] * (hidden_layers + 1)`
with ReLU. This was checked in the FCQNetwork, FCSoftmaxPolicy and FCGaussianPolicy
constructors and their FFI callers. A native setting of one therefore means
two hidden layers, not one.

PPO aligns learning rate, gamma, GAE lambda, rollout length, epochs, minibatch,
policy clipping, value clipping, value coefficient, entropy coefficient and
advantage normalization flag. Native PPO shares its actor/value trunk; official
SB3 has separate MLPs, so matching width/depth does not match total parameter
count. Advantage normalization scope, initialization, optimizer details, and
continuous policy variance/action handling remain implementation differences.

DQN aligns learning rate, replay capacity, minibatch, n-step returns, update and
target intervals, and epsilon schedule. Its warmup is `batch_size` in SB3, whose
update condition is strictly greater than that step. Native Double-DQN and
official SB3 vanilla DQN compute different targets. Replay storage and sampling
also differ.

Continuous SAC aligns actor/critic learning rate, replay capacity and warmup,
batch size, n-step returns, training interval, initial entropy temperature and
automatic target entropy. The selected native configurations all use equal
actor/critic learning rates. Native continuous SAC samples the untrained policy
during warmup; SB3 samples uniform random actions. The policy's variance
parameterization and gradient clipping remain different.

SAC target updates require an explicit translation. Native updates targets on
an environment-step cadence, while the SB3 reference takes one gradient step per
training interval. We set SB3's target gradient interval to one and use
`tau_sb3 = 1 - (1 - tau_native) ** (train_interval / target_interval)`.
This preserves nominal retention across environment steps, but does not make
the ordering of changing critics and target updates identical. The mapping is
saved per run. Examples: Lunar 8/8 retains tau 0.01; Hopper 1/1 retains 0.005;
HalfCheetah 2/1 maps 0.005 to 0.009975; Ant 4/1 maps it to 0.019850499375.

The SB3 reference uses PyTorch 2.14.0 whereas the native benchmark uses LibTorch
2.7.0 (verified from its bundled `torch/version.h`). This difference is recorded
in run metadata. Native core code is not changed by this runner.

## Harness verification

Short actual DQN/PPO/SAC learn/save/evaluate runs passed. A requested 129-step
budget produced 132 DQN steps at collection interval four, 192 PPO steps at
rollout 64, and 129 SAC steps at interval one. Config-driven smoke also passed,
including 3-step DQN replay and MuJoCo HalfCheetah SAC. For each, the sum of
completed episode lengths plus the saved partial episode equaled actual steps;
the evaluation JSONL counts and seeds matched every point, and non-final points
contained no pass/fail judgment. All ten reference case configurations were
successfully translated and inspected. These are harness checks, not convergence
or performance results.
