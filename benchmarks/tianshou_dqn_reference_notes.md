# Supplementary Tianshou Double DQN protocol

This addition follows the original request to compare the native examples with
closely matched public implementations. It supplements, rather than modifies,
the frozen native/SB3 90-run campaign and three Tianshou discrete SAC controls.
No native core, installed native runtime, primary runner, or SAC runner changes
are permitted. All supplementary learning jobs run sequentially, at most one at
a time. The existing SAC jobs finish before adding the Box2D packages and running
either DQN smoke or production learning.

The separate `/private/tmp/reinforcex-tianshou-bench` environment retains
Tianshou 2.0.1, PyTorch 2.7.0, Gymnasium 1.3.0 and NumPy 2.4.6. Box2D 2.3.10 and
pygame 2.6.1 are added with `--no-deps`, without existing dependency upgrades,
after SAC finishes. Gymnasium's `box2d/__init__.py` eagerly imports CarRacing,
which requires pygame even for a non-rendering LunarLander run. The first Lunar
smoke caught this import requirement before any learning; its failed output
directory is retained and the successful smoke uses a distinct directory.
The complete DQN
environment is then recorded in `tianshou_dqn_requirements.txt`. CPU intra-op,
inter-op, BLAS and Numba thread limits are one; rendering is disabled.

The official [Tianshou v2.0.1 DQN implementation](https://github.com/thu-ml/tianshou/blob/v2.0.1/tianshou/algorithm/modelfree/dqn.py)
supports `is_double=True` and `huber_loss_delta=1.0`. Those public arguments are
set explicitly, alongside the native example's network and optimizer settings.
No private method is overridden and no upstream algorithm code is patched.
`DiscreteQLearningPolicy.set_eps_training()` and `add_exploration_noise()` supply
the official epsilon-greedy exploration; inference uses pure argmax.

| Setting | CartPole | LunarLander |
|---|---:|---:|
| Environment | CartPole-v1 | LunarLander-v3 |
| Training transitions per seed | 204800 | 1024000 |
| Training seeds | 42, 123, 2026 | 42, 123, 2026 |
| ReLU hidden layers | 64, 64 | 300, 300, 300 |
| Adam learning rate | 0.0005 | 0.0003 |
| Gamma | 0.99 | 0.99 |
| Batch / readiness threshold | 64 / 64 | 64 / 64 |
| Replay capacity | 50000 | 36000 |
| n-step horizon | 3 | 1 |
| Gradient update every env steps | 4 | 8 |
| Native target env interval | 250 | **50** |
| Tianshou target gradient interval | 63 | 6 |
| Nominal Tianshou target env interval | 252 | 48 |
| Relative target-period difference | +0.8% | -4.0% |
| Epsilon start/end/decay env steps | 1.0 / 0.05 / 10000 | 1.0 / 0.05 / 10000 |
| Training reward | CartPole shaping | raw reward |
| Final raw-return threshold | 475 | 200 |

The original frozen exported config is the sole source for native settings.
Native `hidden_layers` counts additional hidden layers after input-to-hidden,
so its 1 and 2 become two and three hidden layers respectively. Epsilon at the
one-based action step `t` is `start + (end-start)*min(t/decay_steps,1)`, including
the first action's small decrement. No separate fully-random warmup overrides
that schedule. Learning starts after strictly more than 64 collected
transitions, at the next configured update interval: step 68 for CartPole and
step 72 for LunarLander. This aligns the first update and total update counts
with the native delayed-transition loop for these settings (51,184 and 127,992
updates respectively), while replay sample availability still differs. This
choice was fixed from the native control flow and update counts before any
production Double DQN reference run.

Target intervals are rounded to the nearest positive number of gradient steps,
with exact half ties rounded upward. In particular Lunar's frozen interval is
**50**, not 500. Its 50/8=6.25 therefore maps to 6 rather than 63. This rule and
all settings are fixed before observing production results; the interval is
not subsequently tuned for a better score.

The unmodified official learner also copies targets on its first gradient
iteration. Native copies on its absolute environment-step clock. Target
computation/copy/optimizer ordering therefore differs beyond the nominal
period. Native additionally clips gradient norm to 10; official Tianshou DQN
has no public constructor option for this and remains unclipped. The method,
Huber loss, Torch version, architecture, environment, learning rate and replay
settings are closer to native than the SB3 vanilla-DQN control, but this is
still not an identical implementation. Replay sampling, n-step tail readiness,
RNG streams and initialization are documented remaining differences.

Training and evaluation use separate environments. The training seed is used
only on the initial reset, with natural resets afterward. Initial and ten
equally spaced progress evaluations each use ten greedy episodes with fixed
seeds 800000–800009. The final model, without selecting the best validation
checkpoint, runs 100 held-out episodes with seeds 900000–900099. Evaluation
preserves Python/NumPy/PyTorch RNG states and training mode. Only the final
held-out raw-return mean receives a threshold judgment. Each training and
evaluation episode's raw return, length, and flags are saved, and final partial
training episodes are recorded separately without synthetic terminal events.

CartPole training uses 0.01*raw reward except -1 on a true termination before
step 500. True termination at step 500 retains 0.01, matching the example.
LunarLander training and evaluation both use raw reward. Time-limit truncation
stops the n-step reward sequence while retaining bootstrap through the official
Tianshou return computation.

`tianshou_dqn_manifest.json` records all six jobs, outputs, budgets, environment
limits and frozen source/dependency/config hashes. Runs are stored separately
under `runs/tianshou_dqn/{cartpole_dqn,lunar_dqn}/seed_<seed>`. Each run saves the
resolved configuration, versions, metadata, raw episode JSONL, evaluation
summaries, first/checkpoint update diagnostics, final checkpoint and final
result. Main experiment scores and settings are not altered by this addition.

Before production, separate seed-7 1,024-transition smoke runs passed for both
environments. Independent artifact audits passed 407 CartPole checks and 205
LunarLander checks. Reloading the saved weights and taking argmax directly from
the Q network reproduced all five final held-out episode returns and lengths
in each environment. First-update steps were 68/72 and configured target
gradient periods were 63/6, as intended. These are harness checks, not evidence
that 1,024 steps is a sufficient learning budget.
