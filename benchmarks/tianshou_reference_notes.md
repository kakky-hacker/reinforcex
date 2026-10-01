# Supplementary CartPole Discrete SAC control

This additional control fills the discrete SAC gap in the SB3 comparison. It is
not a change to the primary 90-run benchmark protocol or native implementation.
The independent environment is `/private/tmp/reinforcex-tianshou-bench`.

Tianshou 2.0.1, PyTorch 2.7.0, and Gymnasium 1.3.0 were installed with the normal
resolver. `pip check` passed. `tianshou_requirements.txt` pins the complete
environment, including NumPy 2.4.6 and Numba 0.67.0. Box2D and MuJoCo extras are
not used. PyTorch 2.7 matches the native LibTorch version; all models explicitly
remain on CPU and numerical libraries use one thread.

The official [2.0.1 dependency declaration](https://github.com/thu-ml/tianshou/blob/v2.0.1/pyproject.toml)
permits Gymnasium >=0.28 without an upper bound, and Torch 2.x excluding 2.0.1
and 2.1.0. The official [DiscreteSAC implementation](https://github.com/thu-ml/tianshou/blob/v2.0.1/tianshou/algorithm/modelfree/discrete_sac.py)
supports n-step returns and twin critics. Its [AutoAlpha](https://github.com/thu-ml/tianshou/blob/v2.0.1/tianshou/algorithm/modelfree/sac.py)
accepts an arbitrary target entropy and initial log temperature. These APIs
allow the native configuration's low target entropy to be reproduced directly.

`run_tianshou_discrete.py` reads the frozen native CartPole SAC configuration.
The actor and each independently initialized critic have ReLU hidden layers
64,64 (native `hidden_layers=1` includes one additional layer). Actor learning
rate is 0.0003, critic rate 0.0005, alpha rate 0.0003, gamma 0.99, replay capacity
50,000, warmup 512, batch 64, n-step horizon 3, tau 0.005, and one gradient/target
update per collected step after warmup. Initial alpha is 0.05, target entropy
is `0.01 * ln(2) = 0.006931471805599453`. The actor returns logits, explicitly
setting `softmax_output=False` because DiscreteSACPolicy constructs its own
categorical distribution from logits.

The Gymnasium loop uses the official ReplayBuffer and `algorithm.update()` API;
Tianshou's algorithm is not patched or reimplemented. It collects exactly
204,800 environment transitions for each training seed 42, 123, 2026. Initial
and ten equally spaced progress evaluations use ten deterministic episodes
(seeds 800000–800009), followed by 100 final deterministic episodes
(900000–900099). Training reset uses its seed only initially; later resets
continue naturally. Evaluation has its own environment and preserves all
Python/NumPy/PyTorch training RNG states and policy mode.

Training reward is 0.01 times raw reward, except true termination before step
500 receives -1. True termination at step 500 therefore retains 0.01, matching
the native example's condition. All reporting uses raw reward. Every completed
training episode and every evaluation episode is saved. A final partial
training episode is retained separately; no synthetic terminal is inserted at
the training budget. The final weights are used regardless of intermediate
scores; only final held-out mean receives the fixed CartPole 475 threshold.

Remaining implementation differences must accompany results:

- Native discrete critics use Huber loss with delta 1; official Tianshou uses
  MSE. Native actor/critics clip gradient norm to 10; Tianshou DiscreteSAC exposes
  no public clipping constructor option and is used unchanged.
- Native replay exposes completed n-step sequences. Tianshou stores transitions
  immediately and shortens the n-step target at its current unfinished tail.
  Thus a nominal warmup of 512 reaches the first update a few steps differently.
- Native selects its next action before the current update and does not update
  in its episode-stop operation. This reference follows the conventional
  collect-transition/optimize loop and includes updates after terminal
  transitions. Replay sampling, initializers and RNG algorithms also differ.
- Both distinguish `terminated` from `truncated`, retaining bootstrap at a
  time limit. Tianshou's official
  [return computation](https://github.com/thu-ml/tianshou/blob/v2.0.1/tianshou/algorithm/algorithm_base.py)
  uses `~terminated` for value masking and `done` to stop the reward sequence.
- This supplementary comparison matches Torch 2.7 and Gymnasium 1.3 with native.
  The separate SB3 controls retain Torch 2.14 as documented in their report.

Output follows the SB3-style raw-episode schema: `train_episodes.jsonl`,
`eval_episodes.jsonl`, `evaluations.jsonl`, `config.json`, `metadata.json`,
`final.json`, and `final_model.pt`. `update_diagnostics.jsonl` adds the first
update and checkpoint losses/alpha; every update is checked for finite scalar
metrics. Source and checkpoint hashes, resolved parameters, actual steps,
timings and peak RSS in bytes are recorded.

## Harness verification

The separate seed-7 smoke collected exactly 1,024 transitions, performed 513
updates after the 512-transition warmup, and logged 41 complete episodes. Its
individual final evaluation returns exactly reproduced after reloading the
saved checkpoint. Episode lengths plus the partial tail accounted for every
transition. The true-termination reward boundary was checked at both 499 and
500 steps. These checks establish harness behavior, not convergence.

`audit_tianshou.py` independently validates configuration and source hashes,
versions, complete/partial transition accounting, the shaped rewards, raw
evaluation returns and means, all evaluation seeds, fixed final-only threshold,
checkpoint hash, and unchanged native/core and primary benchmark snapshots.
It passed 401 checks on the smoke. Six in-memory corruption checks (training
length, individual evaluation return, evaluation seed, final reported mean,
resolved capacity, checkpoint hash) were all rejected without modifying any
training artifact. `--allow-incomplete` explicitly reports pending jobs.

The three production jobs are saved in
`reports/oss_benchmarks/tianshou_manifest.json` and run sequentially via
`python3 benchmarks/run_matrix.py reports/oss_benchmarks/tianshou_manifest.json --jobs 1`.
The primary manifests and frozen runners are unchanged. After completion run
`python3 benchmarks/audit_tianshou.py` and, using the supplementary environment,
`python benchmarks/summarize_tianshou.py` to produce the supplementary comparison.
