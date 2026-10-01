# CPU learning benchmarks

This directory contains two separate studies: the original frozen-example
benchmark and the subsequent core-improvement study. For the current changes,
read the [improvement results](../reports/core_improvements_20261001/RESULTS.ja.md),
[core change explanations](../reports/core_improvements_20261001/CORE_CHANGES.ja.md),
and [improvement protocol](../reports/core_improvements_20261001/PROTOCOL.ja.md).
The improvement study changes core code and selected example defaults; its
development trials and held-out confirmations remain separate from the original
99-run matrix below. DQN investigation in the improvement study was stopped at
the user's request with its performance gate unconfirmed; all results are retained.

The original runners test the frozen C FFI and examples after the integration fixes.
That historical learning campaign did not modify the core algorithms. It preserves every
training episode, raw evaluation returns, final weights, effective configurations,
dependency versions, and failures.

See the [predeclared protocol](../reports/oss_benchmarks/PROTOCOL.ja.md) and
[published benchmark sources](../reports/oss_benchmarks/reference_sources.ja.md).
The previous short integration tests are not convergence benchmarks.

For a new machine or a full rerun, follow the
[reproduction guide](REPRODUCTION.md). It covers the recorded source patch,
isolated runtimes, local manifest preparation, fresh output directories, and
the analysis commands. The checked-in manifests are historical evidence:
**do not launch them unchanged or rewrite them in place**.

## Environments and runtimes

The measurements use Python 3.11, Gymnasium 1.3.0, MuJoCo 3.13.0, and Box2D
2.3.10. Install `native_requirements.txt` and `reference_requirements.txt` in
**separate virtual environments**. The native library uses LibTorch 2.7.0;
the SB3 reference uses its own PyTorch 2.14.0. Never import these two tensor
runtimes in one Python process.

Build the native library using Rust 1.88 or newer:

```sh
cargo build --locked --release -p reinforcex_ffi
```

Set `REINFORCEX_LIB` to the built shared library. Its default filename is
`target/release/libreinforcex.dylib` on macOS, `libreinforcex.so` on Linux, or
`reinforcex.dll` on Windows. Set the platform library-loader path to the
LibTorch 2.7.0 library directory if it is not already discoverable.
Keep native loader variables local to native commands; a global
`DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` can also affect a separate SB3 process.
The requirement files record the measured macOS arm64 environments; they are
not cross-platform wheel locks. The reproduction guide lists the compiler/SWIG
prerequisites and the separate Tianshou environments.

## Run an individual experiment

From the repository root, with the native Python environment active:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/run_native.py --case lunar_rnd --seed 42 \
  --steps 1024000 --eval-episodes 100 --output /tmp/my-native-run
```

The output directory must not exist. `native_configs.py` lists all cases and
captures the original benchmark configurations; those are no longer a promise
that every current example has the same defaults. Improvement runs record their
effective configurations and overrides separately. Additional experiments can use
`--workers 2`, and RND cases can use `--share-rnd`. `--steps` is the **total**
budget across all workers, not the budget of each worker. Workers have separate
policies and optimizers; PPO workers are not a vectorized single PPO learner.

With the reference Python environment active and no native loader variables:

```sh
python benchmarks/run_sb3.py \
  --config reports/oss_benchmarks/configs/lunar_ppo.json \
  --seed 42 --steps 1024000 --output /tmp/my-sb3-run
```

The config adapter matches environment, reward preprocessing, architecture
width/depth, and available training settings. Metadata records the differences
that cannot be removed without changing the algorithms. In particular,
ReinforceX `hidden_layers` counts layers **after** the first hidden layer;
the corresponding SB3 depth is `hidden_layers + 1`.

## Full recorded matrix: 99 runs, 141 final models

All four manifests under `reports/oss_benchmarks` preserve commands used on the
measurement machine. Their Python and dynamic library paths are machine-specific.
Use `prepare_reproduction.py` to create local copies in a **new** output directory;
it defaults to a dry run and never starts training. See the
[complete preparation procedure](REPRODUCTION.md#prepare-local-manifests).

| Manifest | Conditions × training seeds | Runs | Final worker models |
|---|---|---:|---:|
| `native_manifest.json` | 15 primary + 5 parallel controls, each × 3 | 60 | 102 |
| `sb3_manifest.json` | 10 controls × 3 | 30 | 30 |
| `tianshou_manifest.json` | CartPole discrete SAC × 3 | 3 | 3 |
| `tianshou_dqn_manifest.json` | CartPole/LunarLander Double DQN × 3 | 6 | 6 |

The main comparison is the original 90 native/SB3 runs. The nine Tianshou runs
are separately identified supplementary controls, not extra seeds for a main
condition. Training seeds are 42, 123 and 2026. Per-run budgets are 204,800
transitions for CartPole, 1,024,000 for LunarLander, and 2,048,000 for MuJoCo.
Parallel native workers divide the total budget; their episode counts can differ.

The launcher skips a run whenever `final.json` exists; this is a scheduling
shortcut, **not** an integrity audit or proof of learning success. It refuses to
restart existing partial output directories. A second launcher targeting the
same manifest can race with the first and overwrite launcher logs/status files.
Run one launcher per manifest and keep every rerun in a separate output tree.
Native checkpoints contain weights,
not the complete optimizer/replay/RNG state. Use a fresh directory for a full
rerun; do not treat a weights-only reload as continuation of the same benchmark.

Evaluation uses separate environments, deterministic actions, and unmodified
environment rewards. Progress evaluation seeds and final test seeds are
disjoint. The final policy is evaluated, and the best intermediate checkpoint
or best worker is never selected. Native libtorch initialization and environment
RNG are seeded, but native replay sampling and thread scheduling are not fully
deterministic. Report all learning seeds, not only a successful run.

Concurrent runs share the host. Elapsed time and peak RSS are diagnostics, not a
controlled cross-library speed benchmark. `monitor_matrix.py` optionally records
native statistics snapshots while the matrices run; it does not alter learners.

## Evaluate and report

Use the [post-run commands](REPRODUCTION.md#post-run-analysis) with your prepared
output directory. Audit before interpreting performance. The main report uses
the last model, 100 final evaluation episodes per worker, equal worker weights
within a training seed, and then the three training seeds. Episode-level SD and
training-seed SD are different quantities. Learning curves use raw reward;
training-only reward transforms remain in the logs as separate fields.

For the canonical `reports/oss_benchmarks` directory, the updated
`python3 benchmarks/finalize_results.py --wait` waits for **all 99 runs** across
the four manifests before running audits, inference-only checkpoint checks,
distribution/stability summaries and plots serially. It never starts training.
Without `--wait`, incomplete training returns exit code 2. A final `passed`
requires all 132 native/SB3 models to pass checkpoint checks, all 12 RND runs /
15 modules to pass the strict audit, and all 141 models / 14,100 final evaluation
episodes to be present. This verifies completeness and integrity; reward
threshold attainment is reported separately.

The finalizer and some supplementary tools retain fixed input paths. For a new
output directory, follow the explicit commands and path limitations in the
reproduction guide; do not run canonical defaults on historical results.
