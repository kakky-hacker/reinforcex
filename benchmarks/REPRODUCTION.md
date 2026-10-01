# Reproducing the CPU learning campaign

These instructions create a separate reproduction. They do not resume, delete,
or overwrite the recorded campaign. Run shell examples from the selected
checkout's root; the examples use POSIX shell syntax. The measured platform was
macOS arm64, Python 3.11.5, Rust 1.88, and CPU only. Linux/Windows setup and
bitwise-identical learning were not validated by this campaign.

## Restore the measured source

Use a new checkout or extracted source directory. The source archive must contain
`benchmarks/`, the exported `reports/oss_benchmarks/configs/`, all four manifests,
the source snapshots, dependency files and `source_worktree.patch`. A checkout of
the recorded base commit alone does **not** contain the complete experiment:
the patch contains the integration fixes, while the benchmark scripts and
metadata are additional artifacts.

The recorded base commit is
`757cca7df380113e7eefdc717b9b8ba21b6ff531`. If reconstructing from that commit,
first inspect and check the patch, then apply it only in the new checkout:

```sh
git apply --check /path/to/bundle/reports/oss_benchmarks/source_worktree.patch
git apply /path/to/bundle/reports/oss_benchmarks/source_worktree.patch
```

Copy the bundle's `benchmarks/` directory into this checkout as well. Also retain
the frozen exported `configs/` and `tianshou_dqn_protocol.json` at their canonical
`reports/oss_benchmarks/` locations in that checkout: supplementary source checks
include these inputs. Do not copy old `runs/`, status files or generated result
tables into a new experiment. Do not apply the patch a second time to an already
corrected tree. The authoritative
checks are `source_snapshot_before.json` for 52 source files and
`benchmark_code_snapshot_frozen.json` for the four frozen main runner/config
files. The older `benchmark_code_snapshot_start.json` is not the final freeze.
The supplementary manifests contain their own `frozen_sha256` maps.

The patch was independently applied to an archive of the base commit during the
documentation review: all 52 source hashes matched. This verifies source
reconstruction, not binary reproducibility. Compiler, platform and linked
libraries can produce different binaries and learning trajectories. A new run's
metadata must record its own library hash; never relabel it as the old binary.

## Install isolated dependencies

Use four virtual environments to reproduce the recorded dependency sets:

| Process | Requirements file | Tensor runtime |
|---|---|---|
| Python ctypes → native FFI | `native_requirements.txt` | external LibTorch 2.7.0; no Python torch |
| SB3 | `reference_requirements.txt` | Python PyTorch 2.14.0 / SB3 2.9.0 |
| Tianshou discrete SAC | `tianshou_requirements.txt` | Python PyTorch 2.7.0 / Tianshou 2.0.1 |
| Tianshou Double DQN | `tianshou_dqn_requirements.txt` | Python PyTorch 2.7.0 / Tianshou 2.0.1, plus Box2D/pygame |

For example, choose new local directories and use their Python executables
explicitly, without activating a shared environment:

```sh
python3.11 -m venv /path/to/new-envs/native
python3.11 -m venv /path/to/new-envs/sb3
python3.11 -m venv /path/to/new-envs/tianshou-sac
python3.11 -m venv /path/to/new-envs/tianshou-dqn
```

Install the matching requirement file with each environment's
`python -m pip install -r ...`, then run `python -m pip check` in each one.
Box2D may require a local C/C++ compiler and SWIG when a wheel is unavailable.
Install SWIG before the Box2D source build, and put its executable on that pip
process's `PATH`; listing `swig` in the same requirement file does not guarantee
that it is available while another package's wheel is being built. For example:

```sh
/path/to/new-envs/native/bin/python -m pip install swig==4.5.0
PATH="/path/to/new-envs/native/bin:$PATH" \
  /path/to/new-envs/native/bin/python -m pip install -r benchmarks/native_requirements.txt
```

Use an equivalent SWIG/compiler setup for the SB3 and Tianshou-DQN Box2D installs.
Gymnasium's Box2D import also needs pygame even when rendering is disabled;
the recorded requirement files supply the appropriate pygame package.
Keep the environments separate rather than installing all four files together.

These files are package-version snapshots, not hash-locked, platform-independent
wheel bundles. A different OS/architecture may need additional build packages
or platform-specific dependencies. Record any changes and do not silently drop
version pins. The runners explicitly select CPU; changing the tensor package or
Gymnasium version changes the comparison. See
[the SB3 environment notes](reference_env_notes.md) for why the main reference
uses PyTorch 2.14 despite native LibTorch 2.7.

## Build and isolate the native library

Use Rust 1.88 or newer and the committed Cargo.lock. Default CPU features enable
the `torch-sys` LibTorch download. Build in the reproduction checkout:

```sh
env -u LIBTORCH_USE_PYTORCH -u LIBTORCH_BYPASS_VERSION_CHECK \
  cargo build --locked --release -p reinforcex_ffi
```

If supplying LibTorch explicitly, set `LIBTORCH` for this build to the local
**2.7.0 CPU** distribution root containing `include/` and `lib/`. Remove a stale
`LIBTORCH` from the build environment if using the default download. Do not use
the SB3 Python package as the native build's LibTorch, and do not bypass version
checks. The compiler and LibTorch architecture must match the Python process
loading the resulting library.

The output is `target/release/libreinforcex.dylib` on macOS,
`target/release/libreinforcex.so` on Linux, or `target/release/reinforcex.dll` on
Windows. Discover the actual LibTorch directory from this build; the recorded
`target/debug/build/torch-sys-<hash>/...` path is a build-cache artifact and must
not be copied from the measurement machine. Inspect its `torch/version.h` and
confirm 2.7.0. If locating a downloaded distribution, this read-only search helps:

```sh
rg --files --no-ignore target | rg 'torch/version\.h$|libtorch_cpu\.(dylib|so)$'
```

Use `DYLD_LIBRARY_PATH` on macOS or `LD_LIBRARY_PATH` on Linux only for native
processes, pointing to that distribution's `lib/` directory. The Windows loader
requires the directory containing the LibTorch DLLs on `PATH`; the current
preparation helper's documented loader options cover macOS/Linux. Windows needs
separate manual adaptation and has not been tested here.

Never import Python torch into a process that has loaded the native library.
Separate virtual environments alone do not prevent inherited loader variables
from redirecting another process's shared libraries. Launch the controller and
reference tools from a shell with native loader variables removed; the prepared
native jobs receive their own loader environment.

## Prepare local manifests

Set the following to **your own** paths. `RX_BENCH_OUTPUT` must not exist.
The original bundle and its manifests are read-only inputs.

```sh
export RX_BENCH_BUNDLE=/path/to/recorded-bundle
export RX_BENCH_CHECKOUT=/path/to/new-reinforcex-checkout
export RX_BENCH_OUTPUT=/path/to/new-cpu-campaign
export RX_BENCH_NATIVE_PY=/path/to/new-envs/native/bin/python
export RX_BENCH_SB3_PY=/path/to/new-envs/sb3/bin/python
export RX_BENCH_TI_PY=/path/to/new-envs/tianshou-sac/bin/python
export RX_BENCH_TI_DQN_PY=/path/to/new-envs/tianshou-dqn/bin/python
export RX_BENCH_LIBRARY=/path/to/new-reinforcex-checkout/target/release/libreinforcex.dylib
export RX_BENCH_LIBTORCH_DIR=/path/to/local/libtorch/lib
export RX_BENCH_LOADER_VAR=DYLD_LIBRARY_PATH
```

Use `LD_LIBRARY_PATH` and the `.so` filename for Linux. Here `--libtorch-dir` is
the directory **containing the shared libraries**, usually `LIBTORCH/lib`, not
the `LIBTORCH` distribution root used by Cargo. Review a dry run first:

```sh
python3 "$RX_BENCH_BUNDLE/benchmarks/prepare_reproduction.py" \
  --source-base "$RX_BENCH_BUNDLE/reports/oss_benchmarks" \
  --checkout "$RX_BENCH_CHECKOUT" --output "$RX_BENCH_OUTPUT" \
  --native-python "$RX_BENCH_NATIVE_PY" --sb3-python "$RX_BENCH_SB3_PY" \
  --tianshou-python "$RX_BENCH_TI_PY" --tianshou-dqn-python "$RX_BENCH_TI_DQN_PY" \
  --native-library "$RX_BENCH_LIBRARY" --libtorch-dir "$RX_BENCH_LIBTORCH_DIR" \
  --loader-var "$RX_BENCH_LOADER_VAR"
```

Repeat the same command with `--write` to create the local manifests and copied
configuration/snapshot inputs. The helper checks the selected checkout against
the recorded frozen source, changes only location-related fields, records
provenance, and does not start learning. Check its emitted paths and job counts
before launching. Preserve seeds, budgets, reward transforms, worker counts,
RND sharing flags and exported hyperparameters. Replacing Python paths alone
while leaving the old `REINFORCEX_LIB`, loader path, `--config` or `--output` is
not a valid relocation.

The helper verifies source/configuration hashes and path existence. It does not
execute the selected Python, load the library, confirm installed package
versions, or prove the loader directory contains the right binaries. The
dependency checks and build checks above are separate prerequisites. Its
`launch_commands.json` records suggested controller invocations (four concurrent
jobs for native/SB3, one for each Tianshou manifest); the serial `--jobs 1`
commands below are an explicit lower-contention choice. Either schedule retains
the same per-run settings, but neither guarantees identical native RNG ordering.

The expected topology is 60 native runs / 102 worker models, 30 SB3 runs,
3 supplementary Tianshou discrete-SAC runs, and 6 supplementary Double-DQN
runs: **99 runs, 141 models, 119,193,600 training transitions**. Each condition
uses training seeds 42, 123 and 2026. The main comparison remains 90 runs;
supplementary results are not pooled into a main condition's seed sample.

## Start a new campaign deliberately

Only the following launcher commands start training. Run them sequentially,
checking each exit status. `--jobs 1` is a conservative CPU starting point;
native parallel conditions still create their configured learner workers.

```sh
env -u REINFORCEX_LIB -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  python3 "$RX_BENCH_CHECKOUT/benchmarks/run_matrix.py" "$RX_BENCH_OUTPUT/native_manifest.json" --jobs 1
env -u REINFORCEX_LIB -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  python3 "$RX_BENCH_CHECKOUT/benchmarks/run_matrix.py" "$RX_BENCH_OUTPUT/sb3_manifest.json" --jobs 1
env -u REINFORCEX_LIB -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  python3 "$RX_BENCH_CHECKOUT/benchmarks/run_matrix.py" "$RX_BENCH_OUTPUT/tianshou_manifest.json" --jobs 1
env -u REINFORCEX_LIB -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  python3 "$RX_BENCH_CHECKOUT/benchmarks/run_matrix.py" "$RX_BENCH_OUTPUT/tianshou_dqn_manifest.json" --jobs 1
```

Do not run two launchers against one manifest or output tree. The launcher checks
directory existence but does not lock the campaign, authenticate existing
`final.json`, or protect its status/log files against a second launcher. Existing
partial directories are refused; completed-looking directories are skipped.
Inspect failures and use a separately prepared output for a fresh rerun. Do not
delete a failed directory to make the launcher retry it under the same identity.

The optional command `python3 benchmarks/monitor_matrix.py "$RX_BENCH_OUTPUT"`
can run in a separate terminal from the start of the main matrix. It only records
native progress snapshots and stops when native/SB3 launcher statuses are present;
that stopping condition is not an all-99-run integrity or performance check.

`--steps` counts environment transitions, not episodes. Native parallel budgets
are totals divided across separate learner workers, not full budgets per worker
and not a vectorized single PPO policy. Episode plots compare different numbers
of transitions when policies have different episode lengths; use the step-based
evaluation plots for sample-efficiency comparisons. Final partial training
episodes are excluded from complete-episode reward averages. Evaluation always
uses unshaped raw reward, even when learning uses scaled/shaped reward.

Validation uses 10 episodes at seeds 800000–800009. Final tests use 100 episodes
at seeds 900000–900099. Do not select a checkpoint/worker or tune settings using
either report after observing the outcome. Keep every training seed. Native
replay sampling, PPO shuffling and parallel execution order are not fully seeded,
so rerunning the same numerical seed does not promise the same learning curve.

## Post-run analysis

Run analysis from the reproduction checkout. Use the Tianshou-DQN Python for
the plotting commands below because its pinned environment includes matplotlib;
the native/SB3 requirement files do not include the plotting dependency. These
commands do not change the trained weights. Set `MPLCONFIGDIR` to a writable
local temporary directory if needed; it is only a font/cache location.

The main audit accepts only the native/SB3 manifests. Do not append Tianshou
manifests to this auditor, which has implementation-specific schema checks:

```sh
"$RX_BENCH_TI_DQN_PY" benchmarks/audit_results.py \
  --root "$RX_BENCH_CHECKOUT" --base "$RX_BENCH_OUTPUT" \
  --output "$RX_BENCH_OUTPUT/results_audit.json"
"$RX_BENCH_TI_DQN_PY" benchmarks/render_oss_report.py \
  --runs-root "$RX_BENCH_OUTPUT/runs" --configs-dir "$RX_BENCH_OUTPUT/configs" \
  --output "$RX_BENCH_OUTPUT"
"$RX_BENCH_TI_DQN_PY" benchmarks/analyze_stability.py \
  --runs-root "$RX_BENCH_OUTPUT/runs" --configs-dir "$RX_BENCH_OUTPUT/configs" \
  --output "$RX_BENCH_OUTPUT"
"$RX_BENCH_TI_DQN_PY" benchmarks/analyze_evaluation_distribution.py \
  --base "$RX_BENCH_OUTPUT" --output "$RX_BENCH_OUTPUT"
"$RX_BENCH_TI_DQN_PY" benchmarks/render_overview.py \
  --summary "$RX_BENCH_OUTPUT/summary.json" \
  --tianshou-root "$RX_BENCH_OUTPUT/runs/tianshou/cartpole_sac" \
  --tianshou-dqn-root "$RX_BENCH_OUTPUT/runs/tianshou_dqn" \
  --output "$RX_BENCH_OUTPUT/figures"
python3 benchmarks/verify_checkpoint_reload.py \
  --base "$RX_BENCH_OUTPUT" --output "$RX_BENCH_OUTPUT/checkpoint_reload"
```

The distribution script covers all 99 runs; the main renderer, stability script
and native/SB3 reload verifier cover the main 90. The overview can display the
supplementary comparisons via explicit input paths; it does not replace their
dedicated integrity audits. Reload verification starts
serial inference-only child processes using each prepared manifest's Python
and library paths. It is not a weights-only training resume. The audit returns
exit code 2 for incomplete runs unless `--allow-incomplete` is supplied; missing
results must remain pending rather than being converted to a successful result.

Run the RND audit in the native environment and its renderer in the plotting
environment. The audit reconstructs initial RND tensors in temporary files using
the same library as the run; it does not update the trained predictor:

```sh
env REINFORCEX_LIB="$RX_BENCH_LIBRARY" \
  "$RX_BENCH_LOADER_VAR=$RX_BENCH_LIBTORCH_DIR" \
  "$RX_BENCH_NATIVE_PY" benchmarks/audit_rnd.py \
  --root "$RX_BENCH_OUTPUT" --manifest "$RX_BENCH_OUTPUT/native_manifest.json" \
  --output "$RX_BENCH_OUTPUT/rnd_posthoc_audit.json" --require-complete
"$RX_BENCH_TI_DQN_PY" benchmarks/render_rnd_report.py \
  --audit "$RX_BENCH_OUTPUT/rnd_posthoc_audit.json" \
  --output "$RX_BENCH_OUTPUT/rnd_visualization"
```

Successful integrity or checkpoint-reload audits do not mean the environment's
learning threshold was reached. The primary performance unit is a trained seed:
average the final worker means equally within a seed, then report the three
seed means, their sample SD and the 3-seed t interval. Do not use 300 evaluation
episodes as 300 independent training replications. Worker-level results and the
100-episode distributions remain visible; negative-return and threshold-above
episode counts are descriptive, not new pass/fail gates.

## Remaining path limitations of supplementary tools

The main commands above support a new external output directory. Some later
supplementary helpers were written for the recorded campaign and still resolve
inputs relative to the checkout's `reports/oss_benchmarks`:

| Tool | Relocation limit |
|---|---|
| `audit_tianshou.py`, `audit_tianshou_dqn.py` | `--manifest`/`--output` exist, but snapshot inputs still use fixed `BASE` |
| `summarize_tianshou.py`, `summarize_tianshou_dqn.py` | `--output` relocates generated files, not input runs |
| `verify_tianshou_checkpoints.py` | input runs use the canonical path |
| `verify_tianshou_dqn_checkpoints.py` | manifest input uses the canonical path; job outputs come from it |
| `campaign_status.py` | manifest base is fixed |
| `finalize_results.py` | fixed base; waits for all 99 runs in the four canonical manifests |

Therefore a custom-output main report alone must not be described as a complete
audit of all 99 runs. The updated finalizer covers all 99 runs only when the
canonical directory contains the intended campaign. The preparation helper
does **not** solve this remaining path limitation: it verifies canonical frozen
config/protocol inputs in the checkout and writes the new campaign elsewhere.
The custom output is usable for training all 99 runs, the main 90-run audit and
plots, and the 99-run distribution summary, but the full set of supplementary
audits/plots is not automatically portable yet.

To use the existing supplementary defaults on new results, separately stage
the **prepared manifest copies** and corresponding new result tree in a
dedicated analysis checkout's canonical report directory, with the matching
frozen config/snapshot inputs. Its canonical `runs/` and `summary.json` must refer
to the reproduction, not historical results. This is an additional manual
operation, not part of `--write`. Another option is a separately reviewed path
parameterization of these non-frozen analysis tools, retained as provenance.
Do not invoke their defaults on the source bundle while intending to analyze
new output, and do not rewrite old run metadata merely to make an archived
absolute path load.

Once the canonical directory refers to the new reproduction, the explicit
supplementary commands are:

```sh
"$RX_BENCH_TI_DQN_PY" benchmarks/audit_tianshou.py
"$RX_BENCH_TI_DQN_PY" benchmarks/audit_tianshou_dqn.py
"$RX_BENCH_TI_PY" benchmarks/verify_tianshou_checkpoints.py
"$RX_BENCH_TI_DQN_PY" benchmarks/verify_tianshou_dqn_checkpoints.py
"$RX_BENCH_TI_DQN_PY" benchmarks/summarize_tianshou.py
"$RX_BENCH_TI_DQN_PY" benchmarks/summarize_tianshou_dqn.py
"$RX_BENCH_TI_DQN_PY" benchmarks/render_overview.py
```

The two supplementary checkpoint commands perform inference only, so run them
serially if CPU contention matters. Keep native/SB3 results and the two Tianshou
controls separately identified in the final interpretation.

For a correctly staged canonical campaign, the updated finalizer runs the
complete audit and reporting sequence above, including both supplementary
controls, checkpoint verification, the strict RND audit, distribution/stability
summaries and plots:

```sh
/usr/bin/env -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  python3 benchmarks/finalize_results.py --wait
```

It checks launcher statuses and final artifacts for all four manifests and
waits without starting analysis until all 99 training runs are complete. It
does not start, resume, or modify training. Without `--wait`, pending training
returns exit code 2. Failed training or an invalid artifact returns exit code 1.
The report `finalization.json` can become `passed` only after exact coverage
checks: 132 native/SB3 models with ten matching reload episodes each and zero
training/save calls; 12 RND runs / 15 modules; and 141 models with all 14,100
final test episodes. The Tianshou checkpoint verifiers separately check all 900
episodes. Commands run serially; native LibTorch loader settings and interpreter
paths come from the manifests, while reference tools run without those loader
variables. A `passed` integrity result does not assert that every learning
threshold was reached. Original 93-run finalization records are historical and
do not establish the coverage of this updated procedure.
