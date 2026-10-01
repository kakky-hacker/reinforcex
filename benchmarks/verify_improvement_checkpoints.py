"""Reload each completed improvement run in a fresh CPU process and compare every saved final episode.

No cache, training, checkpoint selection, or model saving. Only verification
artifacts are written. The controller imports no numerical/native libraries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/core_improvements_20261001"
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
DEFAULT_LIBTORCH_DIR = ROOT / "target/debug/build/torch-sys-c1854a431246b133/out/libtorch/libtorch/lib"


def now():
    return datetime.now(timezone.utc).isoformat()


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON value")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def read_json(path):
    value = json.loads(Path(path).read_text())
    finite_tree(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints(paths):
    result = {}
    for path in sorted({Path(p).resolve(strict=True) for p in paths}):
        stat = path.stat()
        if stat.st_size <= 0:
            raise ValueError(f"empty input/checkpoint: {path}")
        result[str(path)] = {"sha256": sha256(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoint_paths(run, worker, spec, metadata):
    if spec["algorithm"] == "sac":
        paths = [run / f"worker{worker}_{part}.ot" for part in ("actor", "critic1", "critic2", "temperature")]
    else:
        paths = [run / f"worker{worker}.ot"]
    if spec.get("rnd_config") is not None:
        index = 0 if metadata.get("share_rnd") else worker
        paths += [run / f"rnd_worker{index}" / f"rnd_{part}.ot" for part in ("predictor", "target")]
    if spec.get("normalization") is not None:
        require(isinstance(spec["normalization"], dict), "normalization settings must be an object")
        paths.append(run / f"worker{worker}.normalization.json")
    if spec.get("learning_rate_schedule") is not None:
        paths.append(run / f"worker{worker}.learning_rate.json")
    return paths


def references(metadata, final):
    require(final.get("status") == "complete", "run is incomplete")
    require(metadata.get("backend") == final.get("backend") == "reinforcex", "unexpected backend")
    for field in ("case", "seed", "env_id", "study_stage"):
        require(metadata[field] == final[field], f"metadata/final {field} mismatch")
    require(metadata["study_stage"] in {"development", "confirmation"}, "unknown study stage")
    specs = metadata["effective_agents"]
    n = metadata["worker_count"]
    total = metadata["requested_total_steps"]
    require(type(n) is int and n > 0 and len(specs) == n, "effective worker count mismatch")
    require(type(total) is int and total > 0 and total % n == 0, "invalid worker budget")
    require(final["actual_total_steps"] == total, "final budget mismatch")
    require(len(final["workers"]) == len(final["test"]) == n, "missing/extra final workers")
    count, seed = metadata["final_test_episodes"], metadata["final_test_seed"]
    require(type(count) is int and count >= 2 and type(seed) is int and seed >= 0, "invalid final evaluation seeds/count")
    split = "test" if metadata["study_stage"] == "confirmation" else "development"
    result = []
    for index, spec in enumerate(specs):
        require(spec["algorithm"] in {"ppo", "dqn", "sac"}, "unsupported algorithm")
        worker, evaluation = final["workers"][index], final["test"][index]
        require((worker["worker"], worker["algorithm"], worker["steps"]) ==
                (index, spec["algorithm"], total // n), "worker budget/identity mismatch")
        require((evaluation["worker"], evaluation["algorithm"], evaluation["aggregate_steps"],
                 evaluation["split"], evaluation["seed_start"], evaluation["episodes"]) ==
                (index, spec["algorithm"], total, split, seed, count), "saved final evaluation protocol mismatch")
        require(evaluation.get("deterministic") is True and evaluation.get("reward") == "raw", "final evaluation must be deterministic/raw")
        returns, lengths = evaluation["returns"], evaluation["lengths"]
        require(len(returns) == len(lengths) == count, "incomplete saved final episodes")
        require(all(type(length) is int and 0 < length <= metadata["max_episode_steps"] for length in lengths), "invalid saved episode lengths")
        for name, expected in (("mean", statistics.mean(returns)), ("std", statistics.stdev(returns)),
                               ("min", min(returns)), ("max", max(returns))):
            require(math.isclose(evaluation[name], expected, rel_tol=1e-10, abs_tol=1e-8), f"saved evaluation {name} mismatch")
        result.append([{"seed": seed + episode, "return": reward, "length": length}
                       for episode, (reward, length) in enumerate(zip(returns, lengths))])
    return result


def reconstruct_config(rx, algorithm, saved):
    # Never call current library defaults or current example/configuration code.
    # Every constructor field must come from the actual training metadata.
    from improvement_configs import as_dict, fill
    if algorithm == "ppo":
        typ = rx.RxPpoConfigV2 if "model" in saved else rx.RxPpoConfig
    elif algorithm == "sac":
        typ = rx.RxSacConfigV2 if "discrete_target_entropy_ratio" in saved else rx.RxSacConfig
    elif algorithm == "dqn":
        typ = rx.RxDqnConfig
    elif algorithm == "rnd":
        typ = rx.RxRndConfig
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    config = typ()
    require(set(saved) == set(as_dict(config)), f"missing/unknown saved {algorithm} config fields")
    fill(config, saved)
    require(as_dict(config) == saved, f"saved {algorithm} config reconstruction mismatch")
    return config


def zero_update_statistics(agent, original_statistics):
    values = agent.statistics()
    finite_tree(values)
    update_key = "n_updates" if "n_updates" in values else "updates"
    require(update_key in values and values[update_key] == 0, "freshly loaded update counter must be zero")
    if "optimizer_steps" in original_statistics:
        require("optimizer_steps" in values, "optimizer counter exposed during training is now missing")
    for name in ("optimizer_steps", "optimizer_steps_last_update", "t", "replay_buffer_len"):
        if name in values:
            require(values[name] == 0, f"freshly loaded {name} counter must be zero")
    return values


def compare_episodes(observed, expected, atol, rtol):
    require(len(observed) == len(expected), "reloaded episode count mismatch")
    result = []
    for actual, saved in zip(observed, expected):
        finite = math.isfinite(actual["return"]) and math.isfinite(saved["return"])
        difference = abs(actual["return"] - saved["return"]) if finite else None
        passed = (finite and actual["seed"] == saved["seed"] and actual["length"] == saved["length"]
                  and math.isclose(actual["return"], saved["return"], abs_tol=atol, rel_tol=rtol))
        result.append({"seed": saved["seed"], "observed_seed": actual["seed"], "expected_return": saved["return"],
                       "observed_return": actual["return"] if finite else None, "absolute_difference": difference,
                       "expected_length": saved["length"], "observed_length": actual["length"], "passed": passed})
    return result


def evaluate_loaded(agent, rx, env, expected, original_statistics, atol, rtol):
    import numpy as np
    calls = {"training": 0, "save": 0}
    def forbidden_training(*args, **kwargs):
        calls["training"] += 1
        raise RuntimeError("training is forbidden in checkpoint verification")
    def forbidden_save(*args, **kwargs):
        calls["save"] += 1
        raise RuntimeError("saving is forbidden in checkpoint verification")
    agent.act_and_train = agent.stop_episode = forbidden_training
    agent.save = forbidden_save
    before = zero_update_statistics(agent, original_statistics)
    normalized_before = agent.state_dict() if hasattr(agent, "state_dict") else None
    observed = []
    for saved in expected:
        obs, _ = env.reset(seed=saved["seed"])
        total = 0.0
        for length in range(1, env.spec.max_episode_steps + 1):
            action = rx.gym_action(agent, agent.act(obs), env.action_space)
            require(np.isfinite(np.asarray(action)).all(), "nonfinite evaluation action")
            obs, reward, terminated, truncated, _ = env.step(action)
            require(np.isfinite(obs).all() and math.isfinite(float(reward)), "nonfinite evaluation observation/reward")
            total += float(reward)
            if terminated or truncated:
                break
        observed.append({"seed": saved["seed"], "return": total, "length": length})
    after = zero_update_statistics(agent, original_statistics)
    normalized_after = agent.state_dict() if hasattr(agent, "state_dict") else None
    normalization_unchanged = normalized_before == normalized_after
    comparisons = compare_episodes(observed, expected, atol, rtol)
    passed = (before == after and normalization_unchanged and calls == {"training": 0, "save": 0}
              and all(row["passed"] for row in comparisons))
    return {"status": "passed" if passed else "failed", "episodes": comparisons,
            "evaluated_episode_count": len(comparisons), "inference_environment_steps": sum(row["length"] for row in observed),
            "statistics_before": before, "statistics_after": after, "statistics_unchanged": before == after,
            "new_updates": 0, "optimizer_steps": after.get("optimizer_steps"),
            "optimizer_counter_available": "optimizer_steps" in after, "training_and_save_calls": calls,
            "normalization_state_unchanged": normalization_unchanged,
            "normalization_state_before": normalized_before, "normalization_state_after": normalized_after,
            "max_absolute_return_difference": None if any(row["absolute_difference"] is None for row in comparisons)
                else max(row["absolute_difference"] for row in comparisons)}


def make_request(run, atol, rtol):
    run = Path(run).resolve(strict=True)
    metadata, final = read_json(run / "metadata.json"), read_json(run / "final.json")
    reference = references(metadata, final)
    normalization_files = {run / f"worker{i}.normalization.json"
                           for i, spec in enumerate(metadata["effective_agents"]) if spec.get("normalization") is not None}
    require(set(run.glob("worker*.normalization.json")) == normalization_files,
            "normalization metadata/state file mismatch; refusing unnormalized evaluation")
    schedule_files = {run / f"worker{i}.learning_rate.json"
                      for i, spec in enumerate(metadata["effective_agents"]) if spec.get("learning_rate_schedule") is not None}
    require(set(run.glob("worker*.learning_rate.json")) == schedule_files,
            "learning rate schedule metadata/state file mismatch")
    library = Path(metadata["library"]).resolve(strict=True)
    require(Path(metadata["library"]).is_absolute(), "saved library path must be absolute")
    require(sha256(library) == metadata["library_sha256"], "library hash differs from training")
    manifest_path = Path(metadata["source_snapshot"]).resolve(strict=True)
    manifest = read_json(manifest_path)
    require(manifest == metadata["build_manifest"] and manifest["library_sha256"] == metadata["library_sha256"], "build manifest differs from training")
    launch_path = run.with_suffix(".launch.json")
    launch = read_json(launch_path)
    require((ROOT / launch["job"]["library"]).resolve() == library, "launch and metadata library paths differ")
    require((ROOT / launch["job"]["manifest"]).resolve() == manifest_path, "launch and metadata manifest paths differ")
    require((ROOT / launch["job"]["output"]).resolve() == run, "launch output differs from run directory")
    checkpoints = [path for i, spec in enumerate(metadata["effective_agents"]) for path in checkpoint_paths(run, i, spec, metadata)]
    sources = [Path(__file__), ROOT / "benchmarks/improvement_configs.py", ROOT / "examples/reinforcex_ffi.py"]
    if normalization_files:
        sources.append(ROOT / "examples/reinforcex_normalization.py")
    if schedule_files:
        sources.append(ROOT / "examples/reinforcex_schedules.py")
    return {"run_dir": str(run), "metadata": metadata, "final": final, "references": reference,
            "checkpoint_fingerprints": fingerprints(checkpoints),
            "reference_fingerprints": fingerprints([run / "metadata.json", run / "final.json", launch_path, manifest_path, library]),
            "verifier_fingerprints": fingerprints(sources), "python": launch["command"][0],
            "absolute_tolerance": atol, "relative_tolerance": rtol}


def run_worker(request):
    started = time.monotonic()
    for name in THREADS:
        os.environ[name] = "1"
    for key in ("checkpoint_fingerprints", "reference_fingerprints", "verifier_fingerprints"):
        require(fingerprints(request[key]) == request[key], f"{key} changed before verification")
    import ctypes as C
    import gymnasium as gym
    sys.path.insert(0, str(ROOT / "examples"))
    import reinforcex_ffi as rx
    metadata = request["metadata"]
    require(sys.version == metadata["python"], "Python runtime differs from training")
    packages = {name: importlib.metadata.version(name) for name in metadata["packages"]}
    require(packages == metadata["packages"], "runtime packages differ from training")
    library = Path(metadata["library"]).resolve(strict=True)
    require(sha256(library) == metadata["library_sha256"], "library changed before loading")
    lib = C.CDLL(str(library))
    rx.configure_ffi(lib)
    require(Path(lib._name).resolve() == library, "loaded a different library")
    require(not rx.cuda_is_available(lib), "checkpoint verifier requires CPU")
    rows = []
    for index, spec in enumerate(metadata["effective_agents"]):
        agent = rnd = env = None
        try:
            seeds = metadata["worker_seeds"][index]
            if spec.get("rnd_config") is not None:
                rx.manual_seed(lib, seeds["rnd_seed"])
                rnd_index = 0 if metadata.get("share_rnd") else index
                rnd = rx.create_rnd(lib, reconstruct_config(rx, "rnd", spec["rnd_config"]), None,
                                    str(Path(request["run_dir"]) / f"rnd_worker{rnd_index}"))
            rx.manual_seed(lib, seeds["policy_seed"])
            config = reconstruct_config(rx, spec["algorithm"], spec["config"])
            path = str(Path(request["run_dir"]) / f"worker{index}.ot")
            if spec["algorithm"] == "ppo":
                agent = rx.create_ppo(lib, config, None, path, rnd, spec.get("coefficient", 0.0))
            elif spec["algorithm"] == "dqn":
                agent = rx.create_dqn(lib, config, None, path)
            else:
                agent = rx.create_sac(lib, config, None, path)
            if spec.get("learning_rate_schedule") is not None:
                from reinforcex_schedules import LinearLearningRateAgent
                agent = LinearLearningRateAgent(
                    agent, config.learning_rate,
                    total_steps=metadata["requested_total_steps"] // metadata["worker_count"],
                    final_fraction=spec["learning_rate_schedule"]["final_fraction"],
                    load_path=Path(request["run_dir"]) / f"worker{index}.learning_rate.json")
            if spec.get("normalization") is not None:
                from reinforcex_normalization import NormalizedAgent
                agent = NormalizedAgent(agent, int(config.agent.obs_size), float(config.agent.gamma),
                                        **spec["normalization"], save_path=None,
                                        load_path=Path(request["run_dir"]) / f"worker{index}.normalization.json")
            env = gym.make(metadata["env_id"])
            require(env.spec.max_episode_steps == metadata["max_episode_steps"], "environment time limit changed")
            row = evaluate_loaded(agent, rx, env, request["references"][index],
                                  request["final"]["workers"][index]["statistics"],
                                  request["absolute_tolerance"], request["relative_tolerance"])
            rows.append({"worker": index, "algorithm": spec["algorithm"], "config_type": type(config).__name__, **row})
        finally:
            if env is not None:
                env.close()
            if agent is not None:
                agent.close()
            if rnd is not None:
                rnd.close()
    checkpoint_after = fingerprints(request["checkpoint_fingerprints"])
    reference_after = fingerprints(request["reference_fingerprints"])
    source_after = fingerprints(request["verifier_fingerprints"])
    immutable = checkpoint_after == request["checkpoint_fingerprints"]
    references_unchanged = reference_after == request["reference_fingerprints"]
    sources_unchanged = source_after == request["verifier_fingerprints"]
    passed = immutable and references_unchanged and sources_unchanged and all(row["status"] == "passed" for row in rows)
    return {"status": "passed" if passed else "failed", "run_dir": request["run_dir"], "workers": rows,
            "study_stage": metadata["study_stage"], "requested_total_steps": metadata["requested_total_steps"],
            "seed_start": metadata["final_test_seed"], "episodes_per_worker": metadata["final_test_episodes"],
            "verified_at_utc": now(), "seconds": time.monotonic() - started, "pid": os.getpid(),
            "python": sys.executable, "python_version": sys.version, "packages": packages,
            "device": "cpu", "library": str(library), "library_sha256": metadata["library_sha256"],
            "checkpoint_before": request["checkpoint_fingerprints"], "checkpoint_after": checkpoint_after,
            "checkpoint_unchanged": immutable, "reference_inputs_unchanged": references_unchanged,
            "verifier_sources_unchanged": sources_unchanged, "cached": False,
            "absolute_tolerance": request["absolute_tolerance"], "relative_tolerance": request["relative_tolerance"],
            "weights_in_memory_scope": "FFI exposes no tensor hashes; act-only calls and unchanged zero update counters checked"}


def worker_mode(request_path, output_path):
    try:
        result = run_worker(read_json(request_path))
    except Exception as error:
        result = {"status": "error", "error": repr(error), "traceback": traceback.format_exc(), "verified_at_utc": now(), "cached": False}
    write_json(output_path, result)
    return 0 if result["status"] == "passed" else 1


def verified_child_result(completed, raw_path, request):
    result = read_json(raw_path) if raw_path.exists() else {"status": "error", "error": "child produced no result"}
    if completed.returncode != 0:
        result.update(status="error", error=f"child exited with status {completed.returncode}")
    if result.get("status") not in {"passed", "failed", "error"}:
        result.update(status="error", error="child returned an invalid status")
    if result.get("status") == "passed":
        rows = result.get("workers", [])
        count = request["metadata"]["final_test_episodes"]
        complete = (len(rows) == request["metadata"]["worker_count"]
                    and all(row.get("worker") == i and row.get("status") == "passed"
                            and row.get("evaluated_episode_count") == count
                            and len(row.get("episodes", [])) == count
                            and all(ep.get("passed") is True for ep in row["episodes"])
                            and row.get("new_updates") == 0 and row.get("statistics_unchanged") is True
                            and row.get("normalization_state_unchanged") is True
                            for i, row in enumerate(rows))
                    and result.get("checkpoint_unchanged") is True
                    and result.get("reference_inputs_unchanged") is True
                    and result.get("verifier_sources_unchanged") is True)
        if not complete:
            result.update(status="error", error="child success report is incomplete")
    for key in ("checkpoint_fingerprints", "reference_fingerprints", "verifier_fingerprints"):
        if fingerprints(request[key]) != request[key]:
            result.update(status="failed", error=f"{key} changed during verification")
    result["cached"] = False
    return result


def controller(args):
    names = [run.resolve().name for run in args.runs]
    require(len(set(names)) == len(names), "run basenames must be unique within an invocation")
    output_dir = args.output_dir.resolve()
    for run in args.runs:
        require(output_dir != run.resolve() and run.resolve() not in output_dir.parents,
                "verification output must be outside training run directories")
    records = []
    for run in args.runs:
        output = output_dir / f"{run.resolve().name}.json"
        attempt = output_dir / ".attempts" / run.resolve().name / uuid.uuid4().hex
        try:
            request = make_request(run, args.atol, args.rtol)
            libtorch = args.libtorch_dir.resolve(strict=True)
            require(libtorch.is_dir(), "libtorch-dir must contain the native LibTorch shared libraries")
            environment = dict(os.environ)
            environment.update({name: "1" for name in THREADS})
            environment.update(REINFORCEX_LIB=request["metadata"]["library"],
                               DYLD_LIBRARY_PATH=str(libtorch), LD_LIBRARY_PATH=str(libtorch))
            request_path, raw_path = attempt / "request.json", attempt / "child.json"
            write_json(request_path, request)
            interpreter = str(args.python) if args.python else request["python"]
            command = [interpreter, str(Path(__file__).resolve()), "--worker-request", str(request_path), "--worker-output", str(raw_path)]
            print(json.dumps({"verifying": str(run), "episodes_per_worker": request["metadata"]["final_test_episodes"]}), flush=True)
            with (attempt / "subprocess.log").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            result = verified_child_result(completed, raw_path, request)
            result.update(subprocess_returncode=completed.returncode, invocation=command,
                          thread_environment={name: environment[name] for name in THREADS}, libtorch_dir=str(libtorch),
                          verifier_fingerprints=request["verifier_fingerprints"], reference_fingerprints=request["reference_fingerprints"])
        except Exception as error:
            result = {"status": "error", "error": repr(error), "traceback": traceback.format_exc(), "cached": False, "verified_at_utc": now()}
        result.update(run_dir=str(run.resolve()), attempt_directory=str(attempt))
        write_json(attempt / "result.json", result)
        write_json(output, result)
        records.append(result)
        print(json.dumps({"verified": str(run), "status": result["status"]}), flush=True)
    return 0 if all(row["status"] == "passed" for row in records) else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="completed run directories; all saved final episodes are required")
    parser.add_argument("--output-dir", type=Path, default=STUDY / "checkpoint_verification")
    parser.add_argument("--libtorch-dir", type=Path, default=DEFAULT_LIBTORCH_DIR, help="directory containing native LibTorch shared libraries")
    parser.add_argument("--python", type=Path, help="optional interpreter override; child still verifies the saved Python/package versions")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.worker_request) != bool(args.worker_output) or (not args.worker_request and not args.runs):
        parser.error("supply completed run directories, or both internal worker options")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error("timeout must be positive; tolerances must be finite and nonnegative")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(worker_mode(arguments.worker_request, arguments.worker_output)
                     if arguments.worker_request else controller(arguments))
