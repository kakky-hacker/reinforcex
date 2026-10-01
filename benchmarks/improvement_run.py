"""Development/confirmation CPU benchmark for the October core-improvement study.

All complete training episodes and every evaluation return are retained. Each
learner owns its environment/trajectory; sharing is through the native handles.
Evaluation never selects a checkpoint or interrupts an ongoing training episode.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import gymnasium as gym
import numpy as np
import reinforcex_ffi as rx
from improvement_configs import (BASELINE, as_dict, effective_configuration, study_request)


def source_contents():
    paths = [Path(__file__), ROOT / "benchmarks/improvement_configs.py",
             ROOT / "benchmarks/improvement_campaign.py", ROOT / "benchmarks/native_configs.py",
             ROOT / "examples/reinforcex_ffi.py"]
    return {str(path.relative_to(ROOT)): path.read_bytes() for path in paths}


# Capture before learning begins; never infer this provenance for older runs.
IMPORTED_SOURCE_CONTENTS = source_contents()
IMPORTED_SOURCE_HASHES = {path: hashlib.sha256(data).hexdigest()
                        for path, data in IMPORTED_SOURCE_CONTENTS.items()}


def load_checked_library(path, build_manifest):
    if path is None:
        raise ValueError("specify --library or REINFORCEX_LIB; benchmark fallback loading is disabled")
    path = Path(path).resolve(strict=True)
    expected_sha = build_manifest["library_sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("requested library differs from build manifest")
    # The general example loader can fall back to an installed/release library.
    # A benchmark must load exactly the binary requested by its saved job.
    lib = C.CDLL(str(path))
    rx.configure_ffi(lib)
    if Path(lib._name).resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("loaded library differs from requested library")
    return lib


def serial(value):
    if isinstance(value, C.Structure):
        return as_dict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(file, value):
    file.write(json.dumps(value, default=serial, allow_nan=False) + "\n")
    file.flush()


def transformed(reward, terminated, step, limit, mode):
    if mode == "cartpole":
        return -1.0 if terminated and step < limit else .01 * reward
    if mode == "scale0.1":
        return .1 * reward
    if mode == "hopper":
        return reward - 1.0
    if mode == "ant_shared":
        return .1 * (reward - 1.0)
    if mode == "raw":
        return reward
    raise ValueError(mode)


def summary(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(min(values)), "max": float(max(values)), "returns": values}


def stats(agent):
    values = agent.statistics()
    assert all(np.isfinite(v) for v in values.values()), values
    return values


def worker_seeds(specs, seed, share_rnd=False):
    """Keep each algorithm's initialization stable when other learners are added."""
    counts = {}
    result = []
    for spec in specs:
        algorithm = spec["algorithm"]
        ordinal = counts.get(algorithm, 0)
        counts[algorithm] = ordinal + 1
        policy_seed = seed + ordinal * 10000
        result.append({"algorithm": algorithm, "algorithm_worker": ordinal,
                       "policy_seed": policy_seed, "environment_seed": policy_seed,
                       "rnd_seed": seed + 1_000_000 + (0 if share_rnd else ordinal * 10000)})
    return result


def evaluate(agent, env_id, count, seed):
    env = gym.make(env_id)
    before = stats(agent)
    returns, lengths = [], []
    try:
        for ep in range(count):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.
            for length in range(1, env.spec.max_episode_steps + 1):
                action = rx.gym_action(agent, agent.act(obs), env.action_space)
                assert np.all(np.isfinite(action)), action
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(length)
        assert before == stats(agent), "evaluation changed learner statistics"
    finally:
        env.close()
    return {**summary(returns), "lengths": lengths, "seed_start": seed,
            "episodes": count, "deterministic": True, "reward": "raw"}


class Worker:
    def __init__(self, agent, env_id, seed, mode, output, index, algorithm, counter):
        self.agent, self.mode = agent, mode
        self.index, self.algorithm, self.counter = index, algorithm, counter
        self.env = gym.make(env_id)
        self.obs, _ = self.env.reset(seed=seed)
        self.previous_reward = self.total = self.learning_total = 0.
        self.length = self.steps = self.episodes = 0
        self.log = (output / f"train_worker{index}.jsonl").open("w")

    def train(self, count, final=False):
        for i in range(count):
            action = self.agent.act_and_train(self.obs, self.previous_reward)
            action = rx.gym_action(self.agent, action, self.env.action_space)
            assert np.all(np.isfinite(action)), action
            self.obs, reward, terminated, truncated, _ = self.env.step(action)
            assert np.all(np.isfinite(self.obs)) and np.isfinite(reward)
            self.length += 1
            self.steps += 1
            self.total += float(reward)
            self.previous_reward = transformed(float(reward), terminated, self.length,
                                               self.env.spec.max_episode_steps, self.mode)
            self.learning_total += self.previous_reward
            with self.counter["lock"]:
                self.counter["steps"] += 1
                aggregate_steps = self.counter["steps"]
            budget_cut = final and i == count - 1 and not (terminated or truncated)
            if terminated or truncated or budget_cut:
                self.agent.stop_episode(self.obs, self.previous_reward, terminated=bool(terminated))
                if not budget_cut:
                    self.episodes += 1
                append_json(self.log, {"worker": self.index, "algorithm": self.algorithm,
                    "episode": self.episodes + int(budget_cut), "steps": self.steps,
                    "aggregate_steps": aggregate_steps, "length": self.length,
                    "reward": self.total, "learning_reward": self.learning_total,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "budget_cut": budget_cut})
                if not (final and i == count - 1):
                    self.obs, _ = self.env.reset()
                self.previous_reward = self.total = self.learning_total = 0.
                self.length = 0
        return {"worker": self.index, "algorithm": self.algorithm, "steps": self.steps,
                "episodes": self.episodes, "statistics": stats(self.agent)}

    def close(self):
        self.log.close()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="Replicate a single-agent case; exact total budget stays fixed")
    parser.add_argument("--share-rnd", action="store_true")
    parser.add_argument("--serial-workers", action="store_true", help="Round-robin blocks; diagnostic only")
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--final-seed", type=int, help="Preregistered fresh confirmation seed block")
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=os.environ.get("REINFORCEX_LIB"))
    args = parser.parse_args()
    build_manifest = json.loads(args.build_manifest.read_text())
    case, specs = effective_configuration(args.case, args.overrides, args.workers, args.share_rnd)
    request = study_request(args.case, args.seed, args.steps, case, specs,
                            stage=args.stage, checkpoints=args.checkpoints,
                            eval_episodes=args.eval_episodes, validation_episodes=args.validation_episodes,
                            share_rnd=args.share_rnd, serial_workers=args.serial_workers,
                            final_seed=args.final_seed)
    validation_seed, final_seed = request["validation_seed"], request["final_seed"]
    args.output.mkdir(parents=True, exist_ok=False)
    for relative, content in IMPORTED_SOURCE_CONTENTS.items():
        snapshot = args.output / "runner_sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    begun = time.monotonic()
    agents, rnds, workers = [], [], []
    replay = None
    try:
        lib = load_checked_library(args.library, build_manifest)
        assert not rx.cuda_is_available(lib), "CPU benchmark requires a CPU execution device"
        actual_sha = hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
        assert actual_sha == build_manifest["library_sha256"], "loaded library differs from build manifest"
        n = len(specs)
        seeds = worker_seeds(specs, args.seed, args.share_rnd)
        assert args.steps > 0 and args.steps % (n * args.checkpoints) == 0
        assert args.eval_episodes >= 2
        if case["shared_replay"]:
            config = next(s["config"] for s in specs if s["algorithm"] != "ppo")
            replay = rx.create_replay_buffer(lib, config.replay_capacity, config.replay_n_steps)
        for index, spec in enumerate(specs):
            rnd = None
            if spec.get("rnd_config") is not None:
                if args.share_rnd and rnds:
                    rnd = rnds[0]
                else:
                    rx.manual_seed(lib, seeds[index]["rnd_seed"])
                    rnd_path = args.output / f"rnd_worker{index}"
                    rnd = rx.create_rnd(lib, spec["rnd_config"], str(rnd_path), None)
                    rnds.append(rnd)
            # Match each PPO policy's initialization between RND on/off conditions.
            rx.manual_seed(lib, seeds[index]["policy_seed"])
            path = str(args.output / f"worker{index}.ot")
            config = spec["config"]
            if spec["algorithm"] == "ppo":
                agent = rx.create_ppo(lib, config, path, None, rnd,
                                      spec.get("coefficient", .01), replay)
            elif spec["algorithm"] == "dqn":
                agent = rx.create_dqn(lib, config, path, None, replay)
            else:
                agent = rx.create_sac(lib, config, path, None, replay)
            if spec.get("learning_rate_schedule") is not None:
                from reinforcex_schedules import LinearLearningRateAgent
                source = ROOT / "examples/reinforcex_schedules.py"
                (args.output / "runner_sources/examples/reinforcex_schedules.py").write_bytes(source.read_bytes())
                native_agent = agent
                try:
                    agent = LinearLearningRateAgent(
                        native_agent, config.learning_rate, total_steps=args.steps // n,
                        final_fraction=spec["learning_rate_schedule"]["final_fraction"],
                        save_path=args.output / f"worker{index}.learning_rate.json")
                except BaseException:
                    native_agent.close()
                    raise
            if spec.get("normalization") is not None:
                from reinforcex_normalization import NormalizedAgent
                normalization_source = ROOT / "examples/reinforcex_normalization.py"
                snapshot = args.output / "runner_sources/examples/reinforcex_normalization.py"
                snapshot.write_bytes(normalization_source.read_bytes())
                normalization_path = args.output / f"worker{index}.normalization.json"
                native_agent = agent
                try:
                    agent = NormalizedAgent(
                        native_agent, config.agent.obs_size, config.agent.gamma,
                        save_path=normalization_path, **spec["normalization"],
                    )
                except BaseException:
                    native_agent.close()
                    raise
            agents.append(agent)
        env_spec = gym.spec(case["env_id"])
        metadata = {"backend": "reinforcex", "case": args.case, "env_id": case["env_id"],
                    "seed": args.seed, "requested_total_steps": args.steps,
                    "worker_count": n, "concurrent_workers": not args.serial_workers,
                    "worker_seeds": seeds,
                    "seed_limitations": "libtorch initialization and per-environment RNG are seeded; Rust thread_rng and concurrent execution order are not",
                    "cpu_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                    "share_rnd": args.share_rnd, "configuration": case,
                    "effective_agents": specs, "max_episode_steps": env_spec.max_episode_steps,
                    "reward_threshold": env_spec.reward_threshold, "device": "cpu",
                    "python": sys.version, "platform": platform.platform(),
                    "packages": {p: importlib.metadata.version(p) for p in ("gymnasium", "numpy", "mujoco", "Box2D")},
                    "library": str(lib._name),
                    "library_sha256": hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
                    "request": request,
                    "runner_source_sha256": IMPORTED_SOURCE_HASHES,
                    "normalization_source_sha256": (
                        hashlib.sha256((ROOT / "examples/reinforcex_normalization.py").read_bytes()).hexdigest()
                        if any(s.get("normalization") is not None for s in specs) else None),
                    "schedule_source_sha256": (
                        hashlib.sha256((ROOT / "examples/reinforcex_schedules.py").read_bytes()).hexdigest()
                        if any(s.get("learning_rate_schedule") is not None for s in specs) else None),
                    "config_file_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                    "override_file_sha256": None if args.overrides is None else hashlib.sha256(args.overrides.read_bytes()).hexdigest(),
                    "source_snapshot": str(args.build_manifest.resolve()),
                    "study_stage": args.stage,
                    "build_manifest": build_manifest,
                    "overrides": None if args.overrides is None else json.loads(args.overrides.read_text()),
                    "final_selection": "last checkpoint, no selection by evaluation",
                    "validation_seed": validation_seed, "validation_episodes": args.validation_episodes,
                    "checkpoints": args.checkpoints,
                    "final_test_seed": final_seed, "final_test_episodes": args.eval_episodes,
                    "max_rss_unit": "bytes" if sys.platform == "darwin" else "KiB"}
        write_json(args.output / "metadata.json", metadata)
        counter = {"lock": threading.Lock(), "steps": 0}
        workers = [Worker(agent, case["env_id"], seeds[index]["environment_seed"],
                          case["reward_mode"], args.output, index, specs[index]["algorithm"], counter)
                   for index, agent in enumerate(agents)]
        initialization_seconds = time.monotonic() - begun
        training_seconds = evaluation_seconds = 0.0
        with (args.output / "evaluations.jsonl").open("w") as eval_log:
            evaluation_started = time.monotonic()
            for index, agent in enumerate(agents):
                append_json(eval_log, {"worker": index, "algorithm": specs[index]["algorithm"],
                    "aggregate_steps": 0, "split": "validation",
                    **evaluate(agent, case["env_id"], args.validation_episodes, validation_seed)})
            evaluation_seconds += time.monotonic() - evaluation_started
            block = args.steps // n // args.checkpoints
            with ThreadPoolExecutor(max_workers=n) as pool:
                for checkpoint in range(1, args.checkpoints + 1):
                    last = checkpoint == args.checkpoints
                    training_started = time.monotonic()
                    if args.serial_workers:
                        progress = [w.train(block, final=last) for w in workers]
                    else:
                        futures = [pool.submit(w.train, block, last) for w in workers]
                        progress = [f.result() for f in futures]
                    training_seconds += time.monotonic() - training_started
                    elapsed = time.monotonic() - begun
                    status = {"checkpoint": checkpoint, "aggregate_steps": counter["steps"],
                              "seconds": elapsed, "workers": progress,
                              "training_seconds": training_seconds,
                              "evaluation_seconds": evaluation_seconds,
                              "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    write_json(args.output / "progress.json", status)
                    with (args.output / "progress_history.jsonl").open("a") as history:
                        append_json(history, status)
                    evaluation_started = time.monotonic()
                    for index, agent in enumerate(agents):
                        result = {"worker": index, "algorithm": specs[index]["algorithm"],
                                  "aggregate_steps": counter["steps"], "split": "validation",
                                  **evaluate(agent, case["env_id"], args.validation_episodes, validation_seed)}
                        append_json(eval_log, result)
                        print(json.dumps({"case": args.case, "seed": args.seed,
                            "worker": index, "steps": counter["steps"],
                            "eval": result["mean"], "seconds": round(elapsed, 1)}), flush=True)
                    evaluation_seconds += time.monotonic() - evaluation_started
            final = []
            for index, agent in enumerate(agents):
                agent.save()
                evaluation_started = time.monotonic()
                result = {"worker": index, "algorithm": specs[index]["algorithm"],
                          "aggregate_steps": counter["steps"], "split": "test" if args.stage == "confirmation" else "development",
                          **evaluate(agent, case["env_id"], args.eval_episodes, final_seed)}
                append_json(eval_log, result)
                evaluation_seconds += time.monotonic() - evaluation_started
                final.append(result)
            for rnd in rnds:
                rnd.save()
        write_json(args.output / "final.json", {"status": "complete", "backend": "reinforcex",
                   "case": args.case, "env_id": case["env_id"], "seed": args.seed,
                   "study_stage": args.stage, "actual_total_steps": counter["steps"], "seconds": time.monotonic() - begun,
                   "initialization_seconds": initialization_seconds,
                   "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                   "training_steps_per_second": counter["steps"] / training_seconds,
                   "workers": progress, "test": final, "threshold": env_spec.reward_threshold})
    except BaseException:
        write_json(args.output / "failure.json", {"traceback": traceback.format_exc(),
                                                  "seconds": time.monotonic() - begun})
        raise
    finally:
        for worker in workers:
            worker.close()
        for agent in reversed(agents):
            agent.close()
        for rnd in rnds:
            rnd.close()
        if replay is not None:
            replay.close()


if __name__ == "__main__":
    main()
