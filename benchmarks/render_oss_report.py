#!/usr/bin/env python3
"""Render fixed-final-checkpoint CPU benchmark evidence without selecting winners.

Example: python3 benchmarks/render_oss_report.py
Dependencies for figures: numpy and matplotlib. Aggregation itself uses the
standard library. This script reads run logs only; it never trains a model.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/oss_benchmarks"
THRESHOLDS = {"CartPole-v1": 475.0, "LunarLander-v3": 200.0,
              "LunarLanderContinuous-v3": 200.0, "Ant-v5": 6000.0,
              "Hopper-v5": 3800.0, "Walker2d-v5": None,
              "HalfCheetah-v5": 4800.0}
PRIMARY_CASES = ("cartpole_dqn", "cartpole_ppo", "cartpole_sac", "lunar_dqn",
                 "lunar_ppo", "lunar_rnd", "lunar_sac", "ant_ppo", "hopper_sac",
                 "walker_ppo", "halfcheetah_hybrid", "halfcheetah_sac",
                 "ant_shared", "ant_rnd_shared", "ant_sac")


def clean_numbers(value: Any, location: str, errors: list[str]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location}: nonfinite number")
        return None
    if isinstance(value, dict):
        return {key: clean_numbers(item, f"{location}.{key}", errors) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_numbers(item, f"{location}[{index}]", errors) for index, item in enumerate(value)]
    return value


def read_json(path: Path, errors: list[str]) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return clean_numbers(value, str(path), errors)
    except (ValueError, OSError) as error:
        errors.append(f"{path}: {error}")
        return {}


def read_jsonl(path: Path, errors: list[str]):
    if not path.exists():
        return
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                yield clean_numbers(value, f"{path}:{number}", errors)
            except ValueError as error:
                errors.append(f"{path}:{number}: {error}")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def seed_summary(values: list[float]) -> dict:
    """The independent sampling unit is a training seed, never an eval episode."""
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    if n < 2:
        interval = None
    else:
        critical = {2: 12.706204736432095, 3: 4.302652729911275}.get(n)
        if critical is None:
            from scipy.stats import t
            critical = float(t.ppf(0.975, n - 1))
        radius = critical * sd / math.sqrt(n)
        interval = [mean - radius, mean + radius]
    return {"n": n, "mean": mean, "sd": sd, "ci95": interval}


def classify(values: list[float], threshold: float | None, complete: bool) -> str:
    if not complete:
        return "incomplete"
    if threshold is None:
        return "no_official_threshold"
    if all(value >= threshold for value in values):
        return "all_seeds_pass"
    if statistics.fmean(values) >= threshold:
        return "mean_only_pass"
    return "below_threshold"


def command_options(command: list[str]) -> dict[str, str]:
    return {str(key).removeprefix("--").replace("-", "_"): str(command[index + 1])
            for index, key in enumerate(command[:-1])
            if str(key).startswith("--") and not str(command[index + 1]).startswith("--")}


def discover(runs_root: Path, manifests: list[Path], errors: list[str]) -> list[dict]:
    runs_root = runs_root.resolve()
    jobs: dict[Path, dict] = {}
    for manifest_path in manifests:
        manifest = read_json(manifest_path, errors)
        for job in manifest.get("jobs", []):
            path = Path(job["output"])
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            # A custom smoke root must not accidentally import production jobs.
            if not path.is_relative_to(runs_root):
                continue
            relative = path.relative_to(runs_root).parts
            if len(relative) != 3 or relative[0] not in ("native", "sb3"):
                errors.append(f"unsupported manifest output layout: {path}")
                continue
            jobs[path] = {**job, "backend": relative[0], "condition": relative[1]}
    for backend in ("native", "sb3"):
        for path in sorted((runs_root / backend).glob("*/seed_*")):
            if path.is_dir():
                try:
                    seed = int(path.name.removeprefix("seed_"))
                except ValueError:
                    errors.append(f"invalid seed directory: {path}")
                    continue
                jobs.setdefault(path.resolve(), {"backend": backend,
                                                  "condition": path.parent.name, "seed": seed})
    return [{**job, "path": path} for path, job in sorted(jobs.items())]


def normalize_run(job: dict, configs_dir: Path, test_episodes: int, test_seed: int,
                  ma_window: int) -> tuple[dict, list[dict], list[dict]]:
    path = job["path"]
    errors: list[str] = []
    metadata = read_json(path / "metadata.json", errors)
    final = read_json(path / "final.json", errors)
    progress = read_json(path / "progress.json", errors)
    failure = read_json(path / "failure.json", errors)
    config = read_json(path / "config.json", errors)
    arguments = {**command_options(job.get("command", [])), **config.get("arguments", {})}
    backend, condition = job["backend"], job["condition"]
    case = metadata.get("case", job.get("case", condition))
    native_config = metadata.get("configuration", {})
    if not native_config:
        native_config = read_json(configs_dir / f"{case}.json", errors)
    seed = final.get("seed", metadata.get("seed", arguments.get("seed", job.get("seed"))))
    seed = int(seed) if seed is not None else None
    env = (final.get("env_id") or final.get("environment") or metadata.get("env_id")
           or arguments.get("env") or native_config.get("env_id"))
    specs = metadata.get("effective_agents", native_config.get("agents", []))
    if backend == "native" and "workers" in arguments and len(specs) == 1:
        specs = specs * int(arguments["workers"])
    if backend == "sb3":
        algorithm = final.get("algorithm", arguments.get("algo"))
        if not algorithm:
            algorithm = condition.rsplit("_", 1)[-1]
        specs = [{"algorithm": algorithm}]
    worker_algorithms = {index: spec["algorithm"] for index, spec in enumerate(specs)}
    worker_statistics = final.get("workers", progress.get("workers", []))
    for worker in worker_statistics:
        worker_algorithms[worker["worker"]] = worker["algorithm"]
    if backend == "sb3":
        worker_statistics = [{"worker": 0, "algorithm": specs[0]["algorithm"],
                              "steps": final.get("actual_steps"),
                              "episodes": final.get("completed_episodes"),
                              "statistics": {"updates": final.get("updates")}}]
    status = "missing" if not path.exists() else "incomplete"
    if failure or metadata.get("status") == "failed":
        status = "failed"
    elif final and (backend == "native" and final.get("status") == "complete"
                   or backend == "sb3" and metadata.get("status") == "complete"):
        status = "complete"
    requested = metadata.get("requested_total_steps", final.get("requested_steps",
                       arguments.get("steps", job.get("steps"))))
    requested = int(requested) if requested is not None else None
    actual = final.get("actual_total_steps", final.get("actual_steps", progress.get("aggregate_steps")))
    common = {"backend": backend, "condition": condition, "seed": seed,
              "environment": env, "source": str(path)}
    train: list[dict] = []
    files = sorted(path.glob("train_worker*.jsonl")) if backend == "native" else [path / "train_episodes.jsonl"]
    for source in files:
        recent: deque[float] = deque(maxlen=ma_window)
        completed = 0
        for record in read_jsonl(source, errors):
            worker = int(record.get("worker", 0))
            algorithm = record.get("algorithm", worker_algorithms.get(worker))
            raw = record.get("reward", record.get("return"))
            partial = bool(record.get("budget_cut", False))
            if not finite(raw):
                errors.append(f"{source}: nonfinite training return at episode {record.get('episode')}")
            elif not partial:
                recent.append(raw)
                completed += 1
            train.append({**common, "algorithm": algorithm, "worker": worker,
                          "episode": record.get("episode"), "completed_episode": completed if not partial else None,
                          "aggregate_steps": record.get("aggregate_steps", record.get("env_steps")),
                          "worker_steps": record.get("steps", record.get("env_steps")),
                          "length": record.get("length"), "raw_return": raw,
                          "learning_return": record.get("learning_reward", record.get("train_return")),
                          "terminated": record.get("terminated"), "truncated": record.get("truncated"),
                          "budget_cut": partial,
                          "ma_return": statistics.fmean(recent) if not partial and len(recent) == ma_window else None})
    partial_sb3 = final.get("partial_episode_excluded")
    if backend == "sb3" and partial_sb3:
        train.append({**common, "algorithm": specs[0]["algorithm"], "worker": 0,
                      "episode": partial_sb3.get("episode"), "completed_episode": None,
                      "aggregate_steps": actual, "worker_steps": actual,
                      "length": partial_sb3.get("length"), "raw_return": partial_sb3.get("return"),
                      "learning_return": partial_sb3.get("train_return"),
                      "terminated": False, "truncated": False, "budget_cut": True, "ma_return": None})
    evaluations = []
    for record in read_jsonl(path / "evaluations.jsonl", errors):
        worker = int(record.get("worker", 0))
        evaluations.append({**common, "worker": worker,
                            "algorithm": record.get("algorithm", worker_algorithms.get(worker)),
                            "split": record.get("split", "test" if record.get("phase") == "final" else "validation"),
                            "aggregate_steps": record.get("aggregate_steps", record.get("env_steps")),
                            "mean": record.get("mean", record.get("mean_return")),
                            "episodes": record.get("episodes"), "seed_start": record.get("seed_start")})
    # A final result is never reconstructed from the highest validation score.
    final_tests = final.get("test", []) if backend == "native" else (
        [{"worker": 0, "algorithm": specs[0]["algorithm"], **final["final_evaluation"]}]
        if final.get("final_evaluation") else [])
    tests = {}
    for test in final_tests:
        worker = int(test.get("worker", 0))
        mean = test.get("mean", test.get("mean_return"))
        reasons = []
        if not finite(mean):
            reasons.append("nonfinite or absent final mean")
        if test.get("episodes") != test_episodes:
            reasons.append(f"final episode count {test.get('episodes')} != {test_episodes}")
        if test.get("seed_start") != test_seed:
            reasons.append(f"final evaluation seed {test.get('seed_start')} != {test_seed}")
        if test.get("deterministic") is not True:
            reasons.append("final deterministic evaluation not recorded")
        if test.get("algorithm", worker_algorithms.get(worker)) != worker_algorithms.get(worker):
            reasons.append("final algorithm differs from configured worker")
        if test.get("aggregate_steps", test.get("env_steps")) != actual:
            reasons.append("final evaluation step differs from completed training budget")
        if backend == "native":
            returns = test.get("returns", [])
            if len(returns) != test_episodes or not all(finite(x) for x in returns):
                reasons.append("invalid or missing final raw returns")
            elif finite(mean) and not math.isclose(statistics.fmean(returns), mean, abs_tol=1e-7, rel_tol=1e-7):
                reasons.append("final mean differs from raw returns")
            if test.get("reward") != "raw":
                reasons.append("raw final reward not recorded")
        if worker in tests:
            reasons.append("duplicate final worker")
        tests[worker] = {"worker": worker, "algorithm": test.get("algorithm", worker_algorithms.get(worker)),
                         "mean": mean if finite(mean) else None, "episodes": test.get("episodes"),
                         "seed_start": test.get("seed_start"), "valid": not reasons,
                         "validation_errors": reasons}
    if backend == "sb3" and final_tests:
        rows = [r for r in read_jsonl(path / "eval_episodes.jsonl", errors) if r.get("phase") == "final"]
        returns = [r.get("return") for r in rows]
        test = tests[0]
        if (len(rows) != test_episodes or not all(finite(x) for x in returns)
                or {r.get("seed") for r in rows} != set(range(test_seed, test_seed + test_episodes))):
            test["validation_errors"].append("invalid or missing final per-episode raw returns/seeds")
        elif test["mean"] is not None and not math.isclose(statistics.fmean(returns), test["mean"], abs_tol=1e-7, rel_tol=1e-7):
            test["validation_errors"].append("final mean differs from raw returns")
        test["valid"] = not test["validation_errors"]
    if set(tests) != set(worker_algorithms) and final:
        errors.append("final evaluation worker set differs from configured worker set")
    budget_valid = (requested is not None and actual is not None
                    and (actual == requested if backend == "native" else actual >= requested))
    if final and not budget_valid:
        errors.append("final step budget is missing or was not completed")
    if seed != job.get("seed", seed):
        errors.append("recorded training seed differs from manifest/directory seed")
    counts = Counter(row["worker"] for row in train if not row["budget_cut"])
    for worker in worker_statistics:
        count = worker.get("episodes")
        if final and count is not None and count != counts[worker["worker"]]:
            errors.append(f"worker {worker['worker']}: final episode count differs from full training log")
    rss = final.get("max_rss", progress.get("max_rss", metadata.get("max_rss")))
    rss_unit = final.get("max_rss_unit", metadata.get("max_rss_unit"))
    # Historical smoke logs did not record units. Do not infer from the host
    # rendering this report; only the run's own platform can establish units.
    if rss is not None and rss_unit is None:
        platform = metadata.get("platform", "").lower()
        rss_unit = "bytes" if "macos" in platform or "darwin" in platform else "KiB" if "linux" in platform else None
    rss_bytes = rss if rss_unit == "bytes" else rss * 1024 if rss is not None and rss_unit == "KiB" else None
    run = {**common, "case": case, "status": status, "requested_steps": requested,
           "actual_steps": actual, "budget_valid": budget_valid,
           "worker_count": len(worker_algorithms), "worker_algorithms": worker_algorithms,
           "wall_seconds": final.get("seconds", final.get("total_seconds", failure.get("seconds", progress.get("seconds")))),
           "training_seconds": final.get("training_seconds", final.get("training_seconds_excluding_evaluation", progress.get("training_seconds"))),
           "evaluation_seconds": final.get("evaluation_seconds", progress.get("evaluation_seconds")),
           "initialization_seconds": final.get("initialization_seconds"),
           "policy_parameter_count": config.get("policy_parameter_count"),
           "max_rss": rss, "max_rss_unit": rss_unit, "max_rss_bytes": rss_bytes,
           "rss_scope": "training checkpoint" if backend == "native" and "max_rss" not in final else "process",
           "completed_episodes": sum(counts.values()),
           "budget_cut_episodes": sum(row["budget_cut"] for row in train),
           "worker_statistics": worker_statistics, "final_tests": list(tests.values()),
           "recorded_threshold": metadata.get("reward_threshold", config.get("environment_spec", {}).get("reward_threshold")),
           "official_threshold": THRESHOLDS.get(env),
           "metadata": metadata, "errors": errors,
           "failure": failure or ({"error": metadata.get("error"), "traceback": metadata.get("traceback")}
                                  if metadata.get("status") == "failed" else None)}
    run["valid_final"] = (status == "complete" and budget_valid and not errors
                          and bool(tests) and all(t["valid"] for t in tests.values()))
    return run, train, evaluations


def aggregate(runs: list[dict], expected_seeds: list[int]) -> list[dict]:
    groups = defaultdict(list)
    for run in runs:
        for algorithm in sorted(set(run["worker_algorithms"].values())):
            groups[(run["backend"], run["condition"], algorithm)].append(run)
    summaries = []
    for (backend, condition, algorithm), members in sorted(groups.items()):
        seed_means = {}
        seed_workers = {}
        rejected = {}
        for run in members:
            tests = [test for test in run["final_tests"] if test["algorithm"] == algorithm]
            if run["seed"] not in expected_seeds:
                continue
            if run["valid_final"] and tests:
                if str(run["seed"]) in seed_means:
                    raise ValueError(f"duplicate training seed in {backend}/{condition}/{algorithm}")
                seed_workers[str(run["seed"])] = {str(t["worker"]): t["mean"] for t in tests}
                seed_means[str(run["seed"])] = statistics.fmean(t["mean"] for t in tests)
            else:
                rejected[str(run["seed"])] = {"status": run["status"], "errors": run["errors"],
                                             "test_errors": [t["validation_errors"] for t in tests if not t["valid"]]}
        values = [seed_means[str(seed)] for seed in expected_seeds if str(seed) in seed_means]
        complete = len(values) == len(expected_seeds)
        env = next((r["environment"] for r in members if r["environment"]), None)
        threshold = THRESHOLDS.get(env)
        stats = seed_summary(values)
        worker_means = [mean for workers in seed_workers.values() for mean in workers.values()]
        worker_counts = {r["seed"]: sum(a == algorithm for a in r["worker_algorithms"].values())
                         for r in members if r["seed"] in expected_seeds}
        inferred_workers = max(worker_counts.values(), default=0)
        summaries.append({"backend": backend, "condition": condition, "algorithm": algorithm,
                          "environment": env, "primary": condition in PRIMARY_CASES,
                          "expected_seeds": expected_seeds, "seed_means": seed_means,
                          "worker_means_by_seed": seed_workers, "rejected_seeds": rejected,
                          "missing_seeds": [s for s in expected_seeds if str(s) not in seed_means],
                          "complete": complete, **stats, "official_threshold": threshold,
                          "total_worker_models": len(worker_means),
                          "expected_total_worker_models": sum(worker_counts.get(seed, inferred_workers) for seed in expected_seeds),
                          "worker_models_passing_threshold": sum(value >= threshold for value in worker_means) if threshold is not None else None,
                          "minimum_worker_evaluation_mean": min(worker_means) if worker_means else None,
                          "all_worker_models_pass": all(value >= threshold for value in worker_means)
                          if complete and threshold is not None and worker_means else None,
                          "criterion": classify(values, threshold, complete),
                          "ci95_lower_meets_threshold": (stats["ci95"][0] >= threshold
                                                         if complete and stats["ci95"] is not None and threshold is not None else None)})
    return summaries


def reference_for(group: dict) -> tuple[str | None, str]:
    condition, algorithm = group["condition"], group["algorithm"]
    same = {"cartpole_dqn", "cartpole_ppo", "lunar_dqn", "lunar_ppo", "lunar_sac",
            "ant_ppo", "hopper_sac", "walker_ppo", "halfcheetah_sac", "ant_sac"}
    if condition in same:
        return condition, "同一環境・予算・主要設定の参照実装（実装差は残る）"
    if condition == "cartpole_dqn_parallel4":
        return "cartpole_dqn", "native 4 learner対SB3 1 learner。合計予算一定、各worker予算1/4"
    if condition.startswith("lunar_rnd") or condition == "lunar_ppo_parallel2":
        return "lunar_ppo", "RNDなし単独PPO対照。RND・worker数の構造差を含む"
    if condition == "halfcheetah_hybrid" and algorithm == "sac":
        return "halfcheetah_sac", "native混合4 learnerのSAC 2 worker対単独SAC。各worker予算1/4"
    if condition in ("ant_shared", "ant_rnd_shared") and algorithm == "sac":
        return "ant_sac", "参考比較。native混合SACはupdate interval=2、単独参照=4、worker予算1/2"
    return None, "同一アルゴリズム・設定のSB3対照なし（離散SACはSB3非対応）" if algorithm == "sac" else "同一設定のSB3対照なし"


def add_comparisons(groups: list[dict], expected_seeds: list[int]) -> None:
    lookup = {(g["backend"], g["condition"], g["algorithm"]): g for g in groups}
    for group in groups:
        if group["backend"] != "native":
            continue
        reference, note = reference_for(group)
        other = lookup.get(("sb3", reference, group["algorithm"]))
        comparison = {"reference_condition": reference, "note": note,
                      "complete": bool(group["complete"] and other and other["complete"])}
        if other:
            common = [str(s) for s in expected_seeds if str(s) in group["seed_means"] and str(s) in other["seed_means"]]
            differences = {s: group["seed_means"][s] - other["seed_means"][s] for s in common}
            comparison.update(reference_mean=other["mean"], seed_differences=differences,
                              difference_summary=seed_summary(list(differences.values())),
                              ratio_to_reference=group["mean"] / other["mean"]
                              if group["mean"] is not None and other["mean"] is not None and other["mean"] > 0 else None)
            if group["official_threshold"] is None:
                auxiliary = 0.8 * other["mean"] if comparison["complete"] and other["mean"] > 0 else None
                comparison.update(auxiliary_threshold=auxiliary,
                                  auxiliary_criterion=classify(list(group["seed_means"].values()), auxiliary,
                                                               comparison["complete"]) if auxiliary is not None else "unavailable")
        group["reference"] = comparison


def native_comparisons(groups: list[dict], expected_seeds: list[int]) -> list[dict]:
    pairs = [
        ("lunar_rnd", "lunar_ppo", "ppo", "単独RND有無"),
        ("lunar_ppo_parallel2", "lunar_ppo", "ppo", "PPO並列化、合計予算一定"),
        ("lunar_rnd_parallel2", "lunar_ppo_parallel2", "ppo", "2 worker・別RND有無"),
        ("lunar_rnd_shared2", "lunar_ppo_parallel2", "ppo", "2 worker・共有RND有無"),
        ("lunar_rnd_shared2", "lunar_rnd_parallel2", "ppo", "2 worker・RND共有の効果"),
        ("lunar_rnd_parallel2", "lunar_rnd", "ppo", "RND付きPPO並列化、合計予算一定"),
        ("cartpole_dqn_parallel4", "cartpole_dqn", "dqn", "4 worker共有replay、合計予算一定"),
        ("cartpole_sac_parallel4", "cartpole_sac", "sac", "4 worker共有replay、合計予算一定"),
        ("ant_rnd_shared", "ant_shared", "ppo", "混合構成PPO・RND有無"),
        ("ant_rnd_shared", "ant_shared", "sac", "混合構成SAC・PPO側RND有無"),
        ("ant_shared", "ant_sac", "sac", "混合対単独。SAC更新間隔と予算配分が異なる"),
        ("halfcheetah_hybrid", "halfcheetah_sac", "sac", "混合対単独。SAC worker総予算1/2、各worker1/4"),
    ]
    lookup = {(g["condition"], g["algorithm"]): g for g in groups if g["backend"] == "native"}
    comparisons = []
    for treatment, control, algorithm, note in pairs:
        treated, baseline = lookup.get((treatment, algorithm)), lookup.get((control, algorithm))
        if not treated or not baseline:
            continue
        differences = {str(seed): treated["seed_means"][str(seed)] - baseline["seed_means"][str(seed)]
                       for seed in expected_seeds
                       if str(seed) in treated["seed_means"] and str(seed) in baseline["seed_means"]}
        stats = seed_summary(list(differences.values()))
        complete = treated["complete"] and baseline["complete"]
        comparisons.append({"treatment": treatment, "control": control, "algorithm": algorithm,
                            "note": note, "seed_differences": differences, **stats, "complete": complete,
                            "all_seed_differences_positive": all(value > 0 for value in differences.values()) if complete else None,
                            "ci95_excludes_zero": (stats["ci95"][0] > 0 or stats["ci95"][1] < 0)
                            if complete and stats["ci95"] is not None else None})
    return comparisons


def evaluation_curves(runs: list[dict], records: list[dict], expected_seeds: list[int]) -> list[dict]:
    expected_workers = {(r["backend"], r["condition"], r["seed"], algorithm):
                        {worker for worker, algo in r["worker_algorithms"].items() if algo == algorithm}
                        for r in runs for algorithm in set(r["worker_algorithms"].values())}
    valid_finals = {(r["backend"], r["condition"], r["seed"]) for r in runs if r["valid_final"]}
    buckets = defaultdict(dict)
    for row in records:
        if row["split"] == "test" and (row["backend"], row["condition"], row["seed"]) not in valid_finals:
            continue
        if row["seed"] in expected_seeds and finite(row["mean"]) and row["aggregate_steps"] is not None:
            key = (row["backend"], row["condition"], row["algorithm"], row["split"], row["aggregate_steps"], row["seed"])
            buckets[key][row["worker"]] = row["mean"]
    points = defaultdict(dict)
    for (*prefix, seed), workers in buckets.items():
        backend, condition, algorithm, _, _ = prefix
        if set(workers) == expected_workers.get((backend, condition, seed, algorithm)):
            points[tuple(prefix)][str(seed)] = statistics.fmean(workers.values())
    curves = []
    for (backend, condition, algorithm, split, steps), means in sorted(points.items()):
        curves.append({"backend": backend, "condition": condition, "algorithm": algorithm,
                       "split": split, "aggregate_steps": steps, "seed_means": means,
                       "complete": len(means) == len(expected_seeds), **seed_summary(list(means.values()))})
    return curves


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False, allow_nan=False)
                             if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})


def save_figures(output: Path, groups: list[dict], train: list[dict], curves: list[dict],
                 ma_window: int, expected_seeds: list[int]) -> dict[str, str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 160})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    train_index = defaultdict(list)
    curve_index = defaultdict(list)
    for row in train:
        if row["seed"] in expected_seeds:
            train_index[(row["backend"], row["condition"], row["algorithm"], row["seed"], row["worker"])].append(row)
    for row in curves:
        curve_index[(row["backend"], row["condition"], row["algorithm"])].append(row)
    result = {}
    for group in groups:
        key = (group["backend"], group["condition"], group["algorithm"])
        sides = [(key, "ReinforceX" if key[0] == "native" else "SB3", "#126b9a")]
        reference = group.get("reference", {}).get("reference_condition")
        if reference:
            sides.append((("sb3", reference, group["algorithm"]), "SB3 reference", "#c46a19"))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for side, label, color in sides:
            first = True
            for worker_key, rows in sorted(train_index.items()):
                if worker_key[:3] != side:
                    continue
                completed = [r for r in rows if not r["budget_cut"] and finite(r["raw_return"])]
                if not completed:
                    continue
                xs = np.array([r["completed_episode"] for r in completed])
                raw = np.array([r["raw_return"] for r in completed])
                if len(xs) > 2500:
                    # Preserve the range of every raw episode in an envelope;
                    # downsample only after the full MA has been calculated.
                    bins = np.array_split(np.arange(len(xs)), 2500)
                    axes[0].fill_between([xs[indexes].mean() for indexes in bins],
                                         [raw[indexes].min() for indexes in bins],
                                         [raw[indexes].max() for indexes in bins],
                                         color=color, alpha=0.07, linewidth=0, rasterized=True)
                else:
                    axes[0].plot(xs, raw, color=color, alpha=0.045, linewidth=0.35, rasterized=True)
                smooth = [r for r in completed if r["ma_return"] is not None]
                selected = np.linspace(0, len(smooth) - 1, min(2500, len(smooth)), dtype=int) if smooth else []
                axes[0].plot([smooth[i]["completed_episode"] for i in selected], [smooth[i]["ma_return"] for i in selected],
                             color=color, alpha=0.7, linewidth=1.0,
                             label=f"{label}: each seed / worker" if first else None)
                first = False
            rows = curve_index.get(side, [])
            for complete, style in ((True, "-"), (False, ":")):
                points = sorted((r for r in rows if r["split"] == "validation" and r["complete"] == complete),
                                key=lambda r: r["aggregate_steps"])
                if points:
                    xs = np.array([r["aggregate_steps"] for r in points])
                    ys = np.array([r["mean"] for r in points])
                    axes[1].plot(xs, ys, style, color=color, marker=".",
                                 label=label + (" validation" if complete else " validation (partial seeds)"))
                    if complete and len(expected_seeds) > 1:
                        sd = np.array([r["sd"] for r in points])
                        axes[1].fill_between(xs, ys - sd, ys + sd, color=color, alpha=0.16)
            tests = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["aggregate_steps"])
            for test in tests:
                axes[1].errorbar(test["aggregate_steps"], test["mean"], yerr=test["sd"],
                                 color=color, marker="D", markersize=6, linestyle="none", capsize=4,
                                 label=f"{label} final test (n={test['n']} seeds)")
        threshold = group["official_threshold"]
        if threshold is not None:
            axes[1].axhline(threshold, color="#666666", linestyle="--", linewidth=0.8,
                            label=f"Registered threshold: {threshold:g}")
        axes[0].set(xlabel="Completed episode per worker", ylabel="Raw environment return",
                    title=f"All training episodes and MA{ma_window}")
        axes[1].set(xlabel="Aggregate environment steps", ylabel="Raw evaluation return",
                    title="Validation and final test; mean +/- seed SD")
        axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        for axis in axes:
            axis.grid(alpha=0.15)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=7, loc="best")
        completion = "COMPLETE" if group["complete"] else "PENDING"
        fig.suptitle(f"{group['environment']} | {group['condition']} | {group['algorithm'].upper()}\n"
                     f"Final training seeds: {group['n']}/{len(expected_seeds)} - {completion}")
        axes[0].text(0.01, 0.01, "Display: <=2500 MA points; full episodes in CSV",
                     transform=axes[0].transAxes, fontsize=7, color="#555555")
        stem = f"{group['backend']}__{group['condition']}__{group['algorithm']}"
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"{stem}.{extension}")
        plt.close(fig)
        result["/".join(key)] = f"figures/{stem}.png"
    return result


LABELS = {"incomplete": "判定保留", "no_official_threshold": "公式基準なし",
          "all_seeds_pass": "全seed達成", "mean_only_pass": "平均のみ達成",
          "below_threshold": "未達", "unavailable": "比較不可"}


def number(value: Any, digits: int = 1) -> str:
    return f"{value:,.{digits}f}" if finite(value) else "—"


def markdown_report(summary: dict, output: Path) -> str:
    groups, runs = summary["groups"], summary["runs"]
    expected = summary["protocol"]["training_seeds"]
    completed = [r for r in runs if r["valid_final"] and r["seed"] in expected]
    native = [g for g in groups if g["backend"] == "native"]
    eligible = [g for g in native if g["complete"]]
    counts = Counter(g["criterion"] for g in eligible)
    rel = lambda path: os.path.relpath(path, output)
    lines = ["# CPU学習性能比較レポート", "",
             f"生成日時: {summary['generated_at_utc']}。これは保存済み実測ログの自動集計である。",
             f"予定または検出された{len(runs)} run中、固定予算と最終評価ログの検証を満たしたrunは{len(completed)}。"
             f"native {len(set(g['condition'] for g in native))}条件・{len(native)}条件×algorithm群のうち、"
             f"全{len(expected)} seedが揃った群は{len(eligible)}。未完成・失敗runは成績判定に補完していない。", "",
             f"nativeの公式基準判定は、全seed達成 {counts['all_seeds_pass']}群、平均のみ達成 {counts['mean_only_pass']}群、"
             f"未達 {counts['below_threshold']}群、公式基準なし {counts['no_official_threshold']}群。"
             "これは動作完了の件数とは別である。", "", "## 評価方法", "",
             f"学習seedは{expected}。各workerの最終{summary['protocol']['final_evaluation_episodes']} episode"
             f"（開始seed={summary['protocol']['final_evaluation_seed']}）の生報酬平均を計算し、"
             "同じalgorithmの全workerを等重みにした値をその学習seedの成績とする。"
             "そのseed平均から平均・標本SD（ddof=1）と95% t区間を計算する。3 seedの臨界値は4.3026527299。"
             "評価episode数やworker数を独立した学習seed数に数えない。最良worker・最良checkpoint・最良seedは選択しない。", "",
             "「全seed達成」は各seedのworker平均が達成したという判定であり、全workerモデル達成とは異なる。"
             "一部workerの失敗を隠さないよう、評価できた全モデル中の登録閾値達成数と最低worker平均も別に示す。", "",
             "合否はGymnasium 1.3.0登録閾値による。全seed達成でも95%区間下限が閾値未満の場合があるため、"
             "区間も併記する。Walker2dのSB3平均80%は事前定義した補助基準であり、公式solved条件でも同等性の証明でもない。"
             "学習中の整形報酬ではなく、評価環境が返す生報酬を使う（環境組み込みの報酬設計は維持）。", "",
             f"詳細は[事前プロトコル]({rel(DEFAULT_OUTPUT / 'PROTOCOL.ja.md')})と"
             f"[公式基準・公開benchmarkの出典]({rel(DEFAULT_OUTPUT / 'reference_sources.ja.md')})を参照。"
             "公開MuJoCo-v3/v4やLunarLander-v2の値を現行環境の合格ラインには転用しない。", "",
             "## 最終checkpointの成績", "",
             "表中のseed平均は左から学習seedの記載順。±は学習seed間SD。欠測があれば判定保留。", "",
             "|実装 / 条件 / algorithm|環境|seed平均|平均 ± SD|95% t区間|登録閾値|判定|",
             "|---|---|---|---:|---|---:|---|"]
    profiles = defaultdict(set)
    for run in runs:
        metadata = run["metadata"]
        if metadata:
            profiles[run["backend"]].add(json.dumps({
                "device": metadata.get("device"), "platform": metadata.get("platform"),
                "packages": metadata.get("packages"), "library_sha256": metadata.get("library_sha256"),
                "threads": metadata.get("threads", metadata.get("cpu_threads")),
            }, sort_keys=True, ensure_ascii=False))
    environment_lines = ["## 実測環境", "",
                         "native LibTorch 2.7.0とSB3側PyTorchは異なる実装版であり、bit単位の再現比較ではない。"
                         "Gymnasium/MuJoCo/Box2Dの環境版一致を優先した。以下は検出runのmetadataから転記する。", "",
                         "|実装|実行device / thread設定|依存版|platform|nativeライブラリSHA256|",
                         "|---|---|---|---|---|"]
    for backend, encoded_profiles in sorted(profiles.items()):
        for encoded in sorted(encoded_profiles):
            profile = json.loads(encoded)
            packages = ", ".join(f"{key}={value}" for key, value in (profile["packages"] or {}).items())
            environment_lines.append(f"|{backend}|{profile['device']} / {profile['threads']}|{packages}|"
                                     f"{profile['platform']}|{profile['library_sha256'] or '—'}|")
    environment_lines.append("")
    insert_at = lines.index("## 最終checkpointの成績")
    lines[insert_at:insert_at] = environment_lines
    for g in groups:
        means = " / ".join(number(g["seed_means"].get(str(s))) for s in expected)
        ci = "—" if g["ci95"] is None else " – ".join(number(x) for x in g["ci95"])
        lines.append(f"|{g['backend']} / {g['condition']} / {g['algorithm']}|{g['environment']}|{means}|"
                     f"{number(g['mean'])} ± {number(g['sd'])}|{ci}|{number(g['official_threshold'], 0)}|{LABELS[g['criterion']]}|")
    uncertain = [g for g in groups if g["criterion"] == "all_seeds_pass" and g["ci95_lower_meets_threshold"] is False]
    if uncertain:
        lines.extend(["", "全seed平均は登録基準を満たすが95%区間下限は届かない群: " +
                      ", ".join(f"{g['backend']}/{g['condition']}/{g['algorithm']}" for g in uncertain) + "。"])
    lines.extend(["", "workerモデルの達成内訳（例: 4 worker×3 seedなら12モデル）:", "",
                  "|実装 / 条件 / algorithm|達成モデル / 評価済モデル / 予定モデル|最低worker評価平均|",
                  "|---|---:|---:|"])
    for group in groups:
        passing = number(group["worker_models_passing_threshold"], 0) if group["official_threshold"] is not None else "公式基準なし"
        lines.append(f"|{group['backend']} / {group['condition']} / {group['algorithm']}|"
                     f"{passing} / {group['total_worker_models']} / {group['expected_total_worker_models']}|"
                     f"{number(group['minimum_worker_evaluation_mean'])}|")
    lines.extend(["", "## 同一環境のSB3対照", "",
                  "差は各学習seedのnative−SB3を求めたうえでの平均と95% t区間。"
                  "同じseed番号でも乱数実装は異なり、初期重みや軌跡が一致する意味ではない。"
                  "同じ環境でも構造・予算配分・更新間隔が異なる補助比較を、厳密に同一条件の対照と混同しない。", "",
                  "PPOはSB3がactor/value別のtrunk、nativeが共有部分を持つため、隠れ層幅・深さを揃えても"
                  "総パラメータ数は一致しない。Gaussian分散、勾配clip、DQNのDouble Q対vanilla Qなどの"
                  "アルゴリズム実装差も残る。SB3の実測パラメータ数はruns.csvと各runのconfig.jsonに記録した。", "",
                  "|native条件 / algorithm|SB3対照|平均差|差の95% t区間|補助判定|比較範囲|",
                  "|---|---|---:|---|---|---|"])
    for g in native:
        ref = g["reference"]
        diff = ref.get("difference_summary", {})
        ci = diff.get("ci95")
        auxiliary = ref.get("auxiliary_criterion")
        verdict = "比較不可" if ref["reference_condition"] is None else LABELS.get(auxiliary, "—") if ref["complete"] else "判定保留"
        lines.append(f"|{g['condition']} / {g['algorithm']}|{ref['reference_condition'] or 'なし'}|"
                     f"{number(diff.get('mean'))}|{' – '.join(number(x) for x in ci) if ci else '—'}|{verdict}|{ref['note']}|")
    lines.extend(["", "## 並列agent・RNDの読み方", "",
                  "並列条件の横軸は全workerの合計環境stepである。workerが増えるほど各learnerに割り当てるstepは減る。"
                  "混合条件も各algorithmの全workerを記載し、良かったworkerだけを抜き出していない。"
                  "PPOのupdatesはrollout更新回数、SB3の_n_updatesはalgorithmによりepoch/gradient回数なので、"
                  "数値の大小だけを実装間の学習量の一致とみなさない。", "",
                  "RNDの評価では外発報酬のみを使う。LunarLanderの単独PPO/PPO+RND、"
                  "2 workerのRNDなし/別RND/共有RNDを比較する。AntのPPO+RND+SACにはRND付きPPOが1 workerだけで、"
                  "複数RND learnerが共有する条件とは異なる。探索の効果は3 seedの差とばらつきで判断し、"
                  "平均の一時的な改善だけから一般的優位を主張しない。", "", "## 実行状態・資源・全episode", "",
                  "RSSはプロセス最大常駐メモリをMiBへ変換した値。nativeでprogress.json由来の場合は学習checkpoint時点までの最大値で、"
                  "最終評価・保存後の値を含まない。未記録は—。wall時間は各runの経過時間であり、並列runの合計を実験全体の経過時間とはみなさない。", "",
                  "|実装 / 条件 / seed|状態|実step / 予定|完了episode / 中断|学習秒 / 評価秒 / wall秒|最大RSS MiB|",
                  "|---|---|---:|---:|---:|---:|"])
    comparison_table = [
        "native間の差（処置条件−対照条件）も同じ学習seedを組にして計算する。"
        "複数比較の補正は行っておらず、少数seedでの探索的な効果診断である。", "",
        "|処置−対照 / algorithm|seedごとの差|平均差|95% t区間|全seed改善|比較内容|",
        "|---|---|---:|---|---|---|",
    ]
    for pair in summary["native_comparisons"]:
        differences = " / ".join(number(pair["seed_differences"].get(str(seed))) for seed in expected)
        interval = " – ".join(number(value) for value in pair["ci95"]) if pair["ci95"] else "—"
        verdict = ("はい" if pair["all_seed_differences_positive"] else "いいえ") if pair["complete"] else "判定保留"
        comparison_table.append(f"|{pair['treatment']}−{pair['control']} / {pair['algorithm']}|"
                                f"{differences}|{number(pair['mean'])}|{interval}|{verdict}|{pair['note']}|")
    comparison_table.append("")
    parallel_conditions = {run["condition"] for run in runs if run["backend"] == "native" and run["worker_count"] > 1}
    worker_details = [g for g in native if g["condition"] in parallel_conditions and g["worker_means_by_seed"]]
    if worker_details:
        comparison_table.extend(["並列構成の全worker成績（workerを選抜しない）:", "",
                                 "|条件 / algorithm|seed|workerごとの最終生報酬平均|等重みseed平均|",
                                 "|---|---:|---|---:|"])
        for group in worker_details:
            for seed in expected:
                means = group["worker_means_by_seed"].get(str(seed))
                if means:
                    workers = "; ".join(f"w{worker}: {number(value)}" for worker, value in means.items())
                    comparison_table.append(f"|{group['condition']} / {group['algorithm']}|{seed}|{workers}|"
                                            f"{number(group['seed_means'][str(seed)])}|")
        comparison_table.append("")
    insert_at = lines.index("## 実行状態・資源・全episode")
    lines[insert_at:insert_at] = comparison_table
    for run in runs:
        status = run["status"] if run["valid_final"] or run["status"] != "complete" else "complete (validation rejected)"
        lines.append(f"|{run['backend']} / {run['condition']} / {run['seed']}|{status}|"
                     f"{number(run['actual_steps'], 0)} / {number(run['requested_steps'], 0)}|"
                     f"{run['completed_episodes']} / {run['budget_cut_episodes']}|"
                     f"{number(run['training_seconds'])} / {number(run['evaluation_seconds'])} / {number(run['wall_seconds'])}|"
                     f"{number(run['max_rss_bytes'] / 2**20 if run['max_rss_bytes'] is not None else None)}|")
    lines.extend(["", "各workerのstep・episode・update・loss・RND統計は[worker_statistics.csv](worker_statistics.csv)と"
                  "[summary.json](summary.json)に保存した。"
                  "全episodeの生報酬・学習報酬・terminated/truncated・予算中断フラグは[training_episodes.csv](training_episodes.csv)にある。", "",
                  "## 学習曲線", "",
                  f"左図は学習の全完了episodeの生報酬（薄線）とMA{summary['protocol']['moving_average_window']}（濃線）。"
                  "移動平均は全episodeからworker別に計算し、最初の窓未満は描かない。"
                  "表示のみ各曲線を最大2,500点に間引き、長い生報酬系列は全episodeの区間最小・最大の帯を描く。"
                  "全episodeはCSVに保持する。予算切れの未完episodeは移動平均から除外する。"
                  "右図は合計環境step対固定validation報酬、帯は学習seed間SD。最終testは菱形で別表示する。"
                  "未完成seed集合は点線で示す。異なるstepへの補間・予算外への外挿を行わない。", ""])
    for g in groups:
        key = "/".join((g["backend"], g["condition"], g["algorithm"]))
        image = summary["figures"].get(key)
        if image:
            lines.extend([f"### {key}", "", f"![{key}]({image})", "", f"[PDF]({image[:-4]}.pdf)", ""])
    lines.extend(["## 欠測・失敗・検証エラー", ""])
    errors_present = False
    for run in runs:
        tests = [f"worker {t['worker']}: {', '.join(t['validation_errors'])}" for t in run["final_tests"] if not t["valid"]]
        notes = [*run["errors"], *tests]
        if run["status"] != "complete" or notes:
            errors_present = True
            detail = run["failure"] or {}
            lines.append(f"- {run['backend']}/{run['condition']}/seed_{run['seed']}: {run['status']}; " +
                         ("; ".join(notes) or detail.get("error") or "最終結果未完了。保存ログを参照。"))
    lines.extend(f"- {error}" for error in summary["errors"])
    if not errors_present and not summary["errors"]:
        lines.append("保存済みログの構造・予算・最終評価整合性検証ではエラーなし。性能の合格を意味しない。")
    lines.extend(["", "## 再生成と保存物", "",
                  "```sh", "python3 benchmarks/render_oss_report.py", "```", "",
                  "[summary.json](summary.json) / [summary.csv](summary.csv) / [runs.csv](runs.csv) / "
                  "[評価曲線CSV](evaluation_curves.csv) / [全episode CSV](training_episodes.csv) / "
                  "[native間の比較CSV](native_comparisons.csv)。"
                  "原始ログ・weights・configは入力runディレクトリに残る。", "",
                  "この生成器はcore、FFI、example、学習済みweightsを変更しない。"
                  "開始前後のソースhash照合は別途保存される監査結果を確認する必要があり、"
                  "この集計だけではソース固定の検証完了を主張しない。", ""])
    return "\n".join(lines)


def render(args: argparse.Namespace) -> dict:
    output, runs_root = args.output.resolve(), args.runs_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    manifests = args.manifest if args.manifest is not None else [
        runs_root.parent / "native_manifest.json", runs_root.parent / "sb3_manifest.json"]
    jobs = discover(runs_root, manifests, errors)
    runs, train, evaluations = [], [], []
    for job in jobs:
        run, episodes, points = normalize_run(job, args.configs_dir, args.test_episodes,
                                             args.test_seed, args.ma_window)
        runs.append(run)
        train.extend(episodes)
        evaluations.extend(points)
    groups = aggregate(runs, args.seeds)
    add_comparisons(groups, args.seeds)
    effects = native_comparisons(groups, args.seeds)
    curves = evaluation_curves(runs, evaluations, args.seeds)
    figures = {} if args.no_plots else save_figures(output, groups, train, curves, args.ma_window, args.seeds)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "runs_root": str(runs_root), "manifests": [str(p) for p in manifests if p.exists()],
               "protocol": {"training_seeds": args.seeds, "final_evaluation_episodes": args.test_episodes,
                            "final_evaluation_seed": args.test_seed, "moving_average_window": args.ma_window,
                            "checkpoint_selection": "fixed final only; no best worker or checkpoint selection",
                            "worker_weighting": "equal within each condition / algorithm / training seed",
                            "ci_unit": "training seed", "ci_3seed_t_critical": 4.302652729911275,
                            "official_thresholds": THRESHOLDS},
               "runs": runs, "groups": groups, "native_comparisons": effects, "evaluation_curves": curves,
               "figures": figures, "errors": errors}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(output / "summary.csv", groups)
    write_csv(output / "runs.csv", [{k: v for k, v in run.items() if k not in ("metadata", "worker_statistics", "final_tests")}
                                   for run in runs])
    worker_rows = []
    for run in runs:
        for worker in run["worker_statistics"]:
            final_test = next((t for t in run["final_tests"] if t["worker"] == worker["worker"]), {})
            worker_rows.append({"backend": run["backend"], "condition": run["condition"], "seed": run["seed"],
                                **{key: value for key, value in worker.items() if key != "statistics"},
                                "final_test_mean": final_test.get("mean"), "final_test_valid": final_test.get("valid"),
                                **worker.get("statistics", {})})
    write_csv(output / "worker_statistics.csv", worker_rows)
    write_csv(output / "training_episodes.csv", train)
    write_csv(output / "evaluation_curves.csv", curves)
    write_csv(output / "native_comparisons.csv", effects)
    (output / "REPORT.ja.md").write_text(markdown_report(summary, output))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_OUTPUT / "runs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--configs-dir", type=Path, default=DEFAULT_OUTPUT / "configs")
    parser.add_argument("--manifest", type=Path, action="append", help="Repeatable; default: manifests beside runs root")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=900000)
    parser.add_argument("--ma-window", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        parser.error("training seeds must be unique nonnegative integers")
    if args.test_episodes <= 0 or args.ma_window <= 0 or args.test_seed < 0:
        parser.error("episode counts/window must be positive and final seed nonnegative")
    return args


if __name__ == "__main__":
    result = render(parse_args())
    print(json.dumps({"runs": len(result["runs"]), "valid_final_runs": sum(r["valid_final"] for r in result["runs"]),
                      "groups": len(result["groups"]), "complete_groups": sum(g["complete"] for g in result["groups"]),
                      "figures": len(result["figures"]), "discovery_errors": len(result["errors"]),
                      "run_errors": sum(len(r["errors"]) + sum(len(t["validation_errors"]) for t in r["final_tests"])
                                        for r in result["runs"])}, ensure_ascii=False))
