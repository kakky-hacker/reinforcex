#!/usr/bin/env python3
"""Audit the supplementary Tianshou runs without altering training artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

from audit_results import (Checks, check_summary, digest, load_json, option,
                           read_run_files, resolve, snapshot_check)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"


def audit_run(job, frozen):
    checks = Checks()
    output = resolve(job["output"], ROOT)
    complete = (output / "final.json").exists()
    files = read_run_files(output, complete, checks)
    m, f, config = (files.get(name, {}) for name in ("metadata.json", "final.json", "config.json"))
    if not complete:
        return {"id": job["id"], "status": "failed" if m.get("status") == "failed" else
                "invalid" if checks.errors else "pending",
                "checks_passed": checks.passed, "errors": checks.errors, "notes": checks.notes}
    source = load_json(resolve(option(job["command"], "--config"), ROOT))
    checks.equal(m["status"], "complete", "metadata.status")
    checks.equal(config["source_config"], source, "source_config")
    for key, expected in (("seed", job["seed"]), ("requested_steps", job["steps"]),
                          ("actual_steps", job["steps"])):
        checks.equal(m[key], expected, "metadata." + key)
        checks.equal(f[key], expected, "final." + key)
    checks.equal(m["device"], "cpu", "device")
    checks.equal(m["torch_num_threads"], 1, "torch_threads")
    checks.equal(m["torch_num_interop_threads"], 1, "torch_interop_threads")
    checks.equal(m["versions"], {"tianshou": "2.0.1", "torch": "2.7.0", "gymnasium": "1.3.0",
                                 "numpy": "2.4.6", "numba": "0.67.0"}, "versions")
    checks.equal(m["runner_sha256"], frozen["benchmarks/run_tianshou_discrete.py"], "runner_hash")
    checks.equal(m["config_sha256"], frozen["reports/oss_benchmarks/configs/cartpole_sac.json"], "config_hash")
    for name, expected in m["official_source_sha256"].items():
        checks.equal(digest(Path(name)), expected, "installed_official_source_hash")
    checks.equal(config["resolved"], {
        "net_arch": [64, 64], "activation": "ReLU", "actor_softmax_output": False,
        "critic_initialization": "two independently initialized networks", "gamma": .99,
        "actor_learning_rate": .0003, "critic_learning_rate": .0005, "alpha_learning_rate": .0003,
        "target_entropy": .01 * math.log(2), "initial_alpha": .05, "tau": .005,
        "n_step_return_horizon": 3, "batch_size": 64, "replay_capacity": 50000,
        "learning_starts": 512, "update_interval": 1, "target_update_interval": 1,
        "critic_loss": "MSE", "max_grad_norm": None, "actor_parameters": 4610,
        "critic_parameters_each": [4610, 4610]}, "resolved_config")
    rows = files["train_episodes.jsonl"]
    cumulative = 0
    for index, row in enumerate(rows):
        checks.equal(row["episode"], index + 1, "train.sequence")
        checks.check(row["terminated"] or row["truncated"], "train.complete")
        checks.check(0 < row["length"] <= 500, "train.length")
        cumulative += row["length"]
        checks.equal(row["env_steps"], cumulative, "train.cumulative_steps")
        checks.near(row["return"], row["length"], "train.raw_return")
        expected = .01 * row["length"] - (1.01 if row["terminated"] and row["length"] < 500 else 0)
        checks.near(row["train_return"], expected, "train.learning_reward")
    partial = f["partial_episode_excluded"]
    if partial:
        checks.equal(partial["complete"], False, "partial.incomplete")
        checks.equal(partial["episode"], len(rows) + 1, "partial.sequence")
        checks.check(0 < partial["length"] < 500, "partial.length")
        checks.near(partial["return"], partial["length"], "partial.raw_return")
        checks.near(partial["train_return"], .01 * partial["length"], "partial.learning_reward")
        cumulative += partial["length"]
    checks.equal(cumulative, job["steps"], "all_transition_accounting")
    checks.equal(f["completed_episodes"], len(rows), "complete_episode_count")
    checks.equal(f["updates"], job["steps"] - 511, "updates_after_warmup")
    checks.near(f["last100_training_mean_return"], statistics.mean(r["return"] for r in rows[-100:]), "last100_mean")
    grouped = defaultdict(list)
    for row in files["eval_episodes.jsonl"]:
        grouped[row["point"]].append(row)
        checks.near(row["return"], row["length"], "eval.raw_return")
        checks.check(row["terminated"] or row["truncated"], "eval.complete")
    points = files["evaluations.jsonl"]
    count = int(option(job["command"], "--checkpoints", 10))
    checks.equal(len(points), count + 2, "eval.point_count")
    checks.equal(len(grouped), len(points), "eval.no_extra_points")
    for index, point in enumerate(points):
        phase = "initial" if index == 0 else "final" if index == count + 1 else "progress"
        checks.equal(point["point"], index + 1, "eval.point_sequence")
        checks.equal(point["phase"], phase, "eval.phase")
        steps = 0 if index == 0 else job["steps"] if phase == "final" else math.ceil(job["steps"] * index / count)
        checks.equal(point["env_steps"], steps, "eval.steps")
        checks.equal(point["updates"], max(0, steps - 511), "eval.updates")
        expected_count = int(option(job["command"], "--eval-episodes", 100)) if phase == "final" else int(option(job["command"], "--progress-eval-episodes", 10))
        expected_seed = 900000 if phase == "final" else 800000
        episodes = grouped[index + 1]
        checks.equal(len(episodes), expected_count, "eval.episode_count")
        checks.equal([r["seed"] for r in episodes], list(range(expected_seed, expected_seed + expected_count)), "eval.seeds")
        checks.equal([r["episode"] for r in episodes], list(range(1, expected_count + 1)), "eval.episode_sequence")
        checks.equal(point["seed_start"], expected_seed, "eval.seed_start")
        checks.check(all(r["phase"] == phase and r["env_steps"] == steps for r in episodes), "eval.episode_context")
        checks.equal(point["deterministic"], True, "eval.deterministic")
        check_summary(checks, point, [r["return"] for r in episodes], [r["length"] for r in episodes], True, "eval")
        checks.equal("passed" in point, phase == "final", "final_only_judgment")
    checks.equal(f["final_evaluation"], points[-1], "final_matches_final_point")
    checks.equal(points[-1]["success_threshold"], 475.0, "final.threshold")
    checks.equal(points[-1]["passed"], points[-1]["mean_return"] >= 475.0, "final.passed")
    checkpoint = output / "final_model.pt"
    checks.check(checkpoint.is_file() and checkpoint.stat().st_size > 0, "checkpoint_exists")
    checks.equal(digest(checkpoint), f["checkpoint_sha256"], "checkpoint_hash")
    checks.near(f["training_seconds_excluding_evaluation"] + f["evaluation_seconds"], f["total_seconds"], "timing_sum")
    checks.equal(f["max_rss_unit"], "bytes", "rss_unit")
    checks.check(f["max_rss"] > 0, "rss_positive")
    return {"id": job["id"], "status": "passed" if not checks.errors else "invalid",
            "checks_passed": checks.passed, "errors": checks.errors, "notes": checks.notes,
            "final_mean_return": f["final_evaluation"]["mean_return"],
            "actual_steps": f["actual_steps"], "final_episodes": points[-1]["episodes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BASE / "tianshou_manifest.json")
    parser.add_argument("--output", type=Path, default=BASE / "tianshou_audit.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    checks = Checks()
    checks.equal(sorted(job["seed"] for job in manifest["jobs"]), [42, 123, 2026], "manifest.seed_set")
    checks.check(all(job["steps"] == 204800 for job in manifest["jobs"]), "manifest.budgets")
    checks.equal(manifest["total_steps"], sum(j["steps"] for j in manifest["jobs"]), "manifest.total_steps")
    checks.equal(len({j["id"] for j in manifest["jobs"]}), len(manifest["jobs"]), "manifest.unique_ids")
    for path, expected in manifest["frozen_sha256"].items():
        checks.equal(digest(ROOT / path), expected, "supplement_frozen_source")
    snapshots = [snapshot_check(BASE / name, ROOT) for name in
                 ("source_snapshot_before.json", "benchmark_code_snapshot_frozen.json")]
    results = []
    for job in manifest["jobs"]:
        try:
            results.append(audit_run(job, manifest["frozen_sha256"]))
        except (OSError, KeyError, TypeError, ValueError) as error:
            results.append({"id": job["id"], "status": "invalid", "error": str(error)})
    invalid = any(r["status"] in ("failed", "invalid") for r in results)
    invalid |= bool(checks.errors) or any(s["status"] != "passed" for s in snapshots)
    pending = sum(r["status"] == "pending" for r in results)
    report = {"status": "failed" if invalid else "incomplete" if pending else "passed",
              "supplementary": True, "expected_runs": len(results), "pending_runs": pending,
              "passed_runs": sum(r["status"] == "passed" for r in results),
              "failed_or_invalid_runs": sum(r["status"] in ("failed", "invalid") for r in results),
              "manifest_errors": checks.errors, "snapshots": snapshots, "runs": results}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("snapshots", "runs")}, indent=2))
    return 1 if invalid else 2 if pending and not args.allow_incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
