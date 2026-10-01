#!/usr/bin/env python3
"""Independently audit native/SB3 benchmark artifacts without loading either model.

Exit 0: all checks passed, or only pending runs remain with --allow-incomplete.
Exit 1: an integrity check or completed/failed run failed. Exit 2: pending runs.
The report always labels pending runs explicitly, including in permissive mode.
Only the requested audit report is written; training artifacts are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def reject_constant(value: str):
    raise ValueError(f"Non-finite JSON constant: {value}")


def finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{path}[{index}]")


def load_json(path: Path) -> Any:
    value = json.loads(path.read_text(), parse_constant=reject_constant)
    finite(value)
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256")
    return checksum.hexdigest()


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def option(command: list[str], name: str, default: Any = None) -> Any:
    if name in command:
        index = command.index(name)
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            return True
        return command[index + 1]
    return default


def close(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        left, right, rel_tol=1e-10, abs_tol=1e-8
    )


class Checks:
    def __init__(self):
        self.passed = 0
        self.errors: list[dict[str, Any]] = []
        self.notes: list[str] = []

    def check(self, condition: bool, code: str, detail: Any = None) -> None:
        if condition:
            self.passed += 1
        else:
            self.errors.append({"code": code, "detail": detail})

    def equal(self, actual: Any, expected: Any, code: str) -> None:
        self.check(actual == expected, code, {"expected": expected, "actual": actual})

    def near(self, actual: Any, expected: Any, code: str) -> None:
        self.check(close(actual, expected), code, {"expected": expected, "actual": actual})


def check_summary(checks: Checks, summary: dict[str, Any], returns: list[float],
                  lengths: list[int], sb3: bool, label: str) -> None:
    checks.equal(summary["episodes"], len(returns), label + ".episode_count")
    checks.equal(len(lengths), len(returns), label + ".length_count")
    checks.check(bool(returns), label + ".nonempty")
    if not returns:
        return
    suffix = "_return" if sb3 else ""
    for key, value in (
        ("mean", statistics.mean(returns)),
        ("std", statistics.stdev(returns) if len(returns) > 1 else 0.0),
        ("min", min(returns)), ("max", max(returns)),
    ):
        checks.near(summary[key + suffix], value, label + ".recomputed_" + key)
    if sb3:
        checks.near(summary["mean_length"], statistics.mean(lengths), label + ".mean_length")
        margin = 1.96 * (statistics.stdev(returns) if len(returns) > 1 else 0.) / math.sqrt(len(returns))
        for actual, expected in zip(summary["normal_approx_ci95"],
                                    [statistics.mean(returns) - margin, statistics.mean(returns) + margin]):
            checks.near(actual, expected, label + ".normal_approx_ci95")
    checks.check(all(isinstance(n, int) and n > 0 for n in lengths), label + ".positive_lengths")


def read_run_files(output: Path, complete: bool, checks: Checks) -> dict[str, Any]:
    result = {}
    for path in sorted(output.glob("*.json")):
        try:
            result[path.name] = load_json(path)
            checks.passed += 1
        except (ValueError, OSError) as error:
            checks.check(False, "json_parse_or_finiteness", {"file": str(path), "error": str(error)})
    for path in sorted(output.glob("*.jsonl")):
        rows = []
        try:
            # A single read is a consistent byte prefix of an actively appended log.
            lines = path.read_text().splitlines(keepends=True)
            for index, line in enumerate(lines):
                if not line.strip():
                    checks.check(False, "empty_jsonl_record", {"file": str(path), "line": index + 1})
                    continue
                try:
                    row = json.loads(line, parse_constant=reject_constant)
                    finite(row)
                except ValueError as error:
                    if not complete and index == len(lines) - 1 and not line.endswith("\n"):
                        checks.notes.append(f"Pending append omitted: {path.name}:{index + 1}")
                        break
                    checks.check(False, "jsonl_parse_or_finiteness", {
                        "file": str(path), "line": index + 1, "error": str(error),
                    })
                    continue
                rows.append(row)
            result[path.name] = rows
            checks.passed += 1
        except OSError as error:
            checks.check(False, "jsonl_read", {"file": str(path), "error": str(error)})
    return result


def check_checkpoint(path: Path, checks: Checks, label: str) -> None:
    checks.check(path.is_file() and path.stat().st_size > 0, label, str(path))


def check_learning_return(checks: Checks, raw: float, learned: float, length: int,
                          terminated: bool, limit: int, mode: str) -> None:
    if mode == "cartpole":
        expected = .01 * raw - (1.01 if terminated and length < limit else 0.)
    elif mode in ("scale", "scale0.1"):
        expected = .1 * raw
    elif mode == "hopper":
        expected = raw - length
    elif mode in ("ant", "ant_shared"):
        expected = .1 * (raw - length)
    else:
        expected = raw
    checks.near(learned, expected, "train.learning_reward_transform")


def expected_native_agents(config: dict[str, Any], command: list[str]) -> list[dict[str, Any]]:
    agents = config["agents"]
    count = option(command, "--workers")
    if count is not None:
        if len(agents) != 1:
            raise ValueError("--workers is only valid for a single-agent source configuration")
        agents = agents * int(count)
    return agents


def audit_native(job: dict[str, Any], files: dict[str, Any], output: Path, config: dict[str, Any],
                 checks: Checks, root: Path, complete: bool, hash_cache: dict[Path, str]) -> None:
    metadata = files["metadata.json"]
    command = job["command"]
    agents = expected_native_agents(config, command)
    count = len(agents)
    budget = job["steps"]
    checks.equal(metadata["case"], job["case"], "metadata.case")
    checks.equal(metadata["seed"], job["seed"], "metadata.seed")
    checks.equal(metadata["env_id"], config["env_id"], "metadata.environment")
    checks.equal(metadata["configuration"], config, "metadata.config_export")
    checks.equal(metadata["effective_agents"], agents, "metadata.effective_agents")
    checks.equal(metadata["worker_count"], count, "metadata.worker_count")
    checks.equal(metadata["requested_total_steps"], budget, "metadata.budget")
    checks.equal(metadata["share_rnd"], "--share-rnd" in command, "metadata.share_rnd")
    checks.equal(metadata["concurrent_workers"], "--serial-workers" not in command, "metadata.concurrency")
    checks.equal(metadata["device"], "cpu", "metadata.cpu")
    checks.equal(metadata["validation_seed"], 800000, "metadata.validation_seed")
    checks.equal(metadata["final_test_seed"], 900000, "metadata.final_seed")
    checks.equal(metadata["final_test_episodes"], 100, "metadata.final_episodes")
    library = resolve(metadata["library"], root)
    if library not in hash_cache:
        hash_cache[library] = digest(library)
    checks.equal(metadata["library_sha256"], hash_cache[library], "metadata.library_hash_unchanged")
    counts: Counter[str] = Counter()
    expected_seeds = []
    for spec in agents:
        algorithm = spec["algorithm"]
        ordinal = counts[algorithm]
        counts[algorithm] += 1
        seed = job["seed"] + ordinal * 10000
        expected_seeds.append({
            "algorithm": algorithm, "algorithm_worker": ordinal,
            "policy_seed": seed, "environment_seed": seed,
            "rnd_seed": job["seed"] + 1_000_000 + (0 if metadata["share_rnd"] else ordinal * 10000),
        })
    checks.equal(metadata["worker_seeds"], expected_seeds, "metadata.worker_seeds")
    checks.equal(budget % count, 0, "budget.divisible_by_workers")
    per_worker = budget // count
    final = files.get("final.json") if complete else None
    final_workers = {row["worker"]: row for row in final["workers"]} if final else {}
    if final:
        for name in ["case", "seed", "env_id"]:
            checks.equal(final[name], metadata[name], "final." + name)
        checks.equal(final["status"], "complete", "final.status")
        checks.equal(final["actual_total_steps"], budget, "final.actual_equals_requested")
        checks.equal(sorted(final_workers), list(range(count)), "final.worker_ids")
        checks.equal(sum(row["steps"] for row in final["workers"]), budget, "final.worker_step_sum")
    logged_total = 0
    aggregate_ends = []
    for index, spec in enumerate(agents):
        key = f"train_worker{index}.jsonl"
        checks.check(key in files, "train_log.exists", key)
        rows = files.get(key, [])
        cumulative, complete_episodes, cuts = 0, 0, 0
        for ordinal, row in enumerate(rows, 1):
            cumulative += row["length"]
            cuts += int(row["budget_cut"])
            complete_episodes += int(not row["budget_cut"])
            checks.equal(row["worker"], index, "train.worker")
            checks.equal(row["algorithm"], spec["algorithm"], "train.algorithm")
            checks.equal(row["episode"], ordinal, "train.episode_order")
            checks.equal(row["steps"], cumulative, "train.cumulative_lengths")
            checks.check(0 < row["length"] <= metadata["max_episode_steps"], "train.length_bounds")
            check_learning_return(checks, row["reward"], row["learning_reward"], row["length"],
                                  row["terminated"], metadata["max_episode_steps"], config["reward_mode"])
            checks.check(cumulative <= row["aggregate_steps"] <= budget, "train.aggregate_bounds")
            aggregate_ends.append(row["aggregate_steps"])
            if row["budget_cut"]:
                checks.check(not row["terminated"] and not row["truncated"], "train.budget_cut_not_terminal")
                checks.equal(ordinal, len(rows), "train.budget_cut_is_last")
                checks.equal(row["steps"], per_worker, "train.budget_cut_at_budget")
            else:
                checks.check(row["terminated"] or row["truncated"], "train.completed_episode_has_end")
        checks.check(cuts <= 1, "train.at_most_one_budget_cut")
        logged_total += cumulative
        if complete:
            checks.equal(cumulative, per_worker, "train.equal_worker_budget")
            worker = final_workers[index]
            checks.equal(worker["steps"], cumulative, "train.final_steps")
            checks.equal(worker["episodes"], complete_episodes, "train.final_complete_episode_count")
            updates = worker["statistics"].get("updates", worker["statistics"].get("n_updates", 0))
            checks.check(updates > 0, "train.worker_did_update")
            if spec["algorithm"] == "sac":
                for component in ["actor", "critic1", "critic2", "temperature"]:
                    check_checkpoint(output / f"worker{index}_{component}.ot", checks, "checkpoint.sac_component")
            else:
                check_checkpoint(output / f"worker{index}.ot", checks, "checkpoint.policy")
            if spec.get("rnd_config") is not None:
                rnd_index = 0 if metadata["share_rnd"] else index
                for component in ["predictor", "target"]:
                    check_checkpoint(output / f"rnd_worker{rnd_index}" / f"rnd_{component}.ot",
                                     checks, "checkpoint.rnd_component")
    checks.equal(len(set(aggregate_ends)), len(aggregate_ends), "train.unique_aggregate_episode_end_steps")
    if complete:
        checks.equal(logged_total, budget, "train.all_lengths_including_budget_cut_equal_actual")
    evals = files.get("evaluations.jsonl", [])
    by_worker: dict[int, list] = defaultdict(list)
    for record in evals:
        index = record["worker"]
        checks.check(index in range(count), "evaluation.worker_id")
        if index not in range(count):
            continue
        by_worker[index].append(record)
        checks.equal(record["algorithm"], agents[index]["algorithm"], "evaluation.algorithm")
        checks.equal(record["reward"], "raw", "evaluation.raw_reward")
        checks.equal(record["deterministic"], True, "evaluation.deterministic")
        test = record["split"] == "test"
        checks.equal(record["seed_start"], 900000 if test else 800000, "evaluation.fixed_seed_start")
        checks.equal(record["episodes"], 100 if test else metadata["validation_episodes"], "evaluation.expected_count")
        check_summary(checks, record, record["returns"], record["lengths"], False, "evaluation")
        checks.check(all(n <= metadata["max_episode_steps"] for n in record["lengths"]), "evaluation.time_limit")
    if complete:
        checkpoints = int(option(command, "--checkpoints", 10))
        expected_progress = [budget * i // checkpoints for i in range(checkpoints + 1)]
        final_test = {row["worker"]: row for row in final["test"]}
        checks.equal(len(final["test"]), count, "final.test_worker_count")
        for index in range(count):
            validation = [r for r in by_worker[index] if r["split"] == "validation"]
            tests = [r for r in by_worker[index] if r["split"] == "test"]
            checks.equal([r["aggregate_steps"] for r in validation], expected_progress, "evaluation.checkpoint_steps")
            checks.equal(len(tests), 1, "evaluation.one_final_test_per_worker")
            if tests:
                checks.equal(tests[0], final_test.get(index), "evaluation.jsonl_equals_final_summary")
                checks.equal(tests[0]["aggregate_steps"], budget, "evaluation.final_at_budget")
        progress = files.get("progress.json", {})
        checks.equal(progress.get("workers"), final["workers"], "progress.final_worker_stats")
        checks.equal(progress.get("checkpoint"), checkpoints, "progress.last_checkpoint")
    checks.notes.append("Native evaluation seeds are reconstructed as seed_start+episode index; raw returns/lengths are individually stored in each summary.")


def expected_sb3_hyperparameters(config: dict[str, Any], steps: int) -> dict[str, Any]:
    spec = config["agents"][0]
    cfg = spec["config"]
    common = cfg["agent"]
    result = {"gamma": common["gamma"]}
    if spec["algorithm"] == "ppo":
        mapping = {"learning_rate": "learning_rate", "gae_lambda": "gae_lambda",
                   "n_steps": "update_interval", "n_epochs": "epochs", "batch_size": "minibatch_size",
                   "clip_range": "policy_clip_epsilon", "clip_range_vf": "value_clip_range",
                   "vf_coef": "value_loss_coefficient", "ent_coef": "entropy_coefficient"}
        result.update({dest: cfg[source] for dest, source in mapping.items()})
        result["normalize_advantage"] = bool(cfg["standardize_gae"])
    elif spec["algorithm"] == "dqn":
        mapping = {"learning_rate": "learning_rate", "batch_size": "batch_size", "buffer_size": "replay_capacity",
                   "learning_starts": "batch_size", "n_steps": "replay_n_steps", "train_freq": "update_interval",
                   "target_update_interval": "target_update_interval", "exploration_initial_eps": "epsilon_start",
                   "exploration_final_eps": "epsilon_end"}
        result.update({dest: cfg[source] for dest, source in mapping.items()})
        result.update(gradient_steps=1, exploration_fraction=cfg["epsilon_decay_steps"] / steps)
    else:
        mapping = {"learning_rate": "actor_learning_rate", "batch_size": "batch_size", "buffer_size": "replay_capacity",
                   "learning_starts": "replay_start_size", "n_steps": "replay_n_steps", "train_freq": "update_interval"}
        result.update({dest: cfg[source] for dest, source in mapping.items()})
        result.update(gradient_steps=1, target_update_interval=1, ent_coef=f"auto_{cfg['alpha']}",
                      target_entropy="auto", tau=1-(1-cfg["tau"])**(cfg["update_interval"]/cfg["target_update_interval"]))
    return result


def audit_sb3(job: dict[str, Any], files: dict[str, Any], output: Path, config: dict[str, Any],
              checks: Checks, root: Path, complete: bool, frozen: dict[str, str]) -> None:
    metadata = files["metadata.json"]
    config_record = files["config.json"]
    args = config_record["arguments"]
    command = job["command"]
    config_path = resolve(option(command, "--config"), root)
    checks.equal(metadata["command"], command, "metadata.command_matches_manifest")
    checks.equal(metadata["script_sha256"], frozen["benchmarks/run_sb3.py"], "metadata.frozen_script_hash")
    checks.equal(metadata["native_config_sha256"], digest(config_path), "metadata.native_config_hash")
    checks.equal(resolve(metadata["native_config_path"], root), config_path, "metadata.native_config_path")
    checks.equal(args["native_config"], config, "config.native_config_export")
    checks.equal(args["env"], config["env_id"], "config.environment")
    checks.equal(args["algo"], config["agents"][0]["algorithm"], "config.algorithm")
    checks.equal(args["seed"], job["seed"], "config.seed")
    checks.equal(args["steps"], job["steps"], "config.budget")
    checks.equal(resolve(args["output"], root), output, "config.output")
    checks.equal(metadata["device"], "cpu", "metadata.cpu")
    checks.equal(metadata["threads"], 1, "metadata.cpu_threads")
    checks.equal(metadata["n_envs"], 1, "metadata.single_environment")
    checks.equal(args["eval_episodes"], 100, "config.final_evaluation_episodes")
    checks.equal(args["final_eval_seed"], 900000, "config.final_evaluation_seed")
    checks.equal(args["validation_seed"], 800000, "config.validation_seed")
    source_agent = config["agents"][0]["config"]["agent"]
    resolved = config_record["resolved_constructor"]
    checks.equal(resolved["policy_kwargs"]["net_arch"], [source_agent["hidden_size"]] * (source_agent["hidden_layers"] + 1),
                 "config.hidden_layer_mapping")
    checks.equal(resolved["policy_kwargs"]["activation_fn"], "<class 'torch.nn.modules.activation.ReLU'>", "config.relu")
    for key, expected in expected_sb3_hyperparameters(config, job["steps"]).items():
        if isinstance(expected, float):
            checks.near(resolved[key], expected, "config.native_mapping." + key)
        else:
            checks.equal(resolved[key], expected, "config.native_mapping." + key)
    expected_reward = {"raw": "raw", "cartpole": "cartpole", "scale0.1": "scale", "hopper": "hopper", "ant_shared": "ant"}
    checks.equal(args["reward_transform"], expected_reward[config["reward_mode"]], "config.reward_transform")
    checks.near(args["reward_scale"], 0.1, "config.reward_scale")
    final = files.get("final.json") if complete else None
    rows = files.get("train_episodes.jsonl", [])
    cumulative = 0
    for ordinal, row in enumerate(rows, 1):
        cumulative += row["length"]
        checks.equal(row["episode"], ordinal, "train.episode_order")
        checks.equal(row["env_steps"], cumulative, "train.cumulative_lengths")
        checks.check(0 < row["length"] <= config_record["environment_spec"]["max_episode_steps"], "train.length_bounds")
        checks.check(row["terminated"] or row["truncated"], "train.complete_episode_has_end")
        check_learning_return(checks, row["return"], row["train_return"], row["length"], row["terminated"],
                              config_record["environment_spec"]["max_episode_steps"], args["reward_transform"])
    if final:
        checks.equal(metadata["status"], "complete", "metadata.complete")
        checks.equal(final["actual_steps"], job["steps"], "final.actual_equals_requested")
        checks.equal(final["requested_steps"], job["steps"], "final.requested_steps")
        checks.equal(final["seed"], job["seed"], "final.seed")
        checks.equal(final["environment"], config["env_id"], "final.environment")
        checks.equal(final["algorithm"], args["algo"], "final.algorithm")
        checks.equal(final["completed_episodes"], len(rows), "final.completed_episode_count")
        partial = final["partial_episode_excluded"]
        partial_steps = partial["length"] if partial else 0
        checks.equal(cumulative + partial_steps, final["actual_steps"], "train.complete_plus_partial_equals_actual")
        if partial:
            checks.equal(partial["complete"], False, "train.partial_not_complete")
            checks.equal(partial["episode"], len(rows) + 1, "train.partial_episode_number")
            checks.check(0 < partial_steps <= config_record["environment_spec"]["max_episode_steps"], "train.partial_length")
            check_learning_return(checks, partial["return"], partial["train_return"], partial_steps, False,
                                  config_record["environment_spec"]["max_episode_steps"], args["reward_transform"])
        last = rows[-100:]
        checks.equal(final["last100_training_episode_count"], len(last), "train.final_last100_count")
        if last:
            checks.near(final["last100_training_mean_return"], statistics.mean(r["return"] for r in last),
                        "train.final_last100_mean")
        checks.check(final["updates"] > 0, "train.learner_did_update")
        checkpoint = resolve(final["checkpoint"], root)
        checks.equal(checkpoint, output / "final_model.zip", "checkpoint.expected_path")
        check_checkpoint(checkpoint, checks, "checkpoint.nonempty")
    points = files.get("evaluations.jsonl", [])
    episode_records = files.get("eval_episodes.jsonl", [])
    by_point: dict[int, list] = defaultdict(list)
    for row in episode_records:
        by_point[row["point"]].append(row)
    for index, point in enumerate(points, 1):
        checks.equal(point["point"], index, "evaluation.point_order")
        phase = point["phase"]
        records = by_point[index]
        if not complete and len(records) < point["episodes"]:
            # Separate JSONL files cannot be snapshotted atomically while running.
            # The point summary may have appeared after the episode-log read.
            checks.notes.append(f"Pending evaluation log snapshot at point {index}")
            continue
        checks.equal([r["episode"] for r in records], list(range(1, point["episodes"] + 1)), "evaluation.episode_order")
        expected_count = 100 if phase == "final" else args["progress_eval_episodes"]
        expected_seed = 900000 if phase == "final" else 800000
        checks.equal(point["episodes"], expected_count, "evaluation.phase_episode_count")
        checks.equal(point["seed_start"], expected_seed, "evaluation.phase_seed_start")
        checks.equal([r["seed"] for r in records], list(range(expected_seed, expected_seed + expected_count)),
                     "evaluation.individual_fixed_seeds")
        checks.equal(point["deterministic"], True, "evaluation.deterministic")
        for row in records:
            checks.equal(row["phase"], phase, "evaluation.episode_phase")
            checks.equal(row["env_steps"], point["env_steps"], "evaluation.episode_model_step")
            checks.check(row["terminated"] or row["truncated"], "evaluation.episode_completed")
        check_summary(checks, point, [r["return"] for r in records], [r["length"] for r in records], True, "evaluation")
        if phase != "final":
            checks.check("passed" not in point, "evaluation.no_interim_pass_judgment")
    if complete:
        expected_steps = sorted({math.ceil(job["steps"] * i / args["eval_checkpoints"])
                                 for i in range(1, args["eval_checkpoints"] + 1)})
        checks.equal([p["phase"] for p in points], ["initial"] + ["progress"] * len(expected_steps) + ["final"],
                     "evaluation.all_requested_phases")
        checks.equal([p["env_steps"] for p in points if p["phase"] == "progress"], expected_steps, "evaluation.checkpoint_steps")
        checks.equal(len(episode_records), sum(p["episodes"] for p in points), "evaluation.all_episode_records_accounted_for")
        if points:
            checks.equal(final["final_evaluation"], points[-1], "evaluation.final_equals_jsonl_summary")
            checks.equal(points[-1]["env_steps"], final["actual_steps"], "evaluation.final_at_budget")
            checks.equal(points[0]["env_steps"], 0, "evaluation.initial_at_zero")


def snapshot_check(path: Path, root: Path) -> dict[str, Any]:
    result = {"path": str(path), "files": [], "status": "passed"}
    try:
        data = load_json(path)
        result["snapshot_sha256"] = digest(path)
        for name, expected in data["files"].items():
            file = resolve(name, root)
            actual = digest(file) if file.is_file() else None
            passed = expected == actual
            result["files"].append({"path": name, "expected": expected, "actual": actual, "passed": passed})
            if not passed:
                result["status"] = "failed"
    except (ValueError, OSError, KeyError) as error:
        result.update(status="failed", error=str(error))
    return result


def audit_job(job: dict[str, Any], root: Path, base: Path, frozen: dict[str, str],
              hash_cache: dict[Path, str], status_record: dict[str, Any] | None) -> dict[str, Any]:
    output = resolve(job["output"], root)
    record = {"id": job["id"], "output": str(output), "status": "pending", "checks_passed": 0,
              "errors": [], "notes": [], "pending_reason": None}
    checks = Checks()
    complete = (output / "final.json").is_file()
    if not output.exists():
        record["pending_reason"] = "not_started_or_output_missing"
        if status_record and status_record.get("status") == "failed":
            record.update(status="failed", errors=[{"code": "runner_failed", "detail": status_record}])
        return record
    files = read_run_files(output, complete, checks)
    metadata = files.get("metadata.json")
    failed = "failure.json" in files or (metadata and metadata.get("status") == "failed")
    if status_record and status_record.get("status") == "failed":
        failed = True
    if failed:
        checks.check(False, "runner_failed", files.get("failure.json", status_record or (metadata or {}).get("error")))
    if metadata is None:
        record["pending_reason"] = "initialization_not_finished"
        if complete:
            checks.check(False, "completed_run_missing_metadata")
    else:
        try:
            command = job["command"]
            checks.equal(int(option(command, "--steps")), job["steps"], "manifest.command_steps")
            checks.equal(int(option(command, "--seed")), job["seed"], "manifest.command_seed")
            checks.equal(resolve(option(command, "--output"), root), output, "manifest.command_output")
            config = load_json(base / "configs" / (job["case"] + ".json"))
            sb3 = any(Path(piece).name == "run_sb3.py" for piece in command)
            record["backend"] = "sb3" if sb3 else "native"
            expected_packages = {"gymnasium": "1.3.0", "numpy": "2.4.6", "mujoco": "3.13.0", "Box2D": "2.3.10"}
            if sb3:
                expected_packages.update({"stable-baselines3": "2.9.0", "torch": "2.14.0"})
            for name, version in expected_packages.items():
                checks.equal(metadata["packages"][name], version, "metadata.package_version." + name)
            if sb3 and complete and metadata.get("status") == "running":
                # final.json is atomically written just before metadata switches
                # to complete; do not mark that brief finishing window corrupt.
                complete = False
                record["pending_reason"] = "final_metadata_commit_pending"
            if sb3 and "config.json" not in files:
                record["pending_reason"] = "model_initialization_not_finished"
                if complete:
                    checks.check(False, "completed_sb3_missing_config")
            elif sb3:
                audit_sb3(job, files, output, config, checks, root, complete, frozen)
            else:
                audit_native(job, files, output, config, checks, root, complete, hash_cache)
            if not complete:
                record["pending_reason"] = record["pending_reason"] or "training_or_final_evaluation_in_progress"
        except (KeyError, TypeError, ValueError, OSError, IndexError, ZeroDivisionError) as error:
            checks.check(False, "schema_or_read_failure", f"{type(error).__name__}: {error}")
    if complete and not checks.errors:
        record.update(status="passed", pending_reason=None)
    elif checks.errors:
        record["status"] = "failed" if failed else "invalid"
    record.update(checks_passed=checks.passed, errors=checks.errors, notes=checks.notes,
                  final_present=complete, parsed_json_files=len([n for n in files if n.endswith(".json")]),
                  parsed_jsonl_records=sum(len(rows) for name, rows in files.items() if name.endswith(".jsonl")))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base", type=Path, default=Path("reports/oss_benchmarks"))
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    base = resolve(args.base, root)
    manifests = [resolve(p, root) for p in args.manifest] if args.manifest else [
        base / "native_manifest.json", base / "sb3_manifest.json",
    ]
    output = resolve(args.output, root) if args.output else base / "results_audit.json"
    snapshots = [snapshot_check(base / name, root) for name in [
        "source_snapshot_before.json", "benchmark_code_snapshot_frozen.json",
    ]]
    try:
        frozen = load_json(base / "benchmark_code_snapshot_frozen.json")["files"]
    except (ValueError, OSError, KeyError):
        frozen = {}
    checks = Checks()
    jobs, manifest_records, statuses = [], [], {}
    for path in manifests:
        try:
            data = load_json(path)
            jobs.extend(data["jobs"])
            checks.equal(sum(j["steps"] for j in data["jobs"]), data["total_steps"], "manifest.total_steps")
            manifest_records.append({"path": str(path), "sha256": digest(path), "jobs": len(data["jobs"])})
            status_path = path.with_suffix(".status.json")
            if status_path.is_file():
                # The scheduler rewrites this file non-atomically. A partial status
                # snapshot is informational; final artifact validation is decisive.
                try:
                    statuses.update({r["id"]: r for r in load_json(status_path)})
                except (ValueError, OSError):
                    checks.notes.append(f"Scheduler status currently being written: {status_path}")
        except (ValueError, OSError, KeyError) as error:
            checks.check(False, "manifest_load", {"path": str(path), "error": str(error)})
    ids = [j["id"] for j in jobs]
    outputs = [resolve(j["output"], root) for j in jobs]
    checks.equal(len(ids), len(set(ids)), "manifest.unique_job_ids")
    checks.equal(len(outputs), len(set(outputs)), "manifest.unique_output_directories")
    if any(output == path or path in output.parents for path in outputs):
        raise ValueError("Audit report must be outside all training output directories")
    hash_cache: dict[Path, str] = {}
    results = [audit_job(j, root, base, frozen, hash_cache, statuses.get(j["id"])) for j in jobs]
    counts = Counter(r["status"] for r in results)
    errors = bool(checks.errors) or any(s["status"] != "passed" for s in snapshots) or counts["failed"] or counts["invalid"]
    pending = counts["pending"]
    status = "failed" if errors else ("incomplete" if pending else "passed")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "all_runs_complete_and_valid": status == "passed",
        "allow_incomplete": args.allow_incomplete,
        "manifests": manifest_records,
        "counts": {"expected": len(jobs), "passed": counts["passed"], "invalid": counts["invalid"],
                   "failed": counts["failed"], "pending": pending,
                   "missing_outputs": sum(not p.exists() for p in outputs)},
        "global_checks_passed": checks.passed,
        "global_errors": checks.errors,
        "notes": checks.notes + [
            "Integrity audit only: passing does not imply that an environment's reward threshold was solved.",
            "Native logs contain seed_start and ordered individual returns, not an explicit seed per evaluation episode.",
            "Current source hashes verify the saved snapshots; native metadata cannot independently attest the script bytes loaded by the process.",
            "Active runs are read as available prefixes and remain pending until final artifacts exist.",
        ],
        "snapshots": snapshots,
        "runs": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": status, "counts": report["counts"], "output": str(output)}, ensure_ascii=False))
    return 1 if errors else (2 if pending and not args.allow_incomplete else 0)


if __name__ == "__main__":
    raise SystemExit(main())
