#!/usr/bin/env python3
"""Diagnose validation deterioration without selecting a checkpoint or a model.

Read-only inputs: fixed-seed evaluations.jsonl and final.json from benchmark
runs. Outputs: stability.json, stability.csv, and STABILITY.ja.md. The primary
performance judgement remains the independent final 100-episode test.
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

try:
    from .render_oss_report import (DEFAULT_OUTPUT, PRIMARY_CASES, THRESHOLDS,
                                   command_options, discover, finite, read_json,
                                   read_jsonl, write_csv)
except ImportError:
    from render_oss_report import (DEFAULT_OUTPUT, PRIMARY_CASES, THRESHOLDS,
                                  command_options, discover, finite, read_json,
                                  read_jsonl, write_csv)


def check_evaluation(record: dict, backend: str, raw_by_point: dict, *,
                     episodes: int, seed_start: int) -> list[str]:
    errors = []
    mean = record.get("mean", record.get("mean_return"))
    if not finite(mean):
        errors.append("nonfinite or missing evaluation mean")
    if record.get("episodes") != episodes:
        errors.append(f"episode count {record.get('episodes')} != {episodes}")
    if record.get("seed_start") != seed_start:
        errors.append(f"evaluation seed {record.get('seed_start')} != {seed_start}")
    if record.get("deterministic") is not True:
        errors.append("deterministic evaluation not recorded")
    if backend == "native":
        returns = record.get("returns", [])
        if record.get("reward") != "raw":
            errors.append("raw reward metric not recorded")
    else:
        raw_rows = raw_by_point.get(record.get("point"), [])
        returns = [row.get("return") for row in raw_rows]
        if {row.get("seed") for row in raw_rows} != set(range(seed_start, seed_start + episodes)):
            errors.append("raw episode seed set mismatch")
        if any(row.get("phase") != record.get("phase") for row in raw_rows):
            errors.append("raw episode evaluation phase mismatch")
    if len(returns) != episodes or not all(finite(value) for value in returns):
        errors.append("missing or nonfinite raw evaluation returns")
    elif finite(mean) and not math.isclose(statistics.fmean(returns), mean, abs_tol=1e-7, rel_tol=1e-7):
        errors.append("recorded mean differs from raw episode returns")
    return errors


def validation_diagnostics(points: list[dict], threshold: float | None) -> dict:
    """Only comparable validation points enter the max/last calculation."""
    ordered = sorted(points, key=lambda point: point["aggregate_steps"])
    peak = max(ordered, key=lambda point: point["mean"]) if ordered else None
    last = ordered[-1] if ordered else None
    crossings = [point for point in ordered if threshold is not None and point["mean"] >= threshold]
    steps = [point["aggregate_steps"] for point in ordered]
    return {"initial_mean": ordered[0]["mean"] if ordered and ordered[0]["aggregate_steps"] == 0 else None,
            "peak_mean": peak["mean"] if peak else None,
            "peak_step": peak["aggregate_steps"] if peak else None,
            "last_mean": last["mean"] if last else None,
            "last_step": last["aggregate_steps"] if last else None,
            "peak_minus_last": peak["mean"] - last["mean"] if peak and last else None,
            "first_threshold_step": crossings[0]["aggregate_steps"] if crossings else None,
            "threshold_reached_then_last_validation_below": bool(crossings) and last["mean"] < threshold
            if ordered and threshold is not None else None,
            "validation_points": ordered,
            "duplicate_validation_steps": sorted(step for step, count in Counter(steps).items() if count > 1)}


def analyze_run(job: dict, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    path, backend = job["path"], job["backend"]
    errors = []
    metadata = read_json(path / "metadata.json", errors)
    final = read_json(path / "final.json", errors)
    failure = read_json(path / "failure.json", errors)
    saved_config = read_json(path / "config.json", errors)
    arguments = {**command_options(job.get("command", [])), **saved_config.get("arguments", {})}
    case = metadata.get("case", job.get("case", job["condition"]))
    config = metadata.get("configuration", {}) or read_json(args.configs_dir / f"{case}.json", errors)
    seed = int(final.get("seed", metadata.get("seed", arguments.get("seed", job["seed"]))))
    env = (final.get("env_id") or final.get("environment") or metadata.get("env_id")
           or arguments.get("env") or config.get("env_id"))
    threshold = THRESHOLDS.get(env)
    specs = metadata.get("effective_agents", config.get("agents", []))
    if backend == "native" and len(specs) == 1 and "workers" in arguments:
        specs = specs * int(arguments["workers"])
    worker_algorithms = {worker: spec["algorithm"] for worker, spec in enumerate(specs)}
    if backend == "sb3":
        algorithm = final.get("algorithm", arguments.get("algo", job["condition"].rsplit("_", 1)[-1]))
        worker_algorithms = {0: algorithm}
    requested_steps = metadata.get("requested_total_steps", final.get("requested_steps", job.get("steps", arguments.get("steps"))))
    requested_steps = int(requested_steps) if requested_steps is not None else None
    actual_steps = final.get("actual_total_steps", final.get("actual_steps"))
    status = "missing" if not path.exists() else "pending"
    if failure or metadata.get("status") == "failed":
        status = "failed"
    elif final and (backend == "native" and final.get("status") == "complete"
                   or backend == "sb3" and metadata.get("status") == "complete"):
        status = "complete"
    budget_complete = (requested_steps is not None and actual_steps is not None
                       and (actual_steps == requested_steps if backend == "native" else actual_steps >= requested_steps))
    if status == "complete" and not budget_complete:
        status = "invalid"
        errors.append("completed run did not satisfy recorded step budget")
    if seed != job["seed"]:
        errors.append("training seed differs from manifest/directory seed")
    raw_by_point = defaultdict(list)
    if backend == "sb3":
        for row in read_jsonl(path / "eval_episodes.jsonl", errors):
            raw_by_point[row.get("point")].append(row)
    validation = defaultdict(list)
    invalid_points = defaultdict(list)
    for record in read_jsonl(path / "evaluations.jsonl", errors):
        is_validation = record.get("split") == "validation" if backend == "native" else record.get("phase") in ("initial", "progress")
        if not is_validation:
            continue
        worker = int(record.get("worker", 0))
        if worker not in worker_algorithms:
            errors.append(f"validation worker {worker} not in configured worker set")
            continue
        point_errors = check_evaluation(record, backend, raw_by_point,
                                        episodes=args.validation_episodes, seed_start=args.validation_seed)
        step = record.get("aggregate_steps", record.get("env_steps"))
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            point_errors.append("invalid aggregate step")
        if record.get("algorithm", worker_algorithms[worker]) != worker_algorithms[worker]:
            point_errors.append("algorithm differs from configured worker")
        if point_errors:
            invalid_points[worker].append({"aggregate_steps": step, "errors": point_errors})
            continue
        validation[worker].append({"aggregate_steps": step,
                                   "mean": record.get("mean", record.get("mean_return")),
                                   "episodes": record["episodes"], "seed_start": record["seed_start"]})
    final_records = final.get("test", []) if backend == "native" else (
        [{"worker": 0, **final["final_evaluation"]}] if final.get("final_evaluation") else [])
    tests = {}
    for record in final_records:
        worker = int(record.get("worker", 0))
        point_errors = check_evaluation(record, backend, raw_by_point,
                                        episodes=args.test_episodes, seed_start=args.test_seed)
        if record.get("aggregate_steps", record.get("env_steps")) != actual_steps:
            point_errors.append("final evaluation step differs from final training step")
        if record.get("algorithm", worker_algorithms.get(worker)) != worker_algorithms.get(worker):
            point_errors.append("final algorithm differs from configured worker")
        if (record.get("split") != "test" if backend == "native" else record.get("phase") != "final"):
            point_errors.append("final evaluation split/phase mismatch")
        if worker in tests or worker not in worker_algorithms:
            point_errors.append("duplicate or unconfigured final worker")
        tests[worker] = {"mean": record.get("mean", record.get("mean_return")),
                         "valid": not point_errors, "errors": point_errors}
    if status == "complete" and set(tests) != set(worker_algorithms):
        errors.append("final test worker set incomplete")
    common = {"backend": backend, "condition": job["condition"], "case": case,
              "primary_condition": job["condition"] in PRIMARY_CASES,
              "environment": env, "seed": seed, "run_status": status,
              "requested_steps": requested_steps, "actual_steps": actual_steps,
              "official_threshold": threshold, "source": str(path)}
    rows = []
    for worker, algorithm in sorted(worker_algorithms.items()):
        diagnostic = validation_diagnostics(validation[worker], threshold)
        test = tests.get(worker, {})
        test_valid = bool(status == "complete" and not errors and test.get("valid"))
        final_mean = test.get("mean") if finite(test.get("mean")) else None
        last_step = diagnostic["last_step"]
        validation_budget_complete = (last_step is not None and budget_complete
                                      and (last_step == actual_steps if backend == "native"
                                           else requested_steps <= last_step <= actual_steps))
        diagnostics_complete = (status == "complete" and not errors and not invalid_points[worker]
                                and not diagnostic["duplicate_validation_steps"]
                                and len(validation[worker]) == args.expected_checkpoints + 1
                                and diagnostic["initial_mean"] is not None
                                and validation_budget_complete and test_valid)
        diagnostic_status = status
        if diagnostics_complete:
            diagnostic_status = "complete"
        elif status == "complete":
            diagnostic_status = "invalid" if errors or invalid_points[worker] or not test_valid else "incomplete_validation"
        rows.append({**common, "worker": worker, "algorithm": algorithm,
                     "diagnostics_status": diagnostic_status,
                     "validation_points_valid": len(validation[worker]),
                     "validation_points_expected": args.expected_checkpoints + 1,
                     **diagnostic, "final_test_mean": final_mean, "final_test_valid": test_valid,
                     "steps_after_last_validation": actual_steps - last_step if actual_steps is not None and last_step is not None else None,
                     "final_test_meets_threshold": final_mean >= threshold if test_valid and threshold is not None else None,
                     "threshold_reached_then_final_test_below": diagnostic["first_threshold_step"] is not None and final_mean < threshold
                     if test_valid and threshold is not None else None,
                     "invalid_validation_points": invalid_points[worker], "final_test_errors": test.get("errors", []),
                     "run_errors": errors})
    run = {**common, "worker_count": len(worker_algorithms), "errors": errors,
           "diagnostics_complete": bool(rows) and all(row["diagnostics_status"] == "complete" for row in rows)}
    return run, rows


def format_number(value):
    return f"{value:,.2f}" if finite(value) else "—"


def markdown(result: dict) -> str:
    runs, rows = result["runs"], result["workers"]
    complete = sum(run["diagnostics_complete"] for run in runs)
    crossings = [row for row in rows if row["threshold_reached_then_final_test_below"] is True]
    lines = ["# 学習中の性能低下診断", "", f"生成日時: {result['generated_at_utc']}。",
             f"検出/予定{len(runs)} run・{len(rows)} workerモデルのうち、全validationと最終testが揃ったrunは{complete}。"
             f"一時的に登録閾値へ到達し、最終testでは未達だったworkerは{len(crossings)}。"
             "未完了runのpeak/lastは保存済み範囲だけの暫定値である。", "",
             "## 読み方と限界", "",
             f"validationは各{result['protocol']['validation_episodes']} episode、固定seed "
             f"{result['protocol']['validation_seed']}から。初期評価と定期checkpoint評価のみを比較する。"
             "peakはその平均の最大値、同点なら最初のcheckpoint。lastは最後に保存されたvalidation。"
             "peak−lastは同一固定seed評価内の低下幅であり、checkpointの選択基準ではない。"
             "初回閾値stepは初めて閾値以上だった観測checkpointで、学習中の厳密な初回到達stepではない。", "",
             "最大値には複数checkpointから最大を取る選抜バイアスがあり、10 episodeという少数評価も影響する。"
             "peakを成績として採用せず、bestモデルの選択・保存・再評価やハイパーパラメータの再調整は行わない。", "",
             f"最終testは別の{result['protocol']['test_episodes']} episode、seed {result['protocol']['test_seed']}から。"
             "validationとはseed集合・episode数が異なるため、last−testの差をそのまま学習劣化とは解釈できない。"
             "SB3のprogress callbackはrollout/gradient更新前に呼ばれ、最後のvalidationから最終testまでに"
             "学習更新が入る。環境stepが同じでも厳密には同一policyの評価とは限らない。"
             "主判定は最終100 episode testのみで行う。", "",
             "登録閾値はGymnasium 1.3.0の環境基準。Walker2dには公式閾値がないので到達判定は空欄。"
             "並列条件のstepは全workerの合計で、表は最良workerを選ばず全workerを列挙する。", "",
             "## 一時達成後に最終test未達となったworker", "",
             "|実装 / 条件 / seed / worker|algorithm|登録閾値|初回到達step|peak (step)|last|peak−last|最終test|",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    for row in crossings:
        lines.append(f"|{row['backend']} / {row['condition']} / {row['seed']} / w{row['worker']}|{row['algorithm']}|"
                     f"{format_number(row['official_threshold'])}|{row['first_threshold_step']}|"
                     f"{format_number(row['peak_mean'])} ({row['peak_step']})|{format_number(row['last_mean'])}|"
                     f"{format_number(row['peak_minus_last'])}|{format_number(row['final_test_mean'])}|")
    if not crossings:
        lines.append("|該当なし（未完了結果は判定対象外）|—|—|—|—|—|—|—|")
    lines.extend(["", "## 条件ごとの診断状況", "",
                  "以下は診断完了workerだけのpeak−lastを要約する。未完runの暫定値はJSON/CSVに残す。"
                  "最大低下は不安定なworkerを見落とさないための指標で、最良モデル選抜ではない。"
                  "報酬尺度の異なる環境間で低下量の大小を比較しない。", "",
                  "|実装 / 条件 / algorithm|診断完了 / 予定モデル|peak−last平均|最大peak−last|",
                  "|---|---:|---:|---:|"])
    groups = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["condition"], row["algorithm"])].append(row)
    for (backend, condition, algorithm), members in sorted(groups.items()):
        decreases = [row["peak_minus_last"] for row in members if row["diagnostics_status"] == "complete"]
        lines.append(f"|{backend} / {condition} / {algorithm}|{len(decreases)} / {len(members)}|"
                     f"{format_number(statistics.fmean(decreases) if decreases else None)}|"
                     f"{format_number(max(decreases) if decreases else None)}|")
    lines.extend(["", "詳細なエラーとvalidation全点は[stability.json](stability.json)、"
                  "worker別の指標は[stability.csv](stability.csv)に保存した。これは診断であり、"
                  "この表のpeakや到達stepを最終性能の主表へ置き換えない。", "",
                  "```sh", "python3 benchmarks/analyze_stability.py", "```", ""])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = args.manifest if args.manifest is not None else [
        args.runs_root.parent / "native_manifest.json", args.runs_root.parent / "sb3_manifest.json"]
    errors = []
    jobs = discover(args.runs_root, manifests, errors)
    runs, workers = [], []
    for job in jobs:
        run, records = analyze_run(job, args)
        runs.append(run)
        workers.extend(records)
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "runs_root": str(args.runs_root.resolve()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": {"validation_episodes": args.validation_episodes, "validation_seed": args.validation_seed,
                           "expected_validation_checkpoints": args.expected_checkpoints,
                           "test_episodes": args.test_episodes, "test_seed": args.test_seed,
                           "peak_tie_rule": "earliest aggregate step", "checkpoint_selection": "none; diagnosis only"},
              "runs": runs, "workers": workers, "errors": errors}
    (args.output / "stability.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "stability.csv", [{key: value for key, value in row.items() if key != "validation_points"} for row in workers])
    (args.output / "STABILITY.ja.md").write_text(markdown(result))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_OUTPUT / "runs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--configs-dir", type=Path, default=DEFAULT_OUTPUT / "configs")
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=800000)
    parser.add_argument("--expected-checkpoints", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=900000)
    args = parser.parse_args()
    if min(args.validation_episodes, args.test_episodes) <= 0 or min(args.validation_seed, args.test_seed, args.expected_checkpoints) < 0:
        parser.error("episode counts must be positive; seeds/checkpoint count must be nonnegative")
    return args


if __name__ == "__main__":
    result = analyze(parse_args())
    print(json.dumps({"runs": len(result["runs"]), "workers": len(result["workers"]),
                      "runs_with_complete_diagnostics": sum(run["diagnostics_complete"] for run in result["runs"]),
                      "transient_threshold_then_final_test_failure_workers":
                      sum(row["threshold_reached_then_final_test_below"] is True for row in result["workers"]),
                      "discovery_errors": len(result["errors"])}, ensure_ascii=False))
