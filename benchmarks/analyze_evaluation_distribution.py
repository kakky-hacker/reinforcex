#!/usr/bin/env python3
"""Describe each final model's fixed 100-episode raw-return distribution.

Read-only analysis of the native, SB3, Tianshou discrete SAC and Tianshou Double
DQN manifests. No inference, model selection, confidence interval or new pass
criterion is performed. Missing/in-progress runs remain explicitly pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"
BACKENDS = ("native", "sb3", "tianshou", "tianshou_dqn")
SEED_START, EPISODES = 900000, 100
METRICS = ("episodes", "mean", "sd", "min", "p05", "median", "p95", "max",
           "return_below_zero_count", "episodes_at_or_above_registered_threshold")


def read_json(path, errors, fingerprints):
    if not path.exists():
        return {}
    data = path.read_bytes()
    fingerprints[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        errors.append(f"{path.name}: {error}")
        return {}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def linear_quantile(values, probability):
    """Hyndman-Fan type 7: linearly interpolate at (n - 1) * probability."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile needs observations and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(returns, threshold):
    if len(returns) != EPISODES or not all(finite(value) for value in returns):
        raise ValueError("distribution requires exactly 100 finite final-test returns")
    return {"episodes": len(returns), "mean": statistics.fmean(returns),
            "sd": statistics.stdev(returns), "min": min(returns),
            "p05": linear_quantile(returns, .05), "median": linear_quantile(returns, .5),
            "p95": linear_quantile(returns, .95), "max": max(returns),
            "return_below_zero_count": sum(value < 0 for value in returns),
            "episodes_at_or_above_registered_threshold":
                sum(value >= threshold for value in returns) if threshold is not None else None}


def command_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def discover(base, errors, fingerprints):
    jobs = []
    for backend in BACKENDS:
        manifest = read_json(base / f"{backend}_manifest.json", errors, fingerprints)
        if not manifest:
            errors.append(f"missing/unreadable manifest: {backend}_manifest.json")
        for job in manifest.get("jobs", []):
            path = Path(job["output"])
            jobs.append({**job, "backend": backend,
                         "path": path if path.is_absolute() else ROOT / path})
    identifiers = [(job["backend"], job["condition"], job["seed"]) for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        errors.append("duplicate manifest backend/condition/seed entries")
    return jobs


def final_episodes(backend, path, final, worker, algorithm, actual_steps, errors, fingerprints):
    if backend == "native":
        records = [row for row in final.get("test", []) if row.get("worker") == worker]
        if len(records) != 1:
            errors.append(f"expected exactly one final test record for worker {worker}")
            return [], []
        test = records[0]
        if test.get("split") != "test" or test.get("reward") != "raw":
            errors.append("native final evaluation must be test split and raw reward")
        if test.get("algorithm") != algorithm:
            errors.append("native final algorithm differs from configured worker")
        returns, lengths = test.get("returns", []), test.get("lengths", [])
        mean = test.get("mean")
        step = test.get("aggregate_steps")
    else:
        test = final.get("final_evaluation", {})
        if test.get("phase") != "final":
            errors.append("reference evaluation phase is not final")
        log = path / "eval_episodes.jsonl"
        records = []
        if log.exists():
            data = log.read_bytes()
            fingerprints[str(log.resolve())] = hashlib.sha256(data).hexdigest()
            try:
                records = [json.loads(line) for line in data.splitlines() if line.strip()]
            except (ValueError, UnicodeDecodeError) as error:
                errors.append(f"eval_episodes.jsonl: {error}")
        records = [row for row in records if row.get("phase") == "final"]
        if len(records) != EPISODES or {row.get("seed") for row in records} != set(range(SEED_START, SEED_START + EPISODES)):
            errors.append("final episode count/seed set missing, duplicated or incorrect")
        if any(row.get("point") != test.get("point") or row.get("env_steps") != actual_steps for row in records):
            errors.append("final episode point/step differs from final evaluation")
        records.sort(key=lambda row: row.get("seed", -1))
        returns = [row.get("return") for row in records]
        lengths = [row.get("length") for row in records]
        mean = test.get("mean_return")
        step = test.get("env_steps")
    if test.get("episodes") != EPISODES or test.get("seed_start") != SEED_START:
        errors.append("final evaluation must contain 100 episodes starting at seed 900000")
    if test.get("deterministic") is not True:
        errors.append("final evaluation is not recorded as deterministic")
    if step != actual_steps:
        errors.append("final evaluation step differs from completed training step")
    if len(returns) != EPISODES or not all(finite(value) for value in returns):
        errors.append("final returns are incomplete or nonfinite")
    elif not finite(mean) or not math.isclose(statistics.fmean(returns), mean, abs_tol=1e-7, rel_tol=1e-7):
        errors.append("recorded final mean differs from raw episode returns")
    if len(lengths) != EPISODES or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in lengths):
        errors.append("final episode lengths are incomplete or invalid")
    return returns, lengths


def analyze_run(job, base, thresholds, fingerprints):
    errors, path, backend = [], job["path"], job["backend"]
    metadata = read_json(path / "metadata.json", errors, fingerprints)
    final = read_json(path / "final.json", errors, fingerprints)
    config = read_json(base / "configs" / f"{job['case']}.json", errors, fingerprints)
    saved_config = read_json(path / "config.json", errors, fingerprints)
    config = config or metadata.get("configuration", {}) or saved_config.get("source_config", {})
    environment = (metadata.get("env_id") or final.get("environment") or final.get("env_id")
                   or saved_config.get("env_id") or config.get("env_id"))
    threshold = thresholds.get(environment)
    expected_specs = config.get("agents", [])
    workers = command_value(job.get("command", []), "--workers")
    if workers:
        expected_specs *= int(workers)
    specs = metadata.get("effective_agents", expected_specs)
    if backend != "native":
        base_algorithm = (config.get("agents") or [{}])[0].get("algorithm", job["case"].rsplit("_", 1)[-1])
        expected_algorithm = {"tianshou": "discrete_sac", "tianshou_dqn": "double_dqn"}.get(backend, base_algorithm)
        for recorded_algorithm in (final.get("algorithm"), metadata.get("algorithm")):
            if recorded_algorithm is not None and recorded_algorithm != expected_algorithm:
                errors.append("reference algorithm differs from manifest backend/config")
        specs = [{"algorithm": expected_algorithm}]
    elif expected_specs and [(v.get("algorithm"), v.get("rnd_config") is not None) for v in specs] != [
            (v.get("algorithm"), v.get("rnd_config") is not None) for v in expected_specs]:
        errors.append("native worker roles differ from manifest config")
        specs = expected_specs
    if not specs:
        errors.append("unable to determine expected model count from manifest config")
    complete = bool(final) and (final.get("status") == "complete" if backend == "native" else metadata.get("status") == "complete")
    status = "complete" if complete else "pending"
    if (path / "failure.json").exists() or metadata.get("status") == "failed":
        status = "failed"
    if complete:
        if environment not in thresholds:
            errors.append("environment absent from registered-threshold snapshot")
        if final.get("seed", metadata.get("seed")) != job["seed"]:
            errors.append("final training seed differs from manifest")
    actual_steps = final.get("actual_total_steps", final.get("actual_steps"))
    if complete and (not isinstance(actual_steps, int) or
                     (actual_steps != job["steps"] if backend == "native" else actual_steps < job["steps"])):
        errors.append("completed training step budget mismatch")
    if complete and backend == "native" and {row.get("worker") for row in final.get("test", [])} != set(range(len(specs))):
        errors.append("native final test worker set differs from configured models")
    common = {"backend": backend, "condition": job["condition"], "case": job["case"],
              "seed": job["seed"], "environment": environment, "source": str(path.resolve()),
              "supplementary": backend in ("tianshou", "tianshou_dqn"),
              "registered_reward_threshold": threshold, "actual_steps": actual_steps,
              "expected_episodes": EPISODES, "evaluation_seed_start": SEED_START}
    rows = []
    for worker, spec in enumerate(specs):
        row_errors, returns, lengths = list(errors), [], []
        if status == "complete":
            returns, lengths = final_episodes(backend, path, final, worker, spec["algorithm"], actual_steps, row_errors, fingerprints)
        distribution_status = "invalid" if status == "complete" and row_errors else status
        metrics = describe(returns, threshold) if distribution_status == "complete" else dict.fromkeys(METRICS)
        rows.append({**common, "worker": worker, "algorithm": spec["algorithm"],
                     "status": distribution_status, **metrics, "errors": row_errors,
                     "evaluation_returns": returns if distribution_status == "complete" else [],
                     "evaluation_lengths": lengths if distribution_status == "complete" else []})
    run = {**common, "status": "invalid" if status == "complete" and any(row["status"] != "complete" for row in rows) else status,
           "expected_worker_models": len(specs), "complete_worker_models": sum(row["status"] == "complete" for row in rows),
           "errors": errors}
    return run, rows


def markdown(result):
    rows = result["workers"]
    complete = [row for row in rows if row["status"] == "complete"]
    lines = ["# 最終評価100エピソードの報酬分布", "", f"生成日時: {result['generated_at_utc']}。",
             f"manifest {len(result['runs'])} runs / {len(rows)} workerモデル。"
             f"状態は {dict(Counter(row['status'] for row in rows))}。"
             f"集計済み{len(complete) * EPISODES} episodeは各モデル別の固定test seed 900000–900099。", "",
             "[全workerのCSV](evaluation_distribution.csv) / [詳細JSON・生return配列](evaluation_distribution.json)。"
             "mean/SD/min/p05/median/p95/max、return < 0件数、登録閾値以上のepisode件数を記録した。"
             "SDは各モデルの100 episode内の標本標準偏差（ddof=1）、分位点は(n−1)p位置の線形補間（type 7）。"
             "訓練seed間・worker間の分散とは異なり、訓練seedを混ぜたCIは計算しない。", "",
             "負報酬の件数と登録閾値以上の件数は分布の記述値で、episode単位の正式な成功/失敗判定でも追加合否gateでもない。"
             "登録閾値のないWalker2dは閾値件数をnull/空欄とした。閾値は"
             "[Gymnasium 1.3.0の保存済み登録値](gymnasium_130_thresholds.json)を使う。"
             "主判定は最終100 episode平均からworker等重み→訓練seed平均を作る既存の3seed評価を維持する。"
             "再推論、設定変更、モデル選抜は行わない。", "",
             "|実装|予定run|集計済みrun|予定model|集計済みmodel|pending|invalid/failed|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for backend in BACKENDS:
        runs = [run for run in result["runs"] if run["backend"] == backend]
        subset = [row for row in rows if row["backend"] == backend]
        lines.append(f"|{backend}|{len(runs)}|{sum(run['status'] == 'complete' for run in runs)}|{len(subset)}|"
                     f"{sum(row['status'] == 'complete' for row in subset)}|{sum(row['status'] == 'pending' for row in subset)}|"
                     f"{sum(row['status'] in ('invalid', 'failed') for row in subset)}|")
    negatives = [row for row in complete if row["return_below_zero_count"]]
    lines.extend(["", f"負報酬episodeを含むモデルは{len(negatives)}個。以下はmanifest順の先頭20個で、全件はCSVにある。"
                  "負報酬を失敗と同義には扱わない。", "",
                  "|実装 / 条件 / 訓練seed / worker|mean|SD|最低|p05|中央値|p95|最高|負報酬/100|閾値以上/100|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in negatives[:20]:
        values = "|".join(f"{row[key]:.2f}" for key in ("mean", "sd", "min", "p05", "median", "p95", "max"))
        count = row["episodes_at_or_above_registered_threshold"]
        lines.append(f"|{row['backend']} / {row['condition']} / {row['seed']} / w{row['worker']}|{values}|"
                     f"{row['return_below_zero_count']}|{count if count is not None else '—'}|")
    if not negatives:
        lines.append("|該当なし（pendingは除外）|—|—|—|—|—|—|—|—|—|")
    lines.extend(["", "未完了・未保存runはpending、完了宣言後の欠落/重複/非有限値や平均不一致はinvalidとし、"
                  "不完全な100 episodeを分布へ混ぜない。native/SB3の主90runに加え、Tianshou離散SAC 3run、"
                  "Tianshou Double DQN 6runを別実装名で記載し、主比較へ混ぜてseed数を増やさない。", "",
                  "再生成: `python3 benchmarks/analyze_evaluation_distribution.py`。学習ログは読み取りのみ。", ""])
    if result["errors"]:
        lines.extend(["集計全体のエラー: " + json.dumps(result["errors"], ensure_ascii=False), ""])
    return "\n".join(lines)


def analyze(base, expected_runs=99, expected_models=141):
    errors, fingerprints = [], {}
    threshold_snapshot = read_json(base / "gymnasium_130_thresholds.json", errors, fingerprints)
    thresholds = {row["environment"]: row["registered_reward_threshold"] for row in threshold_snapshot.get("records", [])}
    jobs = discover(base, errors, fingerprints)
    runs, workers = [], []
    for job in jobs:
        run, rows = analyze_run(job, base, thresholds, fingerprints)
        runs.append(run)
        workers.extend(rows)
    if len(runs) != expected_runs:
        errors.append(f"manifest run count {len(runs)} != expected {expected_runs}")
    if len(workers) != expected_models:
        errors.append(f"worker model count {len(workers)} != expected {expected_models}")
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol": {"episodes_per_model": EPISODES, "evaluation_seed_start": SEED_START,
                         "reward": "raw", "standard_deviation": "sample; ddof=1 within this model's 100 evaluation episodes",
                         "quantiles": "Hyndman-Fan type 7; linear interpolation at (n-1)*p",
                         "threshold_episode_counts_are_descriptive_only": True,
                         "new_performance_gate": False, "confidence_intervals_computed": False,
                         "inference_performed": False, "expected_runs": expected_runs, "expected_worker_models": expected_models},
            "runs": runs, "workers": workers, "source_sha256": fingerprints, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE)
    args = parser.parse_args()
    result = analyze(args.base)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation_distribution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    scalar_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items() if key not in ("evaluation_returns", "evaluation_lengths")}
                   for row in result["workers"]]
    with (args.output / "evaluation_distribution.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(scalar_rows[0]) if scalar_rows else ["status"])
        writer.writeheader()
        writer.writerows(scalar_rows)
    (args.output / "EVALUATION_DISTRIBUTION.ja.md").write_text(markdown(result))
    print(json.dumps({"runs": len(result["runs"]), "models": len(result["workers"]),
                      "states": dict(Counter(row["status"] for row in result["workers"])), "errors": result["errors"]}))
    return 1 if result["errors"] or any(row["status"] in ("invalid", "failed") for row in result["workers"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
