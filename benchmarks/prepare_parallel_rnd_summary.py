#!/usr/bin/env python3
"""Prepare read-only parallel/RND statistics without publishing or running inference.

Outputs live only in restart_20260928/parallel_rnd_preparation. The canonical
interpretation and snapshots are deliberately never overwritten by this tool.
After finalization, rerun with --require-final to require the final audit gates.
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"
OUTPUT = BASE / "restart_20260928/parallel_rnd_preparation"
SEEDS = (42, 123, 2026)
T_CRITICAL_DF2 = 4.302652729911275
CONDITIONS = (
    "cartpole_dqn", "cartpole_sac", "cartpole_dqn_parallel4", "cartpole_sac_parallel4",
    "lunar_ppo", "lunar_rnd", "lunar_ppo_parallel2", "lunar_rnd_parallel2", "lunar_rnd_shared2",
    "halfcheetah_hybrid", "halfcheetah_sac", "ant_shared", "ant_rnd_shared", "ant_sac",
)
COMPARISONS = (
    ("cartpole_dqn_parallel4", "cartpole_dqn", "dqn"),
    ("cartpole_sac_parallel4", "cartpole_sac", "sac"),
    ("lunar_ppo_parallel2", "lunar_ppo", "ppo"),
    ("lunar_rnd", "lunar_ppo", "ppo"),
    ("lunar_rnd_parallel2", "lunar_ppo_parallel2", "ppo"),
    ("lunar_rnd_shared2", "lunar_ppo_parallel2", "ppo"),
    ("lunar_rnd_shared2", "lunar_rnd_parallel2", "ppo"),
    ("lunar_rnd_parallel2", "lunar_rnd", "ppo"),
    ("lunar_rnd_shared2", "lunar_rnd", "ppo"),
    ("halfcheetah_hybrid", "halfcheetah_sac", "sac"),
    ("ant_shared", "ant_sac", "sac"),
    ("ant_rnd_shared", "ant_sac", "sac"),
    ("ant_rnd_shared", "ant_shared", "ppo"),
    ("ant_rnd_shared", "ant_shared", "sac"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_summary(values):
    """Only a complete set of three independent training-seed values gets a CI."""
    complete = set(values) == set(SEEDS)
    ordered = [values[seed] for seed in SEEDS if seed in values]
    result = {"complete": complete, "n": len(ordered), "seed_values": values,
              "mean": None, "sample_sd": None, "ci95": None}
    if complete:
        mean, sd = statistics.mean(ordered), statistics.stdev(ordered)
        half = T_CRITICAL_DF2 * sd / math.sqrt(3)
        result.update(mean=mean, sample_sd=sd, ci95=[mean-half, mean+half])
    return result


def aggregate_group(condition, algorithm, expected_runs, complete_runs, threshold):
    values, selected, counters = {}, [], []
    for run in complete_runs:
        workers = [worker for worker in run["workers"] if worker["algorithm"] == algorithm]
        values[run["seed"]] = statistics.mean(worker["final_mean"] for worker in workers)
        selected.extend(workers)
        counters.append({"seed": run["seed"], "aggregate_environment_steps": run["aggregate_steps"],
                         "algorithm_environment_steps": sum(worker["steps"] for worker in workers),
                         "recorded_updates_sum": sum(worker["recorded_updates"] for worker in workers),
                         "policy_optimizer_steps_sum": sum(worker["policy_optimizer_steps"] for worker in workers),
                         "policy_minibatch_sample_presentations_sum": sum(worker["policy_sample_presentations"] for worker in workers)})
    result = {"condition": condition, "algorithm": algorithm, **seed_summary(values),
              "threshold": threshold, "expected_runs": len(expected_runs),
              "expected_worker_models": sum(sum(spec["algorithm"] == algorithm for spec in run["effective_agents"]) for run in expected_runs),
              "observed_worker_models": len(selected), "per_seed_counters": counters,
              "seeds_at_threshold": sum(value >= threshold for value in values.values()) if threshold is not None else None,
              "worker_models_at_threshold": sum(worker["final_mean"] >= threshold for worker in selected) if threshold is not None else None,
              "lowest_worker_mean": min((worker["final_mean"] for worker in selected), default=None)}
    result["classification"] = "pending" if not result["complete"] else (
        "no_registered_threshold" if threshold is None else "all_seeds_reached" if result["seeds_at_threshold"] == 3
        else "mean_only_reached" if result["mean"] >= threshold else "below_threshold")
    return result


def paired_comparison(left, right):
    a, b = left["seed_values"], right["seed_values"]
    differences = {seed: a[seed]-b[seed] for seed in SEEDS if seed in a and seed in b}
    return {"left": left["condition"], "right": right["condition"], "algorithm": left["algorithm"],
            "direction": "left minus right", **seed_summary(differences),
            "same_seed_labels_do_not_imply_identical_rng_streams": True}


def build_snapshot(base=BASE, root=ROOT, require_final=False):
    fingerprints = {}
    def read(path):
        content = path.read_bytes()
        fingerprints[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    manifest = read(base / "native_manifest.json")
    thresholds = {row["environment"]: row["registered_reward_threshold"] for row in read(base / "gymnasium_130_thresholds.json")["records"]}
    jobs = [job for job in manifest["jobs"] if job["condition"] in CONDITIONS]
    assert len(jobs) == 42 and {(job["condition"],job["seed"]) for job in jobs} == {(case, seed) for case in CONDITIONS for seed in SEEDS}
    runs = []
    for job in jobs:
        path = root / job["output"]
        configuration = read(base / "configs" / f"{job['case']}.json")
        specs = configuration["agents"]
        command = job["command"]
        if "--workers" in command:
            assert len(specs) == 1
            specs = specs * int(command[command.index("--workers") + 1])
        run = {"id": job["id"], "condition": job["condition"], "seed": job["seed"], "steps": job["steps"],
               "output": job["output"], "status": "pending", "effective_agents": specs,
               "configuration": configuration, "workers": [], "test": [], "aggregate_steps": None,
               "share_rnd": "--share-rnd" in command, "threshold": thresholds[configuration["env_id"]]}
        metadata_path, final_path = path / "metadata.json", path / "final.json"
        if metadata_path.exists():
            metadata = read(metadata_path)
            assert metadata["effective_agents"] == specs
            run["worker_seeds"] = metadata["worker_seeds"]
            run["library_sha256"] = metadata["library_sha256"]
        if (path / "failure.json").exists():
            raise ValueError(f"Training failure exists: {job['id']}")
        if not final_path.exists():
            runs.append(run)
            continue
        final = read(final_path)
        assert metadata_path.exists() and final["status"] == "complete" and final["actual_total_steps"] == job["steps"]
        assert len(final["workers"]) == len(specs) == len(final["test"])
        recorded = {row["worker"]: row for row in final["workers"]}
        tests = {row["worker"]: row for row in final["test"]}
        assert set(recorded) == set(tests) == set(range(len(specs)))
        assert sum(row["steps"] for row in recorded.values()) == job["steps"]
        workers = []
        for worker_id, spec in enumerate(specs):
            original, test = recorded[worker_id], tests[worker_id]
            assert original["algorithm"] == test["algorithm"] == spec["algorithm"]
            assert original["steps"] * len(specs) == job["steps"]
            assert test["episodes"] == 100 and test["seed_start"] == 900000 and test["split"] == "test" and test["reward"] == "raw" and test["deterministic"] is True
            assert len(test["returns"]) == len(test["lengths"]) == 100 and all(math.isfinite(value) for value in test["returns"])
            mean = statistics.mean(test["returns"])
            assert math.isclose(mean, test["mean"], rel_tol=1e-12, abs_tol=1e-10)
            config, stats, algorithm = spec["config"], original["statistics"], spec["algorithm"]
            count = stats["n_updates" if algorithm == "sac" else "updates"]
            assert count >= 0 and int(count) == count
            batch = config["minibatch_size"] if algorithm == "ppo" else config["batch_size"]
            policy_steps = int(count)
            if algorithm == "ppo":
                assert original["steps"] == int(count) * config["update_interval"]
                assert config["update_interval"] % batch == 0
                policy_steps *= config["epochs"] * (config["update_interval"] // batch)
            rnd = spec.get("rnd_config")
            worker = {**original, "final_mean": mean, "final_min": min(test["returns"]), "final_max": max(test["returns"]),
                      "training_seed": job["seed"], "condition": job["condition"], "seed_mapping": metadata["worker_seeds"][worker_id],
                      "recorded_updates": int(count), "update_unit": "PPO rollout" if algorithm == "ppo" else "SAC minibatch cycle" if algorithm == "sac" else "DQN optimizer step",
                      "policy_optimizer_steps": policy_steps, "policy_sample_presentations": policy_steps * batch,
                      "rnd_presentations_estimate": int(count)*config["update_interval"] if rnd else None,
                      "rnd_optimizer_steps_estimate": int(count)*math.ceil(config["update_interval"]/rnd["update_interval"]) if rnd else None,
                      "rnd_module_owner": (0 if run["share_rnd"] else worker_id) if rnd else None,
                      "final_intrinsic_coefficient_weighted": stats["intrinsic_reward_mean"]*stats["curiosity_coefficient"] if rnd else None,
                      "replay_size_is_not_cumulative_or_additive": True}
            workers.append(worker)
        run.update(status="complete", workers=workers, test=final["test"], aggregate_steps=final["actual_total_steps"])
        runs.append(run)

    groups = []
    for condition in CONDITIONS:
        expected_runs = [run for run in runs if run["condition"] == condition]
        done = [run for run in expected_runs if run["status"] == "complete"]
        for algorithm in dict.fromkeys(spec["algorithm"] for spec in expected_runs[0]["effective_agents"]):
            groups.append(aggregate_group(condition, algorithm, expected_runs, done, expected_runs[0]["threshold"]))
    by_group = {(group["condition"], group["algorithm"]): group for group in groups}
    paired = [paired_comparison(by_group[left,algorithm],by_group[right,algorithm]) for left,right,algorithm in COMPARISONS]
    worker_groups = []
    for condition in CONDITIONS:
        expected_runs = [run for run in runs if run["condition"] == condition]
        for worker_id, spec in enumerate(expected_runs[0]["effective_agents"]):
            values = {run["seed"]: run["workers"][worker_id]["final_mean"] for run in expected_runs if run["status"] == "complete"}
            worker_groups.append({"condition": condition, "worker": worker_id, "algorithm": spec["algorithm"], **seed_summary(values)})

    audit_path = base / "rnd_posthoc_audit.json"
    audit = read(audit_path) if audit_path.exists() else None
    rnd_final = bool(audit and audit.get("passed") is True and audit.get("complete") is True and audit.get("require_complete") is True
                     and audit.get("completed_runs_audited") == 12 and audit.get("modules_audited") == 15
                     and audit.get("targets_unchanged") == 15 and audit.get("predictors_changed") == 15
                     and not any(audit.get(key) for key in ("pending_runs", "failed_runs", "errors", "unexpected_runs")))
    finalization_path = base / "finalization.json"
    finalization = read(finalization_path) if finalization_path.exists() else None
    finalized = bool(finalization and finalization.get("status") == "passed")
    complete = all(run["status"] == "complete" for run in runs)
    if require_final and not (complete and rnd_final and finalized):
        raise ValueError("Final preparation requires all 42 runs plus strict RND audit and passed campaign finalization")
    checkpoint = {"source": str(audit_path), "source_sha256": fingerprints.get(str(audit_path.resolve())),
                  "authoritative_final_audit_ready": rnd_final,
                  "warning": None if rnd_final else "Stored RND audit is missing or stale/incomplete; completed training alone does not verify target/predictor contracts.",
                  "audit": audit}
    snapshot = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot_not_final": not (complete and rnd_final and finalized),
                "status": "final_ready" if complete and rnd_final and finalized else "preparation_only",
                "scope": "14 conditions, 42 training runs, 84 individual models; no new training/inference/audit",
                "expected_runs": 42, "completed_runs": sum(run["status"] == "complete" for run in runs),
                "expected_worker_models": 84, "completed_worker_models": sum(len(run["workers"]) for run in runs),
                "pending_runs": [run["id"] for run in runs if run["status"] != "complete"],
                "strict_rnd_audit_ready": rnd_final, "campaign_finalization_passed": finalized,
                "runs": runs, "groups": groups, "worker_groups": worker_groups, "paired_comparisons": paired,
                "statistics": {"seed_labels": SEEDS, "t_critical_df2": T_CRITICAL_DF2,
                               "unit": "equal-weight worker means within one algorithm/run, then three training-seed means",
                               "partial_seed_sets": "preserved individually, no final mean/SD/CI computed",
                               "sample_presentations": "includes repeated replay draws / PPO epoch reuse, not unique transitions",
                               "sac_optimizer_units": "policy_optimizer_steps counts actor steps; each SAC cycle also updates both critics and temperature",
                               "rnd_counts": "inferred from rollout count and batch configuration, not an instrumented RND counter"},
                "input_sha256": fingerprints}
    for path, expected_hash in fingerprints.items():
        assert sha256(Path(path)) == expected_hash, f"Input changed during collection: {path}"
    return snapshot, checkpoint


def number(value):
    return "pending" if value is None else f"{value:,.3f}"


def interval(values):
    return "pending" if values is None else f"[{number(values[0])}, {number(values[1])}]"


def render_preparation(snapshot, checkpoint):
    lines = ["# 並列・RND 最終更新の準備データ", "", f"観測時刻: {snapshot['observed_at_utc']}。正式レポートは未差し替え。学習・推論・RND audit を起動せず保存ログだけを読む。", "",
             f"対象42 runのうち {snapshot['completed_runs']} 完了、全84モデルのうち {snapshot['completed_worker_models']} の最終値を保持。pending: {', '.join(snapshot['pending_runs']) or 'なし'}。", "",
             f"既存RND監査の最終根拠利用可: {snapshot['strict_rnd_audit_ready']}。campaign finalization passed: {snapshot['campaign_finalization_passed']}。", "",
             "3 seed が揃わない条件は個別値のみ保存し、最終平均や区間を出さない。以下は同じrun・同じalgorithm内のworker等重み平均→学習seed3値を単位とする。達成worker数は個別方策の未達を隠さない。", "",
             "|条件|algorithm|seed42|seed123|seed2026|3seed mean ± SD|95%t CI|seed達成|worker達成/対象|最低worker平均|", "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for group in snapshot["groups"]:
        values = group["seed_values"]
        seed_count = "N/A" if group["threshold"] is None else f"{group['seeds_at_threshold']}/{group['n']}"
        worker_count = "N/A" if group["threshold"] is None else f"{group['worker_models_at_threshold']}/{group['observed_worker_models']}"
        if not group["complete"]:
            seed_count += "（予定3）"
            worker_count += f"（予定{group['expected_worker_models']}）"
        lines.append(f"|{group['condition']}|{group['algorithm']}|"+"|".join(number(values.get(seed)) for seed in SEEDS)+f"|{number(group['mean'])} ± {number(group['sample_sd'])}|{interval(group['ci95'])}|{seed_count}|{worker_count}|{number(group['lowest_worker_mean'])}|")
    lines += ["", "## 同seed対応差（左−右）", "", "異なるworker数では同じseedラベルでも同じモデル群・RNG列とは限らない。独立した学習seedは3本。差と区間は記述的で、多重比較補正・同等性検定はしていない。", "",
              "|左−右|algorithm|差42|差123|差2026|平均差|95%t CI|", "|---|---|---:|---:|---:|---:|---|"]
    for pair in snapshot["paired_comparisons"]:
        lines.append(f"|{pair['left']} − {pair['right']}|{pair['algorithm']}|"+"|".join(number(pair['seed_values'].get(seed)) for seed in SEEDS)+f"|{number(pair['mean'])}|{interval(pair['ci95'])}|")
    lines += ["", "## 全worker個別の最終値・更新量", "", "PPOの更新はrollout数、DQNはoptimizer step、SACはactor+2critic+temperatureを含むminibatch cycle。下表のoptimizer列はpolicy/Q optimizer回数で、SACの全optimizer合計ではない。sample列は重複を含む投入回数でunique経験数ではない。", "",
              "|条件|seed|worker/algorithm|local step|記録更新数|policy/Q optimizer|sample投入回数|final raw mean|", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for run in snapshot["runs"]:
        for worker in run["workers"]:
            lines.append(f"|{run['condition']}|{run['seed']}|{worker['worker']}/{worker['algorithm']}|{worker['steps']:,}|{worker['recorded_updates']:,}|{worker['policy_optimizer_steps']:,}|{worker['policy_sample_presentations']:,}|{number(worker['final_mean'])}|")
    lines += ["", "## RNDの最終rollout統計", "", "全期間平均ではなく、最後のrolloutのpredictor更新前平均。worker間で時点が同じとは限らない。単調減少やraw報酬への有効性の合格判定には用いない。RNDのstep列は実装と設定からの推定、計測counterではない。", "",
              "|条件|seed|worker/module owner|intrinsic mean|coefficient後|RND観測投入推定|RND optimizer推定|", "|---|---:|---|---:|---:|---:|---:|"]
    for run in snapshot["runs"]:
        for worker in run["workers"]:
            if worker["rnd_module_owner"] is not None:
                lines.append(f"|{run['condition']}|{run['seed']}|{worker['worker']}/{worker['rnd_module_owner']}|{worker['statistics']['intrinsic_reward_mean']:.8g}|{worker['final_intrinsic_coefficient_weighted']:.8g}|{worker['rnd_presentations_estimate']:,}|{worker['rnd_optimizer_steps_estimate']:,}|")
    lines += ["", "## 最終更新で維持する解釈", "",
              "- 合計step固定は各learnerの更新数・総計算量の固定ではない。全workerのpolicyとoptimizerは独立し、最良worker選抜やensemble評価はしていない。",
              "- HalfCheetahは2PPO+2SACに各T/4。PPOはreplayへexportのみ、SACだけが読む。PPOとSACの成績・更新数を混ぜず、SAC1体の予算は単独の1/4。",
              "- AntのsharedはPPO/SAC共有replay。RND付きPPOは1体であり複数PPOの共有RND条件ではない。SACに渡る報酬はextrinsicのみ。SAC更新機会上限はmix interval2 / solo interval4で近いがtarget更新のlocal cadenceなどが異なる。",
              "- Lunar別RND2個対共有RND1個ではtarget数・random mapping・1moduleの観測数と到着順も変わる。target不変/predictor変化という契約と性能改善を分ける。",
              "- CartPole4workerは各T/4、epsilon/target時計はlocal。共有bufferの最終サイズをworkerごとに足さず、sample投入量をunique経験数と呼ばない。",
              "- 共有のみを除いた同worker/同更新対照や経験source別抽出ログがないため、純粋な共有効果・速度優位は分離できない。",
              "- policy seedは同algorithm ordinalごとbase+10000*i。RND有無で初期policyを合わせてもRust RNG/shuffle/thread scheduleは完全一致しない。",
              "", "## target/predictorの旧snapshot生成方法と更新元", "",
              "旧 parallel_rnd_checkpoint_snapshot.json は audit_rnd.py の初期版出力と同じschemaで、seed42 Lunar RND 1moduleを含む。保存済みseed/configと同一libraryから一時ディレクトリに初期RNDを再構成して保存し、ZIP内のtensor payload群をsortして連結したSHA-256、tensor個数、byte数を最終checkpointと比較したもの。serialization時刻は比較しない。今回この監査は再起動しない。", "",
              "最終更新では finalizer が生成する rnd_posthoc_audit.json を確定根拠として取り込む。require_complete=true / passed=true / complete=true、12run・15module、target不変15・predictor変化15、pending/errorsなしを要求する。現状の保存済み監査が古ければ、学習完了からtarget固定を推定しない。", "",
              f"現監査 SHA-256: `{checkpoint['source_sha256']}`。最終根拠利用可: {checkpoint['authoritative_final_audit_ready']}。", "",
              "準備JSONには入力SHA-256、各workerのseed対応・設定・最終100episode配列も保存した。読取終了時に入力hash不変を再確認済み。正式ファイルの変更は親から最終監査完了の追伸を受けてから行う。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args(argv)
    snapshot, checkpoint = build_snapshot(require_final=args.require_final)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("parallel_rnd_snapshot.prepared.json", snapshot), ("parallel_rnd_checkpoint_snapshot.prepared.json", checkpoint)):
        (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "parallel_rnd_preparation.ja.md").write_text(render_preparation(snapshot, checkpoint))
    print(json.dumps({key: snapshot[key] for key in ("status", "completed_runs", "completed_worker_models", "pending_runs", "strict_rnd_audit_ready", "campaign_finalization_passed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
