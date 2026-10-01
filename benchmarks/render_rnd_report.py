#!/usr/bin/env python3
"""Render recorded RND audit evidence without opening models or running learning.

Requires matplotlib only for PNG/PDF output. Run audit_rnd.py first to refresh
the input; this renderer never modifies the audit, checkpoints, or training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identity(run):
    path = Path(run)
    return path.parent.name, int(path.name.removeprefix("seed_"))


def weighted(raw, coefficient):
    return raw * coefficient if finite(raw) and finite(coefficient) else None


def segments(points, field):
    """Break lines at absent/invalid checkpoints; never bridge a missing value."""
    result, current, previous = [], [], None
    for point in sorted(points, key=lambda p: p["checkpoint"]):
        valid = finite(point.get("aggregate_steps")) and finite(point.get(field))
        if not valid or (previous is not None and point["checkpoint"] != previous + 1):
            if current:
                result.append(current)
                current = []
        if valid:
            current.append(point)
        previous = point["checkpoint"]
    if current:
        result.append(current)
    return result


def prepare(audit):
    grouped = defaultdict(list)
    traces, finals = [], []
    for run in audit.get("results", []):
        condition, seed = identity(run["run"])
        grouped[condition].append(run)
        for trace in run.get("worker_traces", []):
            seen, points = set(), []
            for source in trace.get("points", []):
                checkpoint = source["checkpoint"]
                if checkpoint in seen:
                    raise ValueError(f"duplicate checkpoint in {run['run']} worker {trace['worker']}: {checkpoint}")
                seen.add(checkpoint)
                raw, coefficient = source.get("last_rollout_intrinsic_mean"), source.get("coefficient")
                points.append({**source, "intrinsic_reward_mean": raw,
                               "weighted_intrinsic_reward_mean": weighted(raw, coefficient)})
            points.sort(key=lambda p: p["checkpoint"])
            traces.append({"condition": condition, "seed": seed, "worker": trace["worker"],
                           "run": run["run"], "missing_early_checkpoints": trace.get("missing_early_checkpoints", True),
                           "points": points})
            stats = trace.get("final_statistics", {})
            raw, coefficient = stats.get("intrinsic_reward_mean"), stats.get("curiosity_coefficient")
            finals.append({"condition": condition, "seed": seed, "worker": trace["worker"],
                           "aggregate_steps": run.get("actual_total_steps"), "updates": stats.get("updates"),
                           "intrinsic_reward_mean": raw, "coefficient": coefficient,
                           "weighted_intrinsic_reward_mean": weighted(raw, coefficient),
                           "recorded_points": len(points),
                           "missing_early_checkpoints": trace.get("missing_early_checkpoints", True)})
    groups = []
    for condition, runs in sorted(grouped.items()):
        modules = [module for run in runs for module in run.get("modules", [])]
        failed = [run for run in runs if run.get("errors") or run["status"] not in ("audited", "not_started", "in_progress")]
        pending = [run for run in runs if run["status"] in ("not_started", "in_progress")]
        expected = [run.get("expected_modules") for run in runs]
        expected_modules = sum(expected) if all(value is not None for value in expected) else None
        audited = sum(run["status"] == "audited" for run in runs)
        module_failures = sum(not module.get("target_unchanged") or not module.get("predictor_changed") for module in modules)
        groups.append({"condition": condition, "runs_in_audit": len(runs), "audited_runs": audited,
                       "expected_modules": expected_modules, "audited_modules": len(modules),
                       "targets_unchanged": sum(bool(m.get("target_unchanged")) for m in modules),
                       "predictors_changed": sum(bool(m.get("predictor_changed")) for m in modules),
                       "pending_runs": [run["run"] for run in pending],
                       "failed_runs": [run["run"] for run in failed], "failed_modules": module_failures,
                       "status": "FAILED" if failed or module_failures else "PENDING" if pending else "AUDITED",
                       "module_owners": [{"run": run["run"], "owner_worker": module["owner_worker"],
                                          "shared_by_workers": module["shared_by_workers"]}
                                         for run in runs for module in run.get("modules", [])]})
    return groups, sorted(traces, key=lambda x: (x["condition"], x["seed"], x["worker"])), sorted(
        finals, key=lambda x: (x["condition"], x["seed"], x["worker"]))


def save_figures(output, groups, traces):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 160})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    seeds = sorted({trace["seed"] for trace in traces})
    colors = {seed: plt.get_cmap("tab10")(index % 10) for index, seed in enumerate(seeds)}
    generated = {}
    for group in groups:
        condition = group["condition"]
        selected = [trace for trace in traces if trace["condition"] == condition]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
        for axis, field, title in zip(axes, ("intrinsic_reward_mean", "weighted_intrinsic_reward_mean"),
                                      ("Raw intrinsic reward", "Coefficient-weighted intrinsic reward")):
            any_points = False
            for trace in selected:
                pieces = segments(trace["points"], field)
                for index, piece in enumerate(pieces):
                    any_points = True
                    axis.plot([point["aggregate_steps"] for point in piece], [point[field] for point in piece],
                              color=colors[trace["seed"]], linestyle=("-", "--", ":", "-.")[trace["worker"] % 4],
                              marker=("o", "s", "^", "D")[trace["worker"] % 4], markersize=4,
                              linewidth=1.3, label=f"seed {trace['seed']} / worker {trace['worker']}" if index == 0 else None)
            if not any_points:
                axis.text(.5, .5, "No recorded curiosity points\nPending or missing history",
                          transform=axis.transAxes, ha="center", va="center", color="#666666")
            else:
                axis.legend(fontsize=8, loc="best")
                axis.set_xlim(left=0)
            axis.set(xlabel="Aggregate environment steps", ylabel="Last-rollout mean", title=title)
            axis.grid(alpha=.18)
            axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useOffset=False)
            if not any_points:
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(f"{condition} | checkpoint audit: {group['status']}\n"
                     f"Audited runs {group['audited_runs']}/{group['runs_in_audit']} | each seed / worker separately")
        fig.supxlabel("Before predictor update; not a run-wide average. Missing checkpoints are not connected.", fontsize=8)
        stem = "rnd__" + condition
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"{stem}.{extension}")
        plt.close(fig)
        generated[condition] = {ext: str((figures / f"{stem}.{ext}").resolve()) for ext in ("png", "pdf")}
    return generated


def fmt(value):
    return f"{value:.7g}" if finite(value) else "—"


def write_markdown(output, audit_path, audit, summary):
    status = "COMPLETE" if audit.get("complete") else "PENDING / INCOMPLETE"
    if audit.get("failed_runs") or (audit.get("passed") is False and not audit.get("pending_runs")):
        status = "FAILED"
    lines = ["# RNDの状態更新と好奇心報酬", "", f"生成時刻: {summary['generated_at_utc']}。監査全体: **{status}**。",
             "", f"入力は[監査JSON]({audit_path.resolve()})（SHA256 `{summary['audit_sha256']}`）。本書は既存artifactだけを読み、学習・checkpoint・監査JSONを変更しない。",
             "", f"期待 {audit.get('expected_runs', '未記録')} run / {audit.get('expected_modules', '未記録')} module、"
             f"監査済み {audit.get('completed_runs_audited', 0)} run / {audit.get('modules_audited', 0)} module。"
             f"pending {len(audit.get('pending_runs', []))} run、失敗run {len(audit.get('failed_runs', []))}。",
             "", "## module単位のcheckpoint検証", "",
             "target不変は固定targetの契約、predictor変化は保存された学習状態の変化を示す。いずれもraw rewardの改善や探索の有効性を証明するものではない。共有RNDをworker数の分だけ重複計上しない。",
             "", "| 条件 | 状態 | 監査run / 列挙run | 監査module / 期待module | target不変 | predictor変化 | 失敗module |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        expected = group["expected_modules"] if group["expected_modules"] is not None else "未記録"
        lines.append(f"| {group['condition']} | {group['status']} | {group['audited_runs']} / {group['runs_in_audit']} | "
                     f"{group['audited_modules']} / {expected} | {group['targets_unchanged']} | {group['predictors_changed']} | {group['failed_modules']} |")
    lines += ["", "共有moduleのowner / 利用worker:", ""]
    for group in summary["groups"]:
        for module in group["module_owners"]:
            lines.append(f"- `{module['run']}`: owner={module['owner_worker']}、利用worker={module['shared_by_workers']}")
    lines += ["", "## 条件別の好奇心報酬", "",
              "横軸は全worker合計の環境step。左は`intrinsic_reward_mean`の生値、右は各記録時点の`coefficient`を掛けた値。seedとworkerを別線にし、平均化・平滑化・外挿はしない。欠けたcheckpointの前後を線で接続せず、初期欠測を0で埋めない。1点だけの系列は点だけを描く。",
              "", "初期checkpointが欠測の場合、初期noveltyやその減少率は読み取れない。曲線は**直近の完了rolloutで、predictorを更新する前に計算した平均**であり、学習全期間の平均・全環境step平均・predictor更新後の誤差ではない。新しい状態の訪問によって増えることもあるので、単調減少を合格基準にしない。共有RNDでは他workerの先行更新と到着順にも依存する。"]
    for group in summary["groups"]:
        figure = summary["figures"][group["condition"]]
        lines += ["", f"### {group['condition']}", "", f"![{group['condition']} curiosity]({figure['png']})",
                  "", f"[PNG]({figure['png']}) / [PDF]({figure['pdf']})", ""]
        selected = [trace for trace in summary["traces"] if trace["condition"] == group["condition"]]
        for trace in selected:
            points = trace["points"]
            checkpoints = [point["checkpoint"] for point in points]
            missing = sorted(set(range(1, max(checkpoints, default=0) + 1)) - set(checkpoints))
            lines.append(f"- seed{trace['seed']} / worker{trace['worker']}: 記録{len(points)}点、checkpoint={checkpoints}、"
                         f"先頭欠測={'あり' if trace['missing_early_checkpoints'] else 'なし'}、最後の記録までの欠測={missing}。")
        for run in group["pending_runs"]:
            lines.append(f"- `{run}`: pending。完成runの最終値として扱わない。")
    lines += ["", "## 最終curiosity統計（worker別）", "",
              "各完了runの`final_statistics`を表示する。plot最後のmonitor記録と最終統計の時点は必ずしも一致しないため、欠測した最終点を表の値から曲線へ補充しない。",
              "", "| 条件 | seed | worker | 合計step | PPO rollout更新 | intrinsic生値 | coefficient | 係数適用後 | 記録点数 | 初期欠測 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for final in summary["final_statistics"]:
        lines.append(f"| {final['condition']} | {final['seed']} | {final['worker']} | {fmt(final['aggregate_steps'])} | "
                     f"{fmt(final['updates'])} | {fmt(final['intrinsic_reward_mean'])} | {fmt(final['coefficient'])} | "
                     f"{fmt(final['weighted_intrinsic_reward_mean'])} | {final['recorded_points']} | "
                     f"{'あり' if final['missing_early_checkpoints'] else 'なし'} |")
    if not summary["final_statistics"]:
        lines += ["", "完了runの最終curiosity統計はまだない。"]
    lines += ["", "## 再実行", "", "本番終了後、監査用native環境で`audit_rnd.py --require-complete`を実行してから、matplotlibのあるPythonで次を実行する。",
              "", "```sh", f"python3 benchmarks/render_rnd_report.py --audit {audit_path} --output {output}", "```", "",
              "生の抽出点・条件別件数・最終統計は同じ出力先の`summary.json`に保存する。RNDの有効性は別途、同条件RND有無の全worker最終評価と3つのtraining seedの差で判断する。"]
    if audit.get("errors"):
        lines += ["", "監査側の通知:", "", *[f"- {error}" for error in audit["errors"]]]
    (output / "RND_REPORT.ja.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/oss_benchmarks/rnd_posthoc_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/oss_benchmarks/rnd_visualization")
    args = parser.parse_args()
    audit_bytes = args.audit.read_bytes()
    audit = json.loads(audit_bytes)
    groups, traces, finals = prepare(audit)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(), "groups": groups,
               "traces": traces, "final_statistics": finals,
               "figures": save_figures(args.output, groups, traces)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_markdown(args.output, args.audit, audit, summary)
    print(f"Rendered {len(groups)} conditions as PNG/PDF; {len(finals)} final worker statistics. "
          f"Audit complete={audit.get('complete', False)}. Output: {args.output}")


if __name__ == "__main__":
    main()
