#!/usr/bin/env python3
"""Re-read native, SB3 and supplementary Double DQN results and render comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/reinforcex-tianshou-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"
CASES = {"cartpole_dqn": ("CartPole-v1", 204800, 475), "lunar_dqn": ("LunarLander-v3", 1024000, 200)}
SEEDS = (42, 123, 2026)
LABELS = {"native": "ReinforceX Double DQN", "sb3": "SB3 vanilla DQN", "tianshou_dqn": "Tianshou Double DQN"}
COLORS = {"native": "#1464a5", "sb3": "#687078", "tianshou_dqn": "#df752e"}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rolling(values, window=20):
    return [statistics.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def collect():
    records, series, source_hashes = [], [], {}
    for case in CASES:
        for backend in LABELS:
            for seed in SEEDS:
                root = BASE / "runs" / backend / case / f"seed_{seed}"
                if not (root / "final.json").exists():
                    records.append({"case": case, "backend": backend, "seed": seed, "status": "pending"})
                    continue
                final = read(root / "final.json")
                evaluation = rows(root / "evaluations.jsonl")
                train_path = root / ("train_worker0.jsonl" if backend == "native" else "train_episodes.jsonl")
                train = rows(train_path)
                files = [root / "final.json", root / "metadata.json", root / "evaluations.jsonl", train_path]
                if backend == "native":
                    train = [r for r in train if not r["budget_cut"]]
                    raw = [r["reward"] for r in train]
                    episode_steps = [r["steps"] for r in train]
                    test_returns = final["test"][0]["returns"]
                    steps, seconds = final["actual_total_steps"], final["training_seconds"]
                    updates = final["workers"][0]["statistics"]["updates"]
                    validation = [{"steps": p["aggregate_steps"], "mean_return": p["mean"]}
                                  for p in evaluation if p["split"] == "validation"]
                else:
                    raw = [r["return"] for r in train]
                    episode_steps = [r["env_steps"] for r in train]
                    eval_path = root / "eval_episodes.jsonl"
                    test_returns = [r["return"] for r in rows(eval_path) if r["phase"] == "final"]
                    files.append(eval_path)
                    steps, seconds = final["actual_steps"], final["training_seconds_excluding_evaluation"]
                    updates = final["updates"]
                    validation = [{"steps": p["env_steps"], "mean_return": p["mean_return"]}
                                  for p in evaluation if p["phase"] != "final"]
                for path in files:
                    source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                final_mean = statistics.mean(test_returns)
                records.append({"case": case, "backend": backend, "seed": seed, "status": "complete",
                                "actual_steps": steps, "completed_episodes": len(train), "updates": updates,
                                "last100_training_mean": statistics.mean(raw[-100:]),
                                "heldout_episodes": len(test_returns), "final_mean": final_mean,
                                "final_sd": statistics.stdev(test_returns), "threshold": CASES[case][2],
                                "passed": final_mean >= CASES[case][2], "training_seconds": seconds})
                series.append({"case": case, "backend": backend, "seed": seed,
                               "episode_returns": raw, "episode_steps": episode_steps,
                               "validation": validation, "heldout_returns": test_returns})
    return records, series, source_hashes


def plot(series, output, kind):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey="row")
    for row, (case, (env_id, budget, threshold)) in enumerate(CASES.items()):
        relevant = [s for s in series if s["case"] == case]
        y_values = ([v for s in relevant for v in s["episode_returns"]] if kind == "episode" else
                    [p["mean_return"] for s in relevant for p in s["validation"]])
        if case == "cartpole_dqn":
            limits = (0, 515)
        else:
            lower, upper = min([0, *y_values]), max([300, *y_values])
            pad = max(20, (upper-lower)*.04)
            limits = (lower-pad, upper+pad)
        for ax, seed in zip(axes[row], SEEDS):
            for entry in relevant:
                if entry["seed"] != seed:
                    continue
                backend = entry["backend"]
                style = "--" if backend == "sb3" else "-"
                if kind == "episode":
                    raw = entry["episode_returns"]
                    x = range(1, len(raw)+1)
                    ax.plot(x, raw, color=COLORS[backend], linewidth=.5, alpha=.11)
                    ax.plot(x, rolling(raw), color=COLORS[backend], linewidth=1.6,
                            linestyle=style, label=LABELS[backend])
                else:
                    points = entry["validation"]
                    ax.plot([p["steps"] for p in points], [p["mean_return"] for p in points],
                            color=COLORS[backend], linestyle=style, marker="o", markersize=3,
                            label=LABELS[backend])
            if kind == "validation":
                ax.axhline(threshold, color="#999999", linestyle=":", linewidth=.8)
                ax.set_xlim(0, budget)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set(title=f"{env_id} / seed {seed}", ylim=limits,
                   xlabel="Completed training episode" if kind == "episode" else "Training environment transitions")
            ax.grid(alpha=.2)
            if any(e["seed"] == seed for e in relevant):
                ax.legend(loc="lower right", fontsize=7)
            missing = [LABELS[b] for b in LABELS if not any(s["seed"] == seed and s["backend"] == b for s in relevant)]
            if missing:
                ax.text(.02, .96, "Pending: " + ", ".join(missing), transform=ax.transAxes,
                        va="top", fontsize=6.5, wrap=True)
        axes[row, 0].set_ylabel("Raw episode return" if kind == "episode" else "Mean raw deterministic evaluation return")
    title = ("Raw training returns and trailing 20-episode mean" if kind == "episode" else
             "Fixed 10-episode validation: seeds 800000-800009")
    fig.suptitle("DQN / Double DQN controls: " + title, fontsize=14)
    caption = ("Episode counts differ at equal transition budgets. Partial final episodes excluded. "
               if kind == "episode" else "Validation does not select a checkpoint. Final tables use 100 held-out episodes per model. ")
    caption += "Same environment/task settings; implementation differences remain."
    fig.text(.5, .012, caption, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .965))
    name = "episode_vs_reward" if kind == "episode" else "steps_vs_evaluation"
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "double_dqn_reference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, series, hashes = collect()
    aggregate = []
    for case in CASES:
        for backend in LABELS:
            complete = [r for r in records if r["case"] == case and r["backend"] == backend and r["status"] == "complete"]
            values = [r["final_mean"] for r in complete]
            aggregate.append({"case": case, "backend": backend, "completed_seeds": len(values),
                              "required_seeds": 3, "passing_seeds": sum(r["passed"] for r in complete),
                              "mean_over_training_seeds": statistics.mean(values) if values else None,
                              "sample_sd_over_training_seeds": statistics.stdev(values) if len(values) > 1 else None})
    complete_count = sum(r["status"] == "complete" for r in records)
    data = {"status": "complete" if complete_count == 18 else "incomplete", "runs": records,
            "aggregate": aggregate, "series": series, "source_sha256": hashes}
    (args.output / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    plot(series, args.output, "episode")
    plot(series, args.output, "validation")
    lines = ["# Double DQN：追加外部対照", "",
             "ReinforceX と同じ Double DQN / Huber 損失を指定できる Tianshou 2.0.1 を追加し、",
             "既存の SB3 vanilla DQN 対照と合わせて比較します。主実験の設定・結果は変更していません。", "",
             f'集計済み **{complete_count}/18 model**。' + ("全モデル完了。" if complete_count == 18 else
              "未完了モデルは保留として表示しています。全 seed が揃うまでは総合判定を行いません。"), "",
             "各実装の学習 seed は 42 / 123 / 2026。CartPole は各 204,800 transitions、",
             "LunarLander は各 1,024,000 transitions。最終評価は各モデル 100 episode、",
             "全モデル共通の held-out seed 900000–900099 を使う greedy 方策です。", "",
             "![Episode vs reward](episode_vs_reward.png)", "",
             "薄線は全完了 episode の生報酬、太線は直近最大 20 episode 平均です。",
             "予算終了時の未完了 episode は除外。方策によって episode 長が変わるため、",
             "サンプル効率は同じ環境ステップ軸の次図で確認します。", "",
             "![Steps vs evaluation](steps_vs_evaluation.png)", "",
             "中間評価は固定 seed 800000–800009 の 10 episode です。最高点でモデルを選ぶことはせず、",
             "下表の最終 100 episode の平均だけで判定します。", "",
             "SB3 の中間評価 callback は当該 step の勾配更新前、native / Tianshou は更新後です。",
             "したがって、同じ step の中間点でも更新の位相は厳密には一致しません。", "",
             "| 環境 | 実装 | seed | 最終生報酬 mean ± SD | 基準達成 | 完了episode | 更新回数 |",
             "|---|---|---:|---:|:---:|---:|---:|"]
    for r in records:
        prefix = f'| {CASES[r["case"]][0]} | {LABELS[r["backend"]]} | {r["seed"]} |'
        lines.append(prefix + (f' {r["final_mean"]:.2f} ± {r["final_sd"]:.2f} | {"○" if r["passed"] else "×"} | '
                               f'{r["completed_episodes"]} | {r["updates"]:.0f} |' if r["status"] == "complete" else
                               " 未完了 | 保留 | — | — |"))
    lines += ["", "基準は CartPole 475、LunarLander 200。上表の SD は同一モデルの評価 episode 間です。",
              "SD は評価 episode 間・学習 seed 間とも標本標準偏差（ddof=1）を用います。", "",
              "| 環境 | 実装 | 完了seed | seed平均の平均 | seed間 SD | 達成seed |",
              "|---|---|---:|---:|---:|---:|"]
    for row in aggregate:
        done = row["completed_seeds"] == 3
        lines.append(f'| {CASES[row["case"]][0]} | {LABELS[row["backend"]]} | {row["completed_seeds"]}/3 | '
                     + (f'{row["mean_over_training_seeds"]:.2f} | {row["sample_sd_over_training_seeds"]:.2f} | '
                        f'{row["passing_seeds"]}/3 |' if done else "保留 | 保留 | 保留 |"))
    lines += ["", "Tianshou は native と Double DQN、Huber δ1、Torch 2.7、ReLU の層幅・深さ、",
              "学習率、n-step、replay、epsilon と訓練報酬を揃えています。SB3 は vanilla DQN / Torch 2.14 です。", "",
              "完全一致しない主要点はターゲット更新周期と順序、勾配クリッピングです。",
              "Tianshou の公開 API は勾配更新単位なので、CartPole は 63×4=252 step（native250、+0.8%）、",
              "LunarLander は 6×8=48 step（native50、−4%）に事前固定しました。",
              "初回ターゲット更新の位相と更新順序も異なります。native の勾配norm10クリップに対し、",
              "公式 Tianshou DQN はクリップなしです。公式アルゴリズムや private method は変更していません。", "",
              "初回学習を 68/72 step に揃えたため、この予算の更新回数は各 51,184 / 127,992 です。",
              "replay の末尾、乱数系列、初期化、行動選択と遷移登録の順序には差が残ります。",
              "3 seed の比較であり、差が出ても単一の実装要因による性能差だとは断定できません。",
              "時間は他の CPU 実験との同時実行の影響があるため、速度比較に使用しません。", "",
              "再現条件と公式ソース: [事前固定プロトコル](../../../benchmarks/tianshou_dqn_reference_notes.md)。",
              "全設定: [tianshou_dqn_protocol.json](../tianshou_dqn_protocol.json)。",
              "全系列と入力ハッシュ: [comparison.json](comparison.json)。",
              "専用監査: [tianshou_dqn_audit.json](../tianshou_dqn_audit.json)。", ""]
    (args.output / "README.ja.md").write_text("\n".join(lines))
    print(json.dumps({"status": data["status"], "completed_models": complete_count, "aggregate": aggregate,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
