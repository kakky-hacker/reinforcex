#!/usr/bin/env python3
"""Render the supplementary native/Tianshou CartPole discrete SAC comparison."""
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
SEEDS = (42, 123, 2026)
COLORS = {"native": "#1664a6", "tianshou": "#df7631"}
LABELS = {"native": "ReinforceX", "tianshou": "Tianshou 2.0.1"}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return statistics.mean(values)


def rolling(values, window=20):
    return [mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def collect():
    records, series, source_hashes = [], [], {}
    for backend in ("native", "tianshou"):
        for seed in SEEDS:
            directory = BASE / "runs" / backend / "cartpole_sac" / f"seed_{seed}"
            if not (directory / "final.json").exists():
                records.append({"backend": backend, "seed": seed, "status": "pending"})
                continue
            final = read(directory / "final.json")
            evaluation = rows(directory / "evaluations.jsonl")
            train_path = directory / ("train_worker0.jsonl" if backend == "native" else "train_episodes.jsonl")
            train = rows(train_path)
            for path in (directory / "final.json", directory / "evaluations.jsonl", train_path):
                source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if backend == "native":
                complete = [r for r in train if not r["budget_cut"]]
                returns = [r["reward"] for r in complete]
                test = final["test"][0]
                steps, seconds = final["actual_total_steps"], final["training_seconds"]
                test_returns = test["returns"]
                alpha = final["workers"][0]["statistics"]["temperature"]
                updates = final["workers"][0]["statistics"]["n_updates"]
                ev = [{"steps": p["aggregate_steps"], "mean_return": p["mean"]}
                      for p in evaluation if p["split"] == "validation"]
            else:
                returns = [r["return"] for r in train]
                complete = train
                test = final["final_evaluation"]
                steps, seconds = final["actual_steps"], final["training_seconds_excluding_evaluation"]
                test_returns = [r["return"] for r in rows(directory / "eval_episodes.jsonl") if r["phase"] == "final"]
                alpha, updates = final["final_alpha"], final["updates"]
                ev = [{"steps": p["env_steps"], "mean_return": p["mean_return"]}
                      for p in evaluation if p["phase"] != "final"]
            final_mean = mean(test_returns)
            record = {"backend": backend, "seed": seed, "status": "complete", "steps": steps,
                      "completed_episodes": len(complete), "last100_training_mean": mean(returns[-100:]),
                      "heldout_episodes": len(test_returns), "final_mean": final_mean,
                      "final_sd": statistics.stdev(test_returns), "passed_475": final_mean >= 475,
                      "training_seconds": seconds, "updates": updates, "final_alpha": alpha}
            records.append(record)
            series.append({"backend": backend, "seed": seed, "episode_returns": returns,
                           "episode_steps": [r["steps"] if backend == "native" else r["env_steps"] for r in complete],
                           "validation": ev, "heldout_returns": test_returns})
    return records, series, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "discrete_sac_reference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, series, hashes = collect()
    summary = {}
    for backend in LABELS:
        complete = [r for r in records if r["backend"] == backend and r["status"] == "complete"]
        values = [r["final_mean"] for r in complete]
        summary[backend] = {"completed_seeds": len(complete), "required_seeds": len(SEEDS),
                            "mean_over_training_seeds": mean(values) if values else None,
                            "sample_sd_over_training_seeds": statistics.stdev(values) if len(values) > 1 else None,
                            "passing_seeds": sum(r["passed_475"] for r in complete)}
    data = {"status": "complete" if all(r["status"] == "complete" for r in records) else "incomplete",
            "runs": records, "aggregate": summary, "series": series, "source_sha256": hashes}
    (args.output / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for entry in series:
            if entry["seed"] != seed:
                continue
            values, backend = entry["episode_returns"], entry["backend"]
            x = range(1, len(values) + 1)
            ax.plot(x, values, alpha=.15, color=COLORS[backend], linewidth=.6)
            ax.plot(x, rolling(values), color=COLORS[backend], label=LABELS[backend], linewidth=1.7)
        ax.set(title=f"Training seed {seed}", xlabel="Completed training episode", ylim=(0, 515))
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(loc="lower right", fontsize=8)
        missing = [LABELS[b] for b in LABELS if not any(e["seed"] == seed and e["backend"] == b for e in series)]
        if missing:
            ax.text(.02, .94, "Pending: " + ", ".join(missing), transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("Raw episode return")
    fig.suptitle("CartPole-v1 discrete SAC: raw returns and trailing 20-episode mean")
    fig.text(.5, .015, "204,800 transitions per run. Partial final episodes excluded. Episode counts differ with policy performance.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"episode_vs_reward.{suffix}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for entry in series:
            if entry["seed"] == seed:
                backend, points = entry["backend"], entry["validation"]
                ax.plot([p["steps"] for p in points], [p["mean_return"] for p in points],
                        color=COLORS[backend], marker="o", markersize=3, label=LABELS[backend])
        ax.axhline(475, color="#777777", linestyle="--", linewidth=.8)
        ax.set(title=f"Training seed {seed}", xlabel="Training environment transitions", ylim=(0, 515))
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if len(ax.lines) > 1:
            ax.legend(loc="lower right", fontsize=8)
        missing = [LABELS[b] for b in LABELS if not any(e["seed"] == seed and e["backend"] == b for e in series)]
        if missing:
            ax.text(.02, .94, "Pending: " + ", ".join(missing), transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("Mean raw deterministic evaluation return")
    fig.suptitle("CartPole-v1 discrete SAC: fixed 10-episode validation curves")
    fig.text(.5, .015, "Validation seeds 800000-800009; these points do not select checkpoints. Final table uses 100 held-out episodes.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"steps_vs_evaluation.{suffix}", dpi=170)
    plt.close(fig)
    lines = ["# CartPole 離散 SAC：追加外部対照", "",
             "ReinforceX と Tianshou 2.0.1 の同手法比較です。主比較 90 run に対する補足として追加しました。",
             "同一 Gymnasium 1.3.0 / Torch 2.7 系 / CPU 1 thread、各 seed 204,800 transitions。",
             "学習 seed は 42 / 123 / 2026、最終評価は各モデル 100 episode（900000–900099）。", "",
             f'集計済み **{sum(r["status"] == "complete" for r in records)}/6 model**。'
             + ("全モデルの最終評価が完了しています。" if data["status"] == "complete" else
                "未完了モデルの欄は保留です。全 seed が揃うまで実装間の総合判定は行いません。"), "",
             "![Episode vs reward](episode_vs_reward.png)", "",
             "薄線は各完了 episode の生報酬、太線は直近 20 episode 平均です。最後の未完了 episode は除外しています。",
             "同じ学習ステップ数でも episode 数は異なるため、サンプル効率は下の step 軸も参照してください。", "",
             "![Steps vs evaluation](steps_vs_evaluation.png)", "",
             "| 実装 | seed | 最終生報酬 mean ± SD | 475 以上 | 完了 episode | 学習末尾100平均 | alpha |",
             "|---|---:|---:|:---:|---:|---:|---:|"]
    for row in records:
        if row["status"] == "complete":
            lines.append(f'| {LABELS[row["backend"]]} | {row["seed"]} | {row["final_mean"]:.2f} ± {row["final_sd"]:.2f} | '
                         f'{"○" if row["passed_475"] else "×"} | {row["completed_episodes"]} | '
                         f'{row["last100_training_mean"]:.2f} | {row["final_alpha"]:.3g} |')
        else:
            lines.append(f'| {LABELS[row["backend"]]} | {row["seed"]} | 未完了 | 保留 | — | — | — |')
    lines += ["", "SD は同一モデルの評価 episode 間の標準偏差です。訓練 seed 間のばらつきとは区別します。", ""]
    for backend, item in summary.items():
        if item["completed_seeds"] == 3:
            lines.append(f'- {LABELS[backend]}: 3 seed の最終平均の平均 **{item["mean_over_training_seeds"]:.2f}**、'
                         f'seed 間 SD **{item["sample_sd_over_training_seeds"]:.2f}**、475 以上 **{item["passing_seeds"]}/3**。')
    lines += ["", "主要設定は actor/critic 各 ReLU64×2、actor lr0.0003 / critic lr0.0005、gamma0.99、",
              "replay50,000 / warmup512 / batch64 / n-step3、tau0.005、初期alpha0.05、",
              "自動調整の目標entropy0.01×ln2、alpha lr0.0003です。訓練報酬変換も同じです。", "",
              "実装差として、ReinforceX の critic は Huber(delta1)・勾配norm10クリップ、",
              "Tianshou 公式は MSE・クリップなしです。また n-step replay の末尾と warmup、",
              "行動選択と更新の順序、episode 終端後の更新、初期化、乱数系列が異なります。",
              "Tianshou の学習アルゴリズムは改変していません。主 benchmark の判定基準も変更していません。", "",
              "seed 数は 3 のため、一般的な性能優位まで断定する資料ではありません。",
              "所要時間は他の CPU 学習との同時実行の影響を含むため、速度比較には使いません。", "",
              "再現条件・公式ソース: [環境と実装差の記録](../../../benchmarks/tianshou_reference_notes.md)。",
              "全 episode データと入力ハッシュ: [comparison.json](comparison.json)。",
              "整合性監査: [tianshou_audit.json](../tianshou_audit.json)。", ""]
    (args.output / "README.ja.md").write_text("\n".join(lines))
    print(json.dumps({"status": data["status"], "aggregate": summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
