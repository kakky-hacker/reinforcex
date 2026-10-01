"""Summarize saved example logs without re-running or inventing episode returns."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "reports/cpu_validation/example_logs"
PATTERN = re.compile(
    r"agent=(?P<worker>\S+)\s+episode=\s*(?P<episode>\d+)\s+steps=\s*(?P<steps>\d+)"
    r"\s+return=\s*(?P<reward>[-+\d.eE]+)\s+train_return=\s*(?P<train_reward>[-+\d.eE]+)"
    r"\s+mean\((?P<window>\d+)\)=\s*(?P<mean>[-+\d.eE]+)"
)
# Prefer the final build for every PPO case. DQN/SAC-only runs were recorded
# before the final PPO-only changes and are explicitly labelled interim.
RUNS = [
    ("cartpole_dqn", "CartPole / DQN", "CartPole-v1", "DQN", "interim", "cartpole_dqn.log", 40, 100),
    ("cartpole_ppo", "CartPole / PPO", "CartPole-v1", "PPO", "final", "final_build/cartpole_ppo.log", 40, 100),
    ("cartpole_sac", "CartPole / SAC", "CartPole-v1", "SAC", "interim", "cartpole_sac.log", 40, 100),
    ("lunar_lander_dqn", "LunarLander / DQN", "LunarLander-v3", "DQN", "interim", "lunar_lander_dqn.log", 20, 100),
    ("lunar_lander_ppo_rnd", "LunarLander / PPO + RND", "LunarLander-v3", "PPO + RND", "final", "final_build/lunar_lander_ppo_rnd.log", 20, 100),
    ("lunar_lander_sac", "LunarLanderContinuous / SAC", "LunarLanderContinuous-v3", "SAC", "interim", "lunar_lander_sac.log", 40, 100),
    ("ant_ppo", "Ant / PPO", "Ant-v5", "PPO", "final", "final_build/ant_ppo.log", 30, 100),
    ("hopper_sac", "Hopper / SAC (100-episode run)", "Hopper-v5", "SAC", "interim", "hopper_sac_extended.log", 100, 100),
    ("hopper_sac_short", "Hopper / SAC (30-episode run)", "Hopper-v5", "SAC", "interim", "hopper_sac.log", 30, 100),
    ("walker2d_ppo", "Walker2d / PPO", "Walker2d-v5", "PPO", "final", "final_build/walker2d_ppo.log", 30, 100),
    ("half_cheetah_hybrid", "HalfCheetah / PPO + SAC", "HalfCheetah-v5", "mixed", "final", "final_build/half_cheetah_hybrid.log", 4, 64),
    ("ant_ppo_rnd_sac_shared", "Ant / three replay conditions", "Ant-v5", "mixed", "final", "final_build/ant_ppo_rnd_sac_shared.log", 4, 64),
    ("lunar_lander_ppo_rnd_resume", "LunarLander / PPO + RND resumed", "LunarLander-v3", "PPO + RND", "final", "final_build/lunar_lander_ppo_rnd_resume.log", 15, 100),
]


def parse_logs():
    runs = []
    rows = []
    for run_id, title, environment, algorithm, build, filename, episodes, max_steps in RUNS:
        path = LOGS / filename
        run_rows = []
        for match in PATTERN.finditer(path.read_text()):
            g = match.groupdict()
            run_rows.append({"run": run_id, "environment": environment, "algorithm": algorithm,
                             "condition": "", "build": build, "worker": g["worker"],
                             "episode": int(g["episode"]), "steps": int(g["steps"]),
                             "episode_return": float(g["reward"]), "train_return": float(g["train_reward"]),
                             "cumulative_mean": float(g["mean"]), "mean_window": int(g["window"]),
                             "max_episode_steps": max_steps,
                             "source": str(path.relative_to(ROOT)), "precision": "rounded_log"})
        assert run_rows, path
        assert len({(r["worker"], r["episode"]) for r in run_rows}) == len(run_rows)
        assert all(r["mean_window"] == r["episode"] for r in run_rows), "not cumulative mean"
        # Hybrid JSON retains every episode, including those absent from stdout.
        if algorithm == "mixed":
            json_path = path.with_name(path.stem + "_results.json")
            document = json.loads(json_path.read_text())
            actual = []
            if run_id == "half_cheetah_hybrid":
                for algo in ("ppo", "sac"):
                    for worker, rewards in enumerate(document[algo]["training_returns"]):
                        actual.append((f"{algo}-{worker}", algo.upper(), "shared_replay", rewards))
            else:
                for condition, record in document["conditions"].items():
                    for algo, workers in record["training_returns"].items():
                        for rewards in workers:
                            actual.append((f"{condition}-{algo}", "PPO + RND" if algo == "ppo" and condition == "rnd_shared" else algo.upper(), condition, rewards))
            parsed = {(r["worker"], r["episode"]): r for r in run_rows}
            run_rows = []
            for worker, algo, condition, rewards in actual:
                assert len(rewards) == episodes
                for index, reward in enumerate(rewards, 1):
                    logged = parsed.get((worker, index))
                    mean = float(np.mean(rewards[:index]))
                    if logged:
                        assert abs(logged["episode_return"] - reward) <= 0.00501, (worker, index, logged, reward)
                        assert abs(logged["cumulative_mean"] - mean) <= 0.00501, (worker, index)
                    run_rows.append({"run": run_id, "environment": environment, "algorithm": algo,
                                     "condition": condition, "build": build, "worker": worker,
                                     "episode": index, "steps": logged["steps"] if logged else None,
                                     "episode_return": reward, "train_return": logged["train_return"] if logged else None,
                                     "cumulative_mean": mean, "mean_window": index,
                                     "max_episode_steps": max_steps,
                                     "source": str(json_path.relative_to(ROOT)), "precision": "full_json"})
        assert all(np.isfinite(r["episode_return"]) and np.isfinite(r["cumulative_mean"]) for r in run_rows)
        assert max(r["episode"] for r in run_rows) == episodes
        workers = sorted({r["worker"] for r in run_rows})
        runs.append({"id": run_id, "title": title, "environment": environment, "build": build,
                     "episodes_per_worker": episodes, "max_steps": max_steps, "workers": workers,
                     "recorded_points": len(run_rows), "complete_episode_returns": algorithm == "mixed",
                     "log": str(path.relative_to(ROOT))})
        rows.extend(run_rows)
    return runs, rows


def average_series(rows, run_id, algorithm=None, condition=None):
    grouped = defaultdict(list)
    selected = [r for r in rows if r["run"] == run_id and
                (algorithm is None or r["algorithm"] == algorithm) and
                (condition is None or r["condition"] == condition)]
    for row in selected:
        grouped[row["episode"]].append(row)
    worker_count = len({r["worker"] for r in selected})
    assert worker_count > 0
    assert all(len(group) == worker_count for group in grouped.values()), "unbalanced episode logging"
    return [(episode, float(np.mean([r["episode_return"] for r in group])),
             float(np.mean([r["cumulative_mean"] for r in group])))
            for episode, group in sorted(grouped.items())]


def figures(out, runs, rows):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    panels = [
        ("CartPole", [("DQN [interim]", "cartpole_dqn", None, None), ("PPO [final]", "cartpole_ppo", None, None), ("SAC [interim]", "cartpole_sac", None, None)]),
        ("LunarLander (discrete)", [("DQN [interim]", "lunar_lander_dqn", None, None), ("PPO + RND [final]", "lunar_lander_ppo_rnd", None, None)]),
        ("LunarLander (continuous)", [("SAC [interim]", "lunar_lander_sac", None, None)]),
        ("Hopper: two independent runs", [("SAC / 100 ep [interim]", "hopper_sac", None, None), ("SAC / 30 ep [interim]", "hopper_sac_short", None, None)]),
        ("Walker2d", [("PPO [final]", "walker2d_ppo", None, None)]),
        ("Ant: standalone PPO", [("PPO [final]", "ant_ppo", None, None)]),
        ("HalfCheetah: shared replay", [("PPO [final]", "half_cheetah_hybrid", "PPO", None), ("SAC [final]", "half_cheetah_hybrid", "SAC", None)]),
        ("Ant: SAC in each condition", [("SAC + PPO/RND replay", "ant_ppo_rnd_sac_shared", "SAC", "rnd_shared"), ("SAC + PPO replay", "ant_ppo_rnd_sac_shared", "SAC", "ppo_shared"), ("SAC only", "ant_ppo_rnd_sac_shared", "SAC", "sac_only")]),
        ("LunarLander: resumed PPO + RND", [("PPO + RND [final]; episode resets", "lunar_lander_ppo_rnd_resume", None, None)]),
    ]
    colors = ["#147D92", "#D87920", "#6B56A3"]
    for field, index, filename, label in [("raw", 1, "episode_vs_reward.png", "Episode return"),
                                           ("mean", 2, "episode_vs_mean_reward.png", "Cumulative mean return")]:
        fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
        for ax, (title, series) in zip(axes.flat, panels):
            for color, (name, run_id, algo, condition) in zip(colors, series):
                values = average_series(rows, run_id, algo, condition)
                ax.plot([v[0] for v in values], [v[index] for v in values], "o-", lw=1.8, ms=4, color=color, label=name)
            ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
            ax.set_xlabel("Episode (per worker)")
            ax.set_ylabel(label)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
        fig.suptitle("Python FFI examples: episode vs reward" + (" (cumulative mean)" if field == "mean" else " (raw episode returns)") +
                     "\nWorker averages; points are saved observations; connecting lines do not recover missing episodes.", fontsize=16, fontweight="bold")
        fig.savefig(out / filename, dpi=150)
        fig.savefig(out / filename.replace(".png", ".pdf"))
        plt.close(fig)
    nrows = (len(runs) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(17, 3.8 * nrows), constrained_layout=True)
    for ax, run in zip(axes.flat, runs):
        for worker in run["workers"]:
            subset = sorted([r for r in rows if r["run"] == run["id"] and r["worker"] == worker], key=lambda r:r["episode"])
            ax.plot([r["episode"] for r in subset], [r["episode_return"] for r in subset], "o-", ms=3, lw=1.3, label=worker)
        coverage = "all episodes" if run["complete_episode_returns"] else "sampled episodes"
        ax.set_title(f"{run['title']}\n{run['build']} build | {coverage}", loc="left", fontsize=11)
        ax.set_xlabel("Episode (per worker)")
        ax.set_ylabel("Episode return")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2 if len(run["workers"])>2 else 1, framealpha=0.85)
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    fig.suptitle("Episode returns by worker — saved example training results", fontsize=17, fontweight="bold")
    fig.savefig(out / "episode_vs_reward_workers.png", dpi=150)
    plt.close(fig)


def report(out, runs, rows):
    text = """# examples: episode vs reward

既存example 11種類・13実行（Hopperの別runとRND再開を含む）の保存済み結果を再集計した。通常ログ74点＋全episodeを残した混成JSON36点＝110点。学習の再実行はしていない。横軸は**workerごとのepisode番号**、縦軸は**環境が返した報酬のepisode内合計（return）**。報酬整形後の`train_return`やRND内発報酬ではない。

- **通常例はログが間引かれている。** 点は実際に保存されたepisodeのみ。線は記録点をつなぐ補助線で、未記録のepisodeを復元したものではない。
- **累積平均はログの`mean(N)`を使用。** 記録点だけを平均していない。今回は全runが100episode以内なので、各runの先頭からそのepisodeまでの全episode平均に相当する。
- **PPO系は最終修正版、DQN/SAC単独は中間ビルドの記録。** 同一ビルドでの比較実験ではない。step上限は通常100、混成例64。seed・更新予算も異なり、手法間の優劣を確定する資料ではない。特にAnt混成のRNDあり／なし／SAC単独は共有replay遷移326／377／104、SAC更新79／122／18で予算が一致しない。
- HalfCheetahとAnt混成はJSONに全episodeが保存されているため全点を使用。
- Hopperの30episode runと100episode runは別の新規学習。図でも別系列とし、継続として接続しない。
- LunarLander RND再開は別図枠。保存モデルからの15episodeで、episode番号と平均窓は再び1から始まる。前の20episodeと結合していない。

## Episode vs reward

各点は同じepisode番号のworker間平均。下のworker別図には全系列を掲載。

![Episode vs raw environment return](episode_vs_reward.png)

## Episode vs累積平均reward

![Episode vs cumulative mean return](episode_vs_mean_reward.png)

## 数値要約

「最初」「最後」は保存されたepisodeそのもののreturnのworker間平均。「全episode平均」は最終ログの`mean(N)`をworker間平均した値。最初の1episodeは学習前の独立評価ではない。ログ由来の値は小数点2桁に丸められている。

| Example / 系列 | ビルド | ep/worker | workers | 保存点/予定点 | 最初のepisode return | 最後のepisode return | 全episode平均return |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    groups = []
    for run in runs:
        combinations = sorted({(r["algorithm"],r["condition"]) for r in rows if r["run"] == run["id"]})
        for algo, condition in combinations:
            subset = [r for r in rows if r["run"] == run["id"] and r["algorithm"]==algo and r["condition"]==condition]
            values = average_series(rows, run["id"], algo, condition)
            count = len({r["worker"] for r in subset})
            name = run["title"] if len(combinations)==1 else run["environment"] + " / " + algo + " / " + condition
            record = {"series": name, "build": run["build"], "episodes_per_worker":run["episodes_per_worker"],
                      "workers":count,"saved_points":len(subset),"possible_points":count*run["episodes_per_worker"],
                      "first_return":values[0][1],"last_return":values[-1][1],"overall_mean_return":values[-1][2]}
            groups.append(record)
            text += f"| {name} | {run['build']} | {run['episodes_per_worker']} | {count} | {len(subset)}/{count*run['episodes_per_worker']} | {values[0][1]:.2f} | {values[-1][1]:.2f} | {values[-1][2]:.2f} |\n"
    text += """
## 読み取り

- **CartPole SAC**: 記録されたepisode 40は両workerとも100step上限に到達。全40episodeの平均は57.84。DQNは23.89、PPOは31.21で、今回の短期実行の成績は限定的。上限100の試験なのでCartPole本来の500step到達を示さない。
- **Hopper SAC**: episode 100のreturnはworker 0が175.66、worker 1が169.26。最初の記録から改善しているが、100episode平均はworker間で97.89／52.00と差がある。
- **LunarLander・Ant・Walker2d・HalfCheetah**: 短い試験でばらつきが大きく、この記録だけで収束や攻略を確認できない。混成例は4episodeのみ。
- **RND**: 縦軸は外発reward。好奇心が正常に更新されることと、外発rewardが改善することは別。LunarLanderでは悪化したworkerもあり、この図からRNDの性能優位性は主張できない。
- 以前の統合レポートにあるCartPole約338〜461の値は、別の長めのFFI学習matrixの**学習後評価**。ここに掲載した短期examplesのepisode rewardとは別の実験である。

## Worker別の全系列

![Episode vs return by worker](episode_vs_reward_workers.png)

## データと再現

- [各記録episodeのCSV](episode_rewards.csv) — raw return、整形後reward、累積平均、worker、ビルド、出典。JSONにのみ残るepisodeではsteps/整形後rewardは空欄。
- [数値要約CSV](summary.csv)
- [系列・出典・収録範囲JSON](series.json)
- [raw return図PDF](episode_vs_reward.pdf) / [累積平均図PDF](episode_vs_mean_reward.pdf)
- [元のexamples検証レポート](../examples_audit.md)

```sh
MPLCONFIGDIR=/tmp/reinforcex-matplotlib python3 ffi/tests/summarize_example_rewards.py
```

numpy/matplotlibを使用。混成例はJSONの値とログの値が丸め誤差内で一致することを検証してから描画した。通常例の未記録episode値は補間・推測していない。
"""
    (out/"README.ja.md").write_text(text)
    with (out/"summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/cpu_validation/episode_rewards")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    runs,rows=parse_logs()
    with (args.output/"episode_rewards.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"series.json").write_text(json.dumps({"runs":runs,"points":rows},indent=2)+"\n")
    figures(args.output,runs,rows)
    summary=report(args.output,runs,rows)
    print(json.dumps({"runs":len(runs),"points":len(rows),"summary":summary},indent=2))


if __name__=="__main__":
    main()
