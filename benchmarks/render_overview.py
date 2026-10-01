#!/usr/bin/env python3
"""Make a compact, per-environment overview from the audited report summary.

This adds no training and makes no checkpoint or worker selection. Detailed
episode curves, confidence intervals and comparison limitations remain in the
main report. Error bars here are training-seed SD, not confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/oss_benchmarks"
PANELS = [
    ("CartPole-v1", ["cartpole_dqn", "cartpole_ppo", "cartpole_sac",
                     "cartpole_dqn_parallel4", "cartpole_sac_parallel4"]),
    ("LunarLander-v3", ["lunar_dqn", "lunar_ppo", "lunar_rnd",
                        "lunar_ppo_parallel2", "lunar_rnd_parallel2", "lunar_rnd_shared2"]),
    ("LunarLanderContinuous-v3", ["lunar_sac"]),
    ("Ant-v5", ["ant_ppo", "ant_sac", "ant_shared", "ant_rnd_shared"]),
    ("Hopper-v5", ["hopper_sac"]),
    ("Walker2d-v5", ["walker_ppo"]),
    ("HalfCheetah-v5", ["halfcheetah_sac", "halfcheetah_hybrid"]),
]
LABELS = {
    "cartpole_dqn": "DQN", "cartpole_ppo": "PPO", "cartpole_sac": "Discrete SAC",
    "cartpole_dqn_parallel4": "DQN / 4 workers", "cartpole_sac_parallel4": "SAC / 4 workers",
    "lunar_dqn": "DQN", "lunar_ppo": "PPO", "lunar_rnd": "PPO + RND",
    "lunar_ppo_parallel2": "PPO / 2 workers", "lunar_rnd_parallel2": "RND / 2 independent",
    "lunar_rnd_shared2": "RND / 2 shared", "lunar_sac": "SAC",
    "ant_ppo": "PPO", "ant_sac": "SAC", "ant_shared": "Mixed / {algorithm}",
    "ant_rnd_shared": "Mixed + RND / {algorithm}", "hopper_sac": "SAC", "walker_ppo": "PPO",
    "halfcheetah_sac": "SAC", "halfcheetah_hybrid": "2 PPO + 2 SAC / {algorithm}",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT / "summary.json")
    parser.add_argument("--tianshou-root", type=Path, default=DEFAULT / "runs/tianshou/cartpole_sac")
    parser.add_argument("--tianshou-dqn-root", type=Path, default=DEFAULT / "runs/tianshou_dqn")
    parser.add_argument("--output", type=Path, default=DEFAULT / "figures")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    seeds = summary["protocol"]["training_seeds"]
    groups = summary["groups"]
    lookup = {(g["backend"], g["condition"], g["algorithm"]): g for g in groups}
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), layout="constrained")
    colors = {"native": "#176b92", "sb3": "#c36b1c", "tianshou": "#2f8054"}
    def reference_means(directory, budget):
        values = {}
        for seed in seeds:
            final = directory / f"seed_{seed}" / "final.json"
            if not final.exists():
                continue
            record = json.loads(final.read_text())
            evaluation = record["final_evaluation"]
            episodes_path = final.parent / "eval_episodes.jsonl"
            episodes = [row for line in episodes_path.read_text().splitlines()
                        if (row := json.loads(line))["phase"] == "final"]
            if (len(episodes) == evaluation["episodes"] == 100
                    and {row["seed"] for row in episodes} == set(range(900000, 900100))
                    and record["actual_steps"] == budget
                    and math.isclose(statistics.fmean(row["return"] for row in episodes),
                                     evaluation["mean_return"], abs_tol=1e-8)):
                values[str(seed)] = evaluation["mean_return"]
        return values
    tmeans = reference_means(args.tianshou_root, 204800)
    dqn_means = {case: reference_means(args.tianshou_dqn_root / case, budget)
                for case, budget in (("cartpole_dqn", 204800), ("lunar_dqn", 1024000))}

    def draw(axis, values, y, backend, offset):
        if not values:
            return
        mean = statistics.fmean(values.values())
        sd = statistics.stdev(values.values()) if len(values) > 1 else None
        complete = len(values) == len(seeds)
        color = colors[backend]
        axis.errorbar(mean, y + offset, xerr=sd, fmt="o", color=color,
                      markerfacecolor=color if complete else "white", markersize=6,
                      capsize=3, linewidth=1.3, zorder=4)
        for index, seed in enumerate(seeds):
            if str(seed) in values:
                axis.scatter(values[str(seed)], y + offset + (index - 1) * .045,
                             color=color, alpha=.4, marker="|", s=80, zorder=3)
        return mean

    for axis, (environment, cases) in zip(axes.flat, PANELS):
        selected = [g for case in cases for g in groups
                    if g["backend"] == "native" and g["condition"] == case]
        labels = []
        threshold = summary["protocol"]["official_thresholds"][environment]
        for y, group in enumerate(selected):
            label = LABELS[group["condition"]].format(algorithm=group["algorithm"].upper())
            labels.append(f"{label}  [{group['n']}/{len(seeds)}]")
            draw(axis, group["seed_means"], y, "native", -.18)
            reference = group.get("reference", {}).get("reference_condition")
            other = lookup.get(("sb3", reference, group["algorithm"]))
            if other:
                draw(axis, other["seed_means"], y, "sb3", .02)
            if group["condition"] == "cartpole_sac":
                draw(axis, tmeans, y, "tianshou", .22)
            if group["condition"] in dqn_means:
                draw(axis, dqn_means[group["condition"]], y, "tianshou", .22)
        if threshold is not None:
            axis.axvline(threshold, color="#80545a", linestyle="--", alpha=.8, linewidth=1)
            threshold_label = f"Registered threshold: {threshold:g}"
        else:
            threshold_label = "No registered threshold; see matched SB3 comparison"
        axis.set(title=f"{environment}\n{threshold_label}", xlabel="Raw final-test return",
                 yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels) - .5, -.5))
        axis.axvline(0, color="#999999", linewidth=.5, alpha=.35)
        axis.grid(axis="x", alpha=.18)
        axis.tick_params(axis="y", labelsize=8)

    notes = axes.flat[-1]
    notes.axis("off")
    completed = sum(r["valid_final"] for r in summary["runs"])
    dqn_completed = sum(len(values) for values in dqn_means.values())
    complete = (completed == len(summary["runs"]) and len(tmeans) == len(seeds)
                and dqn_completed == 2 * len(seeds))
    notes.text(.02, .96,
               "Fixed final checkpoint, deterministic evaluation\n\n"
               "100 held-out episodes per trained worker\n"
               "Dots: mean across training seeds; bars: seed SD\n"
               "Small ticks: individual training-seed means\n"
               "Parallel workers are averaged within each seed\n"
               "[n/3]: completed native training seeds\n"
               "Hollow markers: incomplete seed set\n\n"
               "Native: example settings / CPU C FFI\n"
               "Reference: same environment, total steps and\n"
               "reward preprocessing; implementation differences remain.\n"
               "Mixed and parallel comparisons are structural controls.\n\n"
               f"Main runs: {completed}/{len(summary['runs'])}\n"
               f"Additional discrete SAC reference: {len(tmeans)}/{len(seeds)}\n"
               f"Additional Double DQN reference: {dqn_completed}/{2 * len(seeds)}\n"
               "See REPORT.ja.md for gates, CI and worker-level results.",
               transform=notes.transAxes, va="top", fontsize=10, linespacing=1.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=colors[b], label=label)
                        for b, label in [("native", "ReinforceX"), ("sb3", "SB3 reference"),
                                         ("tianshou", "Tianshou reference")]],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("CPU learning benchmark | Final performance by environment\n"
                 + ("COMPLETE" if complete else "IN PROGRESS — performance gates remain pending"),
                 fontsize=16)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"overview.{extension}", dpi=180)
    plt.close(fig)
    print(args.output / "overview.png")


if __name__ == "__main__":
    main()
