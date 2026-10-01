"""Render CPU validation figures and compact measured result tables."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/cpu_validation/full.json"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    out = args.input.parent
    groups = defaultdict(list)
    for entry in data["results"]:
        if entry["status"] == "passed" and entry["case"]["type"] == "learning":
            result = entry["result"]
            groups[(result["environment"], result["algorithm"])].append(result)
    names = {"dqn": "DQN", "ppo": "PPO", "sac": "SAC", "rnd": "PPO + RND"}
    table = ["| 環境 | 手法 | 1シードの学習step | 学習前平均 | 学習後平均 ± シード間SD | シード別学習後平均 |",
             "|---|---|---:|---:|---:|---|"]
    for (environment, algorithm), runs in groups.items():
        before = [r["before"]["mean"] for r in runs]
        after = [r["after"]["mean"] for r in runs]
        table.append(f"| {environment} | {names[algorithm]} | {runs[0]['training']['steps']:,} | {np.mean(before):.3f} | {np.mean(after):.3f} ± {np.std(after):.3f} | " + ", ".join(f"{v:.3f}" for v in after) + " |")
    (out / "learning_results.md").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, env, algs in zip(axes, ["CartPole-v1", "Pendulum-v1"], [("dqn", "ppo", "sac", "rnd"), ("ppo", "sac")]):
        positions = np.arange(len(algs))
        for offset, field, label, color in [(-0.18, "before", "Before training", "#9AA8B8"), (0.18, "after", "After training", "#187FA7")]:
            values = [np.mean([r[field]["mean"] for r in groups[(env, a)]]) for a in algs]
            ax.bar(positions + offset, values, 0.34, label=label, color=color)
            for i, algorithm in enumerate(algs):
                ys = [r[field]["mean"] for r in groups[(env, algorithm)]]
                ax.scatter(np.full(len(ys), i + offset), ys, color="#172E42", s=19, zorder=3)
        ax.set_xticks(positions, [names[a] for a in algs])
        ax.set_title(env)
        ax.set_ylabel("Evaluation return (higher is better)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False)
    fig.suptitle("CPU / Python FFI: 3 seeds, 20 evaluation episodes per seed", fontweight="bold")
    fig.savefig(out / "learning_results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for result in groups[("ContextBandit", "rnd")]:
        trace = [t for t in result["training"]["trace"] if t["statistics"].get("intrinsic_reward_mean", 0) > 0]
        ax.plot([t["step"] for t in trace], [t["statistics"]["intrinsic_reward_mean"] for t in trace], label=f"seed {result['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean intrinsic reward (last rollout)")
    ax.set_title("RND predictor error during ContextBandit training")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(out / "rnd_training.png", dpi=160)
    plt.close(fig)
    library = Path(data["library"])
    metadata = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--", "core", "ffi/src", "ffi/include", "examples"])).hexdigest(),
                "matrix_cases": len(data["results"]),
                "execution_failures": sum(x["status"] != "passed" for x in data["results"]),
                "learning_gate_failures": sum(x.get("result", {}).get("learning_gate_passed") is False for x in data["results"]),
                "training_steps_excluding_boundary_and_churn": sum(x["case"].get("steps", 0) * x["case"].get("workers", 1) for x in data["results"]),
                "subprocess_wall_seconds": sum(x["wall_seconds"] for x in data["results"])}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
