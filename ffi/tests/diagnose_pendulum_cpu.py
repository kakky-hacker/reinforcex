"""Bounded CPU FFI PPO budget/configuration diagnosis on Pendulum-v1."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_cpu_validation as validation


def run_case(spec):
    lib = validation.rx.load_reinforcex()
    config = validation.configuration(lib, "ppo", 3, 1, False)
    if spec["configuration"] == "lower_lr_longer_rollout":
        config.learning_rate = 3e-4
        config.update_interval = 1024
        config.agent.hidden_size = 64
        config.epochs = 10
        config.minibatch_size = 64
    validation.rx.manual_seed(lib, spec["seed"])
    agent = validation.make_agent(lib, "ppo", config)
    started = time.monotonic()
    settings = {"learning_rate": config.learning_rate, "update_interval": config.update_interval,
                "hidden_size": config.agent.hidden_size, "hidden_layers": config.agent.hidden_layers,
                "epochs": config.epochs, "minibatch_size": config.minibatch_size,
                "entropy_coefficient": config.entropy_coefficient, "reward_scale": 0.1}
    try:
        result = {"spec": spec, "settings": settings,
                  "before": validation.evaluate(agent, "Pendulum-v1", spec["seed"] + 100000),
                  "checkpoints": []}
        previous = 0
        for budget in (10000, 50000):
            training = validation.train(agent, "Pendulum-v1", spec["seed"], budget - previous)
            assert training["statistics"]["updates"] == budget // config.update_interval
            evaluation = validation.evaluate(agent, "Pendulum-v1", spec["seed"] + 100000)
            result["checkpoints"].append({"total_steps": budget, "evaluation": evaluation,
                                          "training": training,
                                          "gain_from_initial": evaluation["mean"] - result["before"]["mean"]})
            result["seconds"] = time.monotonic() - started
            print("PROGRESS " + json.dumps(result, allow_nan=False), flush=True)
            previous = budget
        return result
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, default=validation.ROOT / "reports/cpu_validation/pendulum_diagnostics.json")
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(run_case(json.loads(args.child)), allow_nan=False))
        return
    specs = [{"configuration": setting, "seed": seed}
             for setting in ("benchmark", "lower_lr_longer_rollout") for seed in (42, 123, 2026)]
    started = time.monotonic()
    deadline = started + args.seconds
    report = {"library": os.environ.get("REINFORCEX_LIB"), "environment": "Pendulum-v1",
              "threads_per_process": 1, "subprocess_workers": 3,
              "evaluation_episodes": 20, "results": [],
              "limitations": ["Rust minibatch shuffling remains unseeded.",
                               "The 10k and 50k checkpoints share one training run.",
                               "The second phase reuses the environment seed sequence.",
                               "Several hyperparameters change together; this is not a one-factor causal ablation."]}
    def invoke(spec):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"spec": spec, "status": "deadline"}
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)],
                                     capture_output=True, text=True, timeout=remaining)
            lines = [line[7:] for line in process.stdout.splitlines() if line.startswith("RESULT ")]
            if process.returncode or len(lines) != 1:
                return {"spec": spec, "status": "failed", "stderr": process.stderr[-6000:], "stdout": process.stdout[-2000:]}
            return {"spec": spec, "status": "completed", "result": json.loads(lines[0])}
        except subprocess.TimeoutExpired as error:
            output = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            progress = [line[9:] for line in output.splitlines() if line.startswith("PROGRESS ")]
            return {"spec": spec, "status": "deadline", "partial_result": json.loads(progress[-1]) if progress else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        for future in as_completed([executor.submit(invoke, spec) for spec in specs]):
            entry = future.result()
            report["results"].append(entry)
            report["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(entry["spec"], entry["status"], flush=True)
    print(f"Saved {args.output} ({report['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
