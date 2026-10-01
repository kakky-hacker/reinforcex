"""Bounded CPU FFI diagnosis of SparseChain budget and exploration sensitivity."""
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


def greedy_actions(agent):
    return [int(agent.act([position / 6, 1])) for position in range(6)]


def initial_right_probabilities(lib, config, seed):
    probe_config = type(config).from_buffer_copy(bytes(config))
    probe_config.update_interval = 1000000
    validation.rx.manual_seed(lib, seed)
    agent = validation.make_agent(lib, "ppo", probe_config)
    try:
        probabilities = []
        for position in range(6):
            right = 0
            for _ in range(256):
                right += int(agent.act_and_train([position / 6, 1], 0.0) == 1)
                agent.stop_episode([position / 6, 1], 0.0, terminated=False)
            probabilities.append(right / 256)
        assert agent.statistics()["updates"] == 0
        return probabilities
    finally:
        agent.close()


def run_case(spec):
    lib = validation.rx.load_reinforcex()
    config = validation.configuration(lib, spec["algorithm"], 2, 2, True)
    config.entropy_coefficient = spec["entropy"]
    initial_probabilities = initial_right_probabilities(lib, config, spec["seed"])
    validation.rx.manual_seed(lib, spec["seed"] + 1000000)
    rnd = validation.make_rnd(lib, 2, None) if spec["algorithm"] == "rnd" else None
    validation.rx.manual_seed(lib, spec["seed"])
    agent = validation.make_agent(lib, spec["algorithm"], config, rnd=rnd)
    started = time.monotonic()
    try:
        result = {"spec": spec, "initial_right_probabilities_256_samples": initial_probabilities,
                  "initial_greedy_actions": greedy_actions(agent), "checkpoints": []}
        previous = 0
        for budget in (4000, 32000):
            training = validation.train(agent, "SparseChain", spec["seed"], budget - previous)
            assert training["statistics"]["updates"] == budget // config.update_interval
            evaluation = validation.evaluate(agent, "SparseChain", spec["seed"] + 100000, episodes=1)
            checkpoint = {"total_steps": budget, "greedy_success": evaluation["mean"],
                          "greedy_actions": greedy_actions(agent),
                          "phase_success_rate": training["terminal_episodes"] / training["episodes"],
                          "training": training}
            result["checkpoints"].append(checkpoint)
            previous = budget
        result["seconds"] = time.monotonic() - started
        return result
    finally:
        agent.close()
        if rnd:
            rnd.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, default=validation.ROOT / "reports/cpu_validation/sparse_diagnostics.json")
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(run_case(json.loads(args.child)), allow_nan=False))
        return
    specs = [{"algorithm": algorithm, "seed": seed, "entropy": entropy}
             for entropy in (0.001, 0.05) for seed in (42, 123, 2026) for algorithm in ("ppo", "rnd")]
    deadline = time.monotonic() + args.seconds
    started = time.monotonic()
    report = {"library": os.environ.get("REINFORCEX_LIB"), "threads_per_process": 1,
              "subprocess_workers": 3, "seeds": [42, 123, 2026], "results": [],
              "limitations": ["Rust minibatch shuffling remains unseeded.",
                               "4k and 32k checkpoints share one training run; the 4k budget truncates its last episode.",
                               "SparseChain evaluation is deterministic, so one greedy episode suffices."]}
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
        except subprocess.TimeoutExpired:
            return {"spec": spec, "status": "deadline"}
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
