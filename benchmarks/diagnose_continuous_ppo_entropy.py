#!/usr/bin/env python3
"""Post-hoc Walker entropy hypothesis experiment, separate from primary benchmarks.

Two fresh-seed 200k-step CPU runs differ only in entropy coefficient. Validation
uses seeds 800000..800009; the held-out primary test seeds are never evaluated.
Core, examples, native runner/configuration and primary results stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


def tensor_digest(path):
    with zipfile.ZipFile(path) as archive:
        payloads = sorted(hashlib.sha256(archive.read(name)).hexdigest()
                          for name in archive.namelist() if "/data/" in name)
    assert payloads
    return hashlib.sha256(json.dumps(payloads).encode()).hexdigest()


def run(output, entropy):
    import native_configs
    import run_native as native
    lib = native.rx.load_reinforcex()
    output.mkdir(parents=True, exist_ok=False)
    case = native_configs.config(lib, "walker_ppo")
    cfg = case["agents"][0]["config"]
    cfg.entropy_coefficient = entropy
    native.rx.manual_seed(lib, 3101)
    initial_path = output / "initial.ot"
    # Use two fresh agents with identical policy initialization to save the
    # initial tensor fingerprint without changing the final checkpoint path.
    initial = native.rx.create_ppo(lib, cfg, str(initial_path), None)
    initial.save()
    initial.close()
    native.rx.manual_seed(lib, 3101)
    agent = native.rx.create_ppo(lib, cfg, str(output / "final.ot"), None)
    started = time.monotonic()
    counter = {"steps": 0, "lock": threading.Lock()}
    worker = native.Worker(agent, case["env_id"], 3101, case["reward_mode"],
                           output, 0, "ppo", counter)
    libpath = Path(os.environ["REINFORCEX_LIB"])
    native.write_json(output / "metadata.json", {
        "experiment": "post-hoc exploratory entropy hypothesis; not primary benchmark",
        "trigger": "Walker seed42 high latent entropy after primary run",
        "training_seed": 3101, "policy_seed": 3101, "environment_seed": 3101,
        "requested_steps": 200000, "validation_seed": 800000, "validation_episodes": 10,
        "held_out_test_evaluated": False, "configuration": native_configs.as_dict(case),
        "initial_tensor_payload_sha256": tensor_digest(initial_path),
        "library_sha256": hashlib.sha256(libpath.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_limitations": "Rust minibatch shuffle thread_rng remains unseeded; one pair is exploratory, not a multi-seed causal estimate",
        "device": "cpu", "threads": 1,
    })
    points = []
    try:
        for checkpoint in range(6):
            if checkpoint:
                worker.train(40000, final=checkpoint == 5)
            point = {"steps": counter["steps"], "entropy_coefficient": entropy,
                     "statistics": agent.statistics(),
                     "validation": native.evaluate(agent, case["env_id"], 10, 800000),
                     "seconds": time.monotonic() - started}
            points.append(point)
            native.write_json(output / "progress.json", point)
            print(json.dumps(point), flush=True)
        agent.save()
        native.write_json(output / "result.json", {"status": "complete", "points": points,
                                                   "seconds": time.monotonic() - started})
    finally:
        worker.close()
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/oss_benchmarks/exploratory/walker_entropy_seed3101")
    parser.add_argument("--entropy", type=float)
    args = parser.parse_args()
    if args.entropy is not None:
        run(args.output, args.entropy)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    children = []
    for label, coefficient in (("entropy_0_01", 0.01), ("entropy_0", 0.0)):
        log = (args.output / f"{label}.log").open("w")
        process = subprocess.Popen([sys.executable, str(Path(__file__)), "--output",
                                    str(args.output / label), "--entropy", str(coefficient)],
                                   stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        children.append((label, process, log))
    statuses = {}
    for label, process, log in children:
        statuses[label] = process.wait()
        log.close()
    if any(statuses.values()):
        raise RuntimeError(statuses)
    metadata = [json.loads((args.output / label / "metadata.json").read_text())
                for label, _, _ in children]
    assert metadata[0]["initial_tensor_payload_sha256"] == metadata[1]["initial_tensor_payload_sha256"]
    summary = {"experiment": "post-hoc auxiliary pair; cannot replace primary three-seed result",
               "initial_tensor_payloads_equal": True, "status": statuses,
               "results": {label: json.loads((args.output / label / "result.json").read_text())
                           for label, _, _ in children}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"initial_tensor_payloads_equal": True, "status": statuses}))


if __name__ == "__main__":
    main()
