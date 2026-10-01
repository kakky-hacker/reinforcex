#!/usr/bin/env python3
"""Read-only final PPO distribution audit; native and Python Torch use separate processes.

Run with the reference Python environment. Checkpoints and running learners are
never modified. Native probe agents cannot reach their million-transition update
interval. Only two short evaluation episodes per case interact with Gymnasium.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CASES = ("walker_ppo", "ant_ppo")


def native_probe(scratch: Path, selected_case=None, workers=(0,), training_seed=42):
    import numpy as np
    import gymnasium as gym
    sys.path.insert(0, str(ROOT / "examples"))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import reinforcex_ffi as rx
    import native_configs

    lib = rx.load_reinforcex()
    library_sha256 = hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
    result = {}
    selections = [(case, worker) for case in ([selected_case] if selected_case else CASES)
                  for worker in workers]
    for case, worker in selections:
        key = f"{case}_worker{worker}" if selected_case else case
        run = ROOT / "reports/oss_benchmarks/runs/native" / case / f"seed_{training_seed}"
        metadata = json.loads((run / "metadata.json").read_text())
        assert metadata["library_sha256"] == library_sha256, "training library differs from probe library"
        checkpoint = run / f"worker{worker}.ot"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        spec = native_configs.config(lib, case)
        assert spec["agents"][worker]["algorithm"] == "ppo", "only PPO workers can be diagnosed"
        cfg = spec["agents"][worker]["config"]
        assert cfg.agent.hidden_layers == 1 and cfg.agent.hidden_size == 256
        assert cfg.min_action == -1 and cfg.max_action == 1
        cfg.update_interval = 1_000_000
        policy_seed = metadata["worker_seeds"][worker]["policy_seed"]
        rx.manual_seed(lib, policy_seed)
        initial = rx.create_ppo(lib, cfg, str(scratch / f"{key}_initial.ot"), None)
        initial.save()
        initial.close()
        agent = rx.create_ppo(lib, cfg, None, str(checkpoint))
        episodes, observations, means = [], [], []
        for stochastic in (False, True):
            env = gym.make(spec["env_id"])
            observation, _ = env.reset(seed=910000)
            total, clips, components, steps = 0.0, 0, 0, 0
            rx.manual_seed(lib, 910000)
            while True:
                if not stochastic and steps % 8 == 0 and len(observations) < 128:
                    observations.append(np.asarray(observation, dtype=np.float32).tolist())
                    means.append(np.asarray(agent.act(observation)).reshape(-1).tolist())
                action = agent.act_and_train(observation, 0.0) if stochastic else agent.act(observation)
                action = rx.gym_action(agent, action, env.action_space)
                clips += int(np.sum(np.abs(action) >= 1.0))
                components += action.size
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    if stochastic:
                        agent.stop_episode(observation, 0.0, terminated=bool(terminated))
                    break
            env.close()
            episodes.append({"mode": "sampled" if stochastic else "mean", "seed": 910000,
                             "return": total, "length": steps,
                             "executed_action_boundary_fraction": clips / components})
        # Actual FFI samples at a fixed, representative subset of the same observations.
        probes = observations[::max(1, len(observations) // 16)][:16]
        clips = total = 0
        for observation in probes:
            for _ in range(256):
                action = np.asarray(agent.act_and_train(observation, 0.0))
                clips += int(np.sum(np.abs(action) >= 1.0))
                total += action.size
        statistics = agent.statistics()
        assert statistics.get("updates") == 0, statistics
        agent.close()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        result[key] = {"case": case, "worker": worker, "training_seed": training_seed,
                        "policy_initialization_seed": policy_seed, "library_sha256": library_sha256,
                        "checkpoint": str(checkpoint.relative_to(ROOT)), "sha256": digest,
                        "episodes": episodes, "observations": observations, "native_means": means,
                        "probe_observations": probes, "sampled_probe_component_count": total,
                        "sampled_probe_boundary_fraction": clips / total, "updates": 0,
                        "min_variance": float(cfg.min_variance)}
    (scratch / "native.json").write_text(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--case", help="One native case; otherwise probe the original Walker/Ant pair")
    parser.add_argument("--workers", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        help="Required with --case to preserve the original Walker/Ant diagnostic artifact")
    parser.add_argument("--native-python", default="/private/tmp/reinforcex-gym-smoke/bin/python")
    parser.add_argument("--native-libtorch", default=str(
        ROOT / "target/debug/build/torch-sys-c1854a431246b133/out/libtorch/libtorch/lib"))
    args = parser.parse_args()
    if args.case and not args.output and not args.native:
        parser.error("--case requires a separate --output path")
    if any(worker < 0 for worker in args.workers) or len(set(args.workers)) != len(args.workers):
        parser.error("--workers must contain distinct nonnegative indices")
    args.output = args.output or ROOT / "reports/oss_benchmarks/continuous_ppo_diagnostics.json"
    if args.native:
        native_probe(args.native, args.case, args.workers, args.seed)
        return
    import torch
    torch.set_num_threads(1)

    def distribution(path, observations, minimum):
        state = torch.jit.load(str(path), map_location="cpu").state_dict()
        x = torch.tensor(observations, dtype=torch.float32)
        # Exact constructor order in FCGaussianPolicyWithValue: value head,
        # two shared hidden layers, action mean head, scalar variance head.
        assert state["weight__5"].shape == (256, 256)
        h = torch.nn.functional.linear(x, state["weight__3"], state["bias__2"]).relu()
        h = torch.nn.functional.linear(h, state["weight__5"], state["bias__4"]).relu()
        mean = torch.nn.functional.linear(h, state["weight__7"], state["bias__6"]).tanh()
        var = torch.nn.functional.softplus(
            torch.nn.functional.linear(h, state["weight__9"], state["bias__8"])) + minimum
        std = var.sqrt().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        clipping = normal.cdf(torch.full_like(mean, -1.0)) + 1 - normal.cdf(torch.ones_like(mean))
        stats = {"std_min": std.min().item(), "std_median": std.median().item(),
                 "std_max": std.max().item(), "std_geometric_mean": std.log().mean().exp().item(),
                 "mean_abs_gt_0_95_fraction": (mean.abs() > 0.95).float().mean().item(),
                 "expected_action_boundary_fraction": clipping.mean().item(),
                 "raw_gaussian_entropy_mean": normal.entropy().sum(-1).mean().item()}
        return mean, stats

    with tempfile.TemporaryDirectory(prefix="reinforcex-ppo-audit-") as temporary:
        scratch = Path(temporary)
        child_env = os.environ.copy()
        child_env["DYLD_LIBRARY_PATH"] = args.native_libtorch
        selection_args = ["--seed", str(args.seed), "--workers", *map(str, args.workers)]
        if args.case:
            selection_args += ["--case", args.case]
        subprocess.run([args.native_python, str(Path(__file__)), "--native", str(scratch), *selection_args],
                       check=True, env=child_env, cwd=ROOT)
        cases = json.loads((scratch / "native.json").read_text())
        for case, record in cases.items():
            observations = record.pop("observations")
            native_means = record.pop("native_means")
            probes = record.pop("probe_observations")
            final_path = ROOT / record["checkpoint"]
            with torch.no_grad():
                mean, record["final_distribution_on_greedy_trajectory"] = distribution(
                    final_path, observations, record["min_variance"])
                record["reconstruction_max_action_error"] = float(
                    (mean - torch.tensor(native_means)).abs().max())
                assert record["reconstruction_max_action_error"] < 1e-5
                _, record["final_distribution_on_probe_states"] = distribution(
                    final_path, probes, record["min_variance"])
                _, record["initial_distribution_on_same_greedy_states"] = distribution(
                    scratch / f"{case}_initial.ot", observations, record["min_variance"])
            record["greedy_trajectory_observation_count"] = len(observations)
        output = {"scope": f"seed{args.seed} final checkpoints only; no optimizer updates; separate processes",
                  "limitations": "One diagnostic episode per mode; not a performance estimate. Probe states follow final greedy policy; not training-state distribution.",
                  "cases": cases}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
