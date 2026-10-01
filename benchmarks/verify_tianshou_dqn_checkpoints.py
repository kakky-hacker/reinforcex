#!/usr/bin/env python3
"""Reproduce final Double DQN held-out returns directly from reloaded Q networks."""
import argparse
import json
from pathlib import Path

import run_tianshou_dqn as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/oss_benchmarks/tianshou_dqn_checkpoint_audit.json"))
    args = parser.parse_args()
    runner.torch.set_num_threads(1)
    runner.torch.set_num_interop_threads(1)
    manifest = json.loads(Path("reports/oss_benchmarks/tianshou_dqn_manifest.json").read_text())
    results = []
    for job in manifest["jobs"]:
        root = Path(job["output"])
        if not (root / "final.json").exists():
            results.append({"id": job["id"], "status": "pending"})
            continue
        config = json.loads((root / "config.json").read_text())
        final = json.loads((root / "final.json").read_text())
        episodes = [json.loads(line) for line in (root / "eval_episodes.jsonl").read_text().splitlines()]
        expected = [e for e in episodes if e["phase"] == "final"]
        env = runner.gym.make(config["env_id"])
        model, _ = runner.build_algorithm(config["source_config"], env)
        model.load_state_dict(runner.torch.load(root / "final_model.pt", map_location="cpu", weights_only=True))
        model.eval()
        returns, mismatches = [], []
        with runner.torch.no_grad():
            for episode in expected:
                obs, _ = env.reset(seed=episode["seed"])
                total, length = 0.0, 0
                while True:
                    q_values, _ = model.policy.model(runner.np.asarray([obs]))
                    action = int(q_values.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    length += 1
                    if terminated or truncated:
                        break
                returns.append(total)
                if total != episode["return"] or length != episode["length"]:
                    mismatches.append({"seed": episode["seed"], "actual_return": total,
                                       "saved_return": episode["return"], "actual_length": length,
                                       "saved_length": episode["length"]})
        env.close()
        passed = (not mismatches and len(returns) == 100 and
                  float(runner.np.mean(returns)) == final["final_evaluation"]["mean_return"])
        result = {"id": job["id"], "status": "passed" if passed else "failed", "episodes": len(returns),
                  "mean_return": float(runner.np.mean(returns)),
                  "checkpoint_sha256": runner.sha256(root / "final_model.pt"), "mismatches": mismatches}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"status": "failed" if any(r["status"] == "failed" for r in results) else
              "incomplete" if any(r["status"] == "pending" for r in results) else "passed",
              "method": "Reload final weights; direct Q-network argmax, separate Gymnasium environment, fixed held-out episode seeds.",
              "runs": results}
    runner.write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
