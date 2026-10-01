"""Regression cases for temporal diagnostics, never best-checkpoint selection."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks import analyze_stability as stability


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(configs_dir=self.root / "configs", expected_checkpoints=2,
                                       validation_episodes=10, validation_seed=800000,
                                       test_episodes=100, test_seed=900000)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    @staticmethod
    def jsonl(path, records):
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def native(self, test_mean=290):
        path = self.root / "runs/native/cartpole_dqn/seed_42"
        self.write(path / "metadata.json", {"case": "cartpole_dqn", "seed": 42,
                                             "env_id": "CartPole-v1", "requested_total_steps": 1000,
                                             "effective_agents": [{"algorithm": "dqn"}]})
        points = [{"worker": 0, "algorithm": "dqn", "aggregate_steps": step,
                   "split": "validation", "mean": mean, "returns": [mean] * 10,
                   "episodes": 10, "seed_start": 800000, "deterministic": True, "reward": "raw"}
                  for step, mean in [(0, 10), (500, 500), (1000, 300)]]
        test = {**points[-1], "split": "test", "mean": test_mean, "returns": [test_mean] * 100,
                "episodes": 100, "seed_start": 900000}
        self.jsonl(path / "evaluations.jsonl", points + [test])
        self.write(path / "final.json", {"status": "complete", "seed": 42, "env_id": "CartPole-v1",
                                         "actual_total_steps": 1000, "test": [test]})
        job = {"path": path, "backend": "native", "condition": "cartpole_dqn", "seed": 42}
        return job

    def test_validation_peak_last_and_final_are_separate(self):
        run, rows = stability.analyze_run(self.native(), self.args)
        row = rows[0]
        self.assertTrue(run["diagnostics_complete"])
        self.assertEqual((row["peak_mean"], row["peak_step"], row["last_mean"], row["peak_minus_last"]),
                         (500, 500, 300, 200))
        self.assertEqual(row["first_threshold_step"], 500)
        self.assertTrue(row["threshold_reached_then_last_validation_below"])
        self.assertTrue(row["threshold_reached_then_final_test_below"])
        self.assertEqual(row["final_test_mean"], 290)

    def test_high_final_test_cannot_become_validation_peak(self):
        _, rows = stability.analyze_run(self.native(test_mean=1000), self.args)
        self.assertEqual(rows[0]["peak_mean"], 500)
        self.assertFalse(rows[0]["threshold_reached_then_final_test_below"])

    def test_tied_peak_selects_earliest_observed_step_including_zero(self):
        points = [{"aggregate_steps": step, "mean": mean} for step, mean in [(100, 300), (50, 500), (0, 500)]]
        result = stability.validation_diagnostics(points, 475)
        self.assertEqual(result["peak_step"], 0)
        self.assertEqual(result["first_threshold_step"], 0)
        self.assertEqual(result["initial_mean"], 500)

    def test_pending_run_reports_provisional_validation_without_test_verdict(self):
        job = self.native()
        (job["path"] / "final.json").unlink()
        run, rows = stability.analyze_run(job, self.args)
        self.assertEqual(run["run_status"], "pending")
        self.assertEqual(rows[0]["last_mean"], 300)
        self.assertIsNone(rows[0]["final_test_mean"])
        self.assertIsNone(rows[0]["threshold_reached_then_final_test_below"])

    def test_wrong_validation_seed_is_excluded_from_the_peak(self):
        job = self.native()
        points = [json.loads(line) for line in (job["path"] / "evaluations.jsonl").read_text().splitlines()]
        points[1]["seed_start"] = 7
        self.jsonl(job["path"] / "evaluations.jsonl", points)
        run, rows = stability.analyze_run(job, self.args)
        self.assertFalse(run["diagnostics_complete"])
        self.assertEqual(rows[0]["diagnostics_status"], "invalid")
        self.assertEqual(rows[0]["peak_mean"], 300)
        self.assertEqual(len(rows[0]["invalid_validation_points"]), 1)

    def test_no_official_threshold_has_no_crossing_verdict(self):
        point = {"aggregate_steps": 0, "mean": 10000}
        result = stability.validation_diagnostics([point], None)
        self.assertIsNone(result["first_threshold_step"])
        self.assertIsNone(result["threshold_reached_then_last_validation_below"])

    def test_sb3_last_validation_before_rollout_overshoot_is_complete(self):
        path = self.root / "runs/sb3/cartpole_ppo/seed_42"
        self.write(path / "metadata.json", {"status": "complete"})
        self.write(path / "config.json", {"arguments": {"env": "CartPole-v1", "algo": "ppo", "seed": 42, "steps": 1000}})
        points = [{"point": index, "phase": "initial" if index == 1 else "progress",
                   "env_steps": step, "mean_return": mean, "episodes": 10,
                   "seed_start": 800000, "deterministic": True}
                  for index, (step, mean) in enumerate([(0, 10), (500, 500), (1000, 300)], 1)]
        final_point = {**points[-1], "point": 4, "phase": "final", "env_steps": 1024,
                       "mean_return": 290, "episodes": 100, "seed_start": 900000}
        self.jsonl(path / "evaluations.jsonl", points + [final_point])
        raw = [{"point": point["point"], "phase": point["phase"], "seed": seed, "return": point["mean_return"]}
               for point in points + [final_point]
               for seed in range(point["seed_start"], point["seed_start"] + point["episodes"])]
        self.jsonl(path / "eval_episodes.jsonl", raw)
        self.write(path / "final.json", {"seed": 42, "algorithm": "ppo", "environment": "CartPole-v1",
                                         "requested_steps": 1000, "actual_steps": 1024,
                                         "final_evaluation": final_point})
        job = {"path": path, "backend": "sb3", "condition": "cartpole_ppo", "seed": 42}
        run, rows = stability.analyze_run(job, self.args)
        self.assertTrue(run["diagnostics_complete"])
        self.assertEqual(rows[0]["steps_after_last_validation"], 24)
        self.assertTrue(rows[0]["threshold_reached_then_final_test_below"])
        self.assertEqual(rows[0]["peak_minus_last"], 200)


if __name__ == "__main__":
    unittest.main()
