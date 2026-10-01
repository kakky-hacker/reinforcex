"""Analytical regression fixtures for benchmark evidence aggregation."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("render_oss_report", Path(__file__).with_name("render_oss_report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.configs = self.root / "configs"
        self.configs.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    @staticmethod
    def lines(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def native(self, seed, means, condition="cartpole_dqn", env="CartPole-v1"):
        path = self.runs / "native" / condition / f"seed_{seed}"
        metadata = {"backend": "reinforcex", "case": condition, "seed": seed,
                    "env_id": env, "requested_total_steps": 1000,
                    "effective_agents": [{"algorithm": "dqn"} for _ in means],
                    "worker_count": len(means), "max_rss_unit": "bytes"}
        self.write(path / "metadata.json", metadata)
        tests = [{"worker": worker, "algorithm": "dqn", "aggregate_steps": 1000,
                  "split": "test", "mean": mean, "returns": [mean] * 100,
                  "episodes": 100, "seed_start": 900000, "deterministic": True, "reward": "raw"}
                 for worker, mean in enumerate(means)]
        workers = [{"worker": worker, "algorithm": "dqn", "steps": 1000 // len(means),
                    "episodes": 101, "statistics": {"updates": 12.0}}
                   for worker in range(len(means))]
        self.write(path / "final.json", {"status": "complete", "seed": seed, "env_id": env,
                                         "actual_total_steps": 1000, "test": tests, "workers": workers})
        self.lines(path / "evaluations.jsonl", [
            {**test, "split": "validation", "aggregate_steps": 500, "mean": 99999}
            for test in tests] + tests)
        for worker in range(len(means)):
            train = [{"worker": worker, "algorithm": "dqn", "episode": i,
                      "steps": i, "aggregate_steps": i * len(means), "length": 1,
                      "reward": float(i), "learning_reward": -1, "terminated": True,
                      "truncated": False, "budget_cut": False} for i in range(1, 102)]
            train.append({**train[-1], "episode": 102, "reward": 1e9, "budget_cut": True})
            self.lines(path / f"train_worker{worker}.jsonl", train)
        return path

    def normalized(self, path):
        job = {"path": path, "backend": path.parent.parent.name,
               "condition": path.parent.name, "seed": int(path.name.split("_")[1])}
        return report.normalize_run(job, self.configs, 100, 900000, 100)

    def test_equal_worker_seed_summary_and_no_best_checkpoint_selection(self):
        paths = [self.native(seed, means) for seed, means in [
            (42, [400, 600]), (123, [450, 650]), (2026, [300, 500])]]
        runs = [self.normalized(path)[0] for path in paths]
        self.assertTrue(all(run["valid_final"] for run in runs))
        group = report.aggregate(runs, [42, 123, 2026])[0]
        self.assertEqual(group["seed_means"], {"42": 500, "123": 550, "2026": 400})
        self.assertAlmostEqual(group["mean"], 1450 / 3)
        self.assertAlmostEqual(group["sd"], statistics.stdev([500, 550, 400]))
        radius = 4.302652729911275 * statistics.stdev([500, 550, 400]) / math.sqrt(3)
        self.assertAlmostEqual(group["ci95"][0], 1450 / 3 - radius)
        self.assertEqual(group["criterion"], "mean_only_pass")
        self.assertFalse(group["ci95_lower_meets_threshold"])
        self.assertEqual(group["worker_models_passing_threshold"], 3)
        self.assertEqual(group["total_worker_models"], 6)
        self.assertEqual(group["expected_total_worker_models"], 6)
        self.assertEqual(group["minimum_worker_evaluation_mean"], 300)

    def test_all_seed_pass_does_not_imply_all_workers_pass(self):
        runs = [self.normalized(self.native(seed, [400, 600]))[0] for seed in [42, 123, 2026]]
        group = report.aggregate(runs, [42, 123, 2026])[0]
        self.assertEqual(group["criterion"], "all_seeds_pass")
        self.assertEqual(group["worker_models_passing_threshold"], 3)
        self.assertFalse(group["all_worker_models_pass"])

    def test_incomplete_seed_does_not_become_a_pass(self):
        run = self.normalized(self.native(42, [500]))[0]
        group = report.aggregate([run], [42, 123, 2026])[0]
        self.assertEqual(group["criterion"], "incomplete")
        self.assertEqual(group["missing_seeds"], [123, 2026])
        self.assertIsNone(group["sd"])
        self.assertIsNone(group["ci95"])

    def test_budget_cut_never_contaminates_ma100(self):
        run, train, _ = self.normalized(self.native(42, [100]))
        self.assertEqual(run["completed_episodes"], 101)
        self.assertEqual(run["budget_cut_episodes"], 1)
        self.assertIsNone(train[98]["ma_return"])
        self.assertEqual(train[99]["ma_return"], 50.5)
        self.assertEqual(train[100]["ma_return"], 51.5)
        self.assertIsNone(train[101]["ma_return"])

    def test_missing_worker_or_wrong_test_seed_rejects_final(self):
        path = self.native(42, [500, 500])
        final = json.loads((path / "final.json").read_text())
        final["test"] = final["test"][:1]
        final["test"][0]["seed_start"] = 800000
        self.write(path / "final.json", final)
        run, _, points = self.normalized(path)
        self.assertFalse(run["valid_final"])
        self.assertTrue(any("worker set" in error for error in run["errors"]))
        self.assertFalse(any(point["split"] == "test" for point in report.evaluation_curves([run], points, [42])))

    def test_nonfinite_native_stat_is_rejected_and_serializable(self):
        path = self.native(42, [500])
        final = json.loads((path / "final.json").read_text())
        final["workers"][0]["statistics"]["loss"] = float("nan")
        self.write(path / "final.json", final)
        run, _, _ = self.normalized(path)
        self.assertFalse(run["valid_final"])
        self.assertTrue(any("nonfinite" in error for error in run["errors"]))
        json.dumps(run, allow_nan=False)

    def test_failure_json_prevents_final_success(self):
        path = self.native(42, [500])
        self.write(path / "failure.json", {"traceback": "synthetic failure"})
        run = self.normalized(path)[0]
        self.assertEqual(run["status"], "failed")
        self.assertFalse(run["valid_final"])

    def test_walker_auxiliary_gate_requires_complete_positive_reference(self):
        def group(backend, means):
            return {"backend": backend, "condition": "walker_ppo", "algorithm": "ppo",
                    "complete": True, "official_threshold": None,
                    "mean": statistics.fmean(means),
                    "seed_means": dict(zip(["42", "123", "2026"], means))}
        native, sb3 = group("native", [900, 850, 750]), group("sb3", [1000] * 3)
        report.add_comparisons([native, sb3], [42, 123, 2026])
        self.assertEqual(native["reference"]["auxiliary_threshold"], 800)
        self.assertEqual(native["reference"]["auxiliary_criterion"], "mean_only_pass")
        sb3["complete"] = False
        report.add_comparisons([native, sb3], [42, 123, 2026])
        self.assertEqual(native["reference"]["auxiliary_criterion"], "unavailable")

    def test_missing_manifest_run_is_reported(self):
        manifest = self.root / "native_manifest.json"
        path = self.runs / "native/cartpole_dqn/seed_42"
        self.write(manifest, {"jobs": [{"output": str(path), "seed": 42, "case": "cartpole_dqn"}]})
        self.write(self.configs / "cartpole_dqn.json", {"env_id": "CartPole-v1", "agents": [{"algorithm": "dqn"}]})
        errors = []
        jobs = report.discover(self.runs, [manifest], errors)
        self.assertEqual(len(jobs), 1)
        run = report.normalize_run(jobs[0], self.configs, 100, 900000, 100)[0]
        self.assertEqual(run["status"], "missing")
        self.assertFalse(report.aggregate([run], [42, 123, 2026])[0]["complete"])

    def test_sb3_final_requires_the_complete_raw_episode_log(self):
        path = self.runs / "sb3/cartpole_ppo/seed_42"
        self.write(path / "metadata.json", {"status": "complete"})
        self.write(path / "config.json", {"arguments": {"algo": "ppo", "env": "CartPole-v1", "steps": 1000, "seed": 42}})
        point = {"phase": "final", "env_steps": 1024, "mean_return": 500,
                 "seed_start": 900000, "episodes": 100, "deterministic": True}
        self.write(path / "final.json", {"algorithm": "ppo", "environment": "CartPole-v1", "seed": 42,
                                         "actual_steps": 1024, "requested_steps": 1000, "completed_episodes": 0,
                                         "final_evaluation": point})
        self.lines(path / "eval_episodes.jsonl", [{"phase": "final", "return": 500, "seed": seed}
                                                 for seed in range(900000, 900100)])
        self.assertTrue(self.normalized(path)[0]["valid_final"])
        self.lines(path / "eval_episodes.jsonl", [{"phase": "final", "return": 500, "seed": 900000}])
        self.assertFalse(self.normalized(path)[0]["valid_final"])

    def test_report_generation_keeps_all_episode_records(self):
        self.native(42, [300, 600])
        self.native(123, [500, 500])
        self.native(2026, [500, 500])
        output = self.root / "output"
        args = argparse.Namespace(output=output, runs_root=self.runs, configs_dir=self.configs,
                                  manifest=[], test_episodes=100, test_seed=900000,
                                  ma_window=100, seeds=[42, 123, 2026], no_plots=True)
        result = report.render(args)
        self.assertEqual(len(result["groups"]), 1)
        self.assertTrue(result["groups"][0]["complete"])
        self.assertEqual(len((output / "training_episodes.csv").read_text().splitlines()), 6 * 102 + 1)
        self.assertTrue((output / "REPORT.ja.md").exists())


if __name__ == "__main__":
    unittest.main()
