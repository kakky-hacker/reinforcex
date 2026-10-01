"""Protect fixed final-test membership and model-level distribution semantics."""
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

import analyze_evaluation_distribution as distribution


class DistributionTests(unittest.TestCase):
    def test_quantiles_and_sample_sd_have_explicit_conventions(self):
        result = distribution.describe(list(range(1, 101)), 95)
        self.assertEqual(result["mean"], 50.5)
        self.assertAlmostEqual(result["sd"], math.sqrt(100 * 101 / 12))
        self.assertAlmostEqual(result["p05"], 5.95)
        self.assertEqual(result["median"], 50.5)
        self.assertAlmostEqual(result["p95"], 95.05)
        self.assertEqual(result["episodes_at_or_above_registered_threshold"], 6)
        self.assertNotIn("passed", result)
        self.assertNotIn("ci95", result)

    def test_negative_count_is_strict_and_no_threshold_remains_null(self):
        result = distribution.describe(list(range(-50, 50)), None)
        self.assertEqual(result["return_below_zero_count"], 50)
        self.assertIsNone(result["episodes_at_or_above_registered_threshold"])
        self.assertEqual(result["min"], -50)
        self.assertEqual(result["max"], 49)

    def test_missing_or_nonfinite_returns_cannot_make_a_distribution(self):
        for values in [[1.] * 99, [1.] * 99 + [float("nan")], [1.] * 99 + [float("inf")]]:
            with self.assertRaises(ValueError):
                distribution.describe(values, 200)

    def make_reference(self, path, returns):
        records = [{"phase": "final", "point": 12, "env_steps": 1000, "seed": 900000 + i,
                    "return": value, "length": 100} for i, value in enumerate(returns)]
        validation = [{"phase": "progress", "point": 11, "env_steps": 1000, "seed": 800000 + i,
                       "return": 999999, "length": 100} for i in range(10)]
        (path / "eval_episodes.jsonl").write_text("\n".join(json.dumps(row) for row in list(reversed(records)) + validation))
        return {"final_evaluation": {"point": 12, "phase": "final", "env_steps": 1000,
                                    "seed_start": 900000, "episodes": 100, "deterministic": True,
                                    "mean_return": statistics.fmean(returns)}}

    def test_sb3_and_tianshou_select_final_seeds_even_at_same_validation_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            final = self.make_reference(path, list(range(100)))
            for backend in ["sb3", "tianshou", "tianshou_dqn"]:
                errors = []
                returns, lengths = distribution.final_episodes(backend, path, final, 0, "dqn", 1000, errors, {})
                self.assertEqual(errors, [])
                self.assertEqual(returns, list(range(100)))
                self.assertEqual(lengths, [100] * 100)
            log = path / "eval_episodes.jsonl"
            records = [json.loads(line) for line in log.read_text().splitlines()]
            records[0]["seed"] = records[1]["seed"]
            log.write_text("\n".join(json.dumps(row) for row in records))
            errors = []
            distribution.final_episodes("tianshou", path, final, 0, "sac", 1000, errors, {})
            self.assertTrue(any("seed set" in error for error in errors))

    def native_fixture(self, base, complete=True):
        path = base / "run"
        path.mkdir()
        (base / "configs").mkdir()
        config = {"env_id": "Example-v1", "agents": [{"algorithm": "ppo", "rnd_config": None}]}
        (base / "configs/example.json").write_text(json.dumps(config))
        job = {"path": path, "backend": "native", "condition": "parallel2", "case": "example",
               "seed": 42, "steps": 1000, "command": ["python", "script", "--workers", "2"]}
        if complete:
            tests = [{"worker": worker, "algorithm": "ppo", "split": "test", "reward": "raw",
                      "deterministic": True, "episodes": 100, "seed_start": 900000,
                      "aggregate_steps": 1000, "returns": [value] * 100, "lengths": [100] * 100,
                      "mean": value} for worker, value in enumerate([-100., 100.])]
            (path / "final.json").write_text(json.dumps({"status": "complete", "seed": 42,
                                                        "actual_total_steps": 1000, "test": tests}))
        return job

    def test_parallel_workers_are_described_separately_not_pooled(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            job = self.native_fixture(base)
            run, rows = distribution.analyze_run(job, base, {"Example-v1": 50}, {})
            self.assertEqual(run["status"], "complete")
            self.assertEqual([row["mean"] for row in rows], [-100, 100])
            self.assertEqual([row["sd"] for row in rows], [0, 0])
            self.assertEqual([row["return_below_zero_count"] for row in rows], [100, 0])
            self.assertEqual([row["episodes_at_or_above_registered_threshold"] for row in rows], [0, 100])

    def test_missing_final_retains_all_expected_workers_as_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            job = self.native_fixture(base, complete=False)
            run, rows = distribution.analyze_run(job, base, {"Example-v1": None}, {})
            self.assertEqual(run["status"], "pending")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["status"] == "pending" and row["mean"] is None for row in rows))

    def test_missing_reference_results_keep_supplementary_algorithm_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            job = self.native_fixture(base, complete=False)
            for backend, algorithm in [("tianshou", "discrete_sac"), ("tianshou_dqn", "double_dqn")]:
                _, rows = distribution.analyze_run({**job, "backend": backend}, base, {"Example-v1": None}, {})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "pending")
                self.assertEqual(rows[0]["algorithm"], algorithm)

    def test_completed_but_malformed_worker_is_invalid_without_partial_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            job = self.native_fixture(base)
            path = job["path"] / "final.json"
            final = json.loads(path.read_text())
            final["test"][1]["returns"] = [100.] * 99
            path.write_text(json.dumps(final))
            run, rows = distribution.analyze_run(job, base, {"Example-v1": 50}, {})
            self.assertEqual(run["status"], "invalid")
            self.assertEqual(rows[0]["status"], "complete")
            self.assertEqual(rows[1]["status"], "invalid")
            self.assertIsNone(rows[1]["mean"])
            self.assertEqual(rows[1]["evaluation_returns"], [])


if __name__ == "__main__":
    unittest.main()
