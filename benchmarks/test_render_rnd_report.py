"""Analytical fixtures for RND trace plots and module/worker accounting."""
import importlib.util
import copy
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("render_rnd_report", Path(__file__).with_name("render_rnd_report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class RndReportTests(unittest.TestCase):
    @staticmethod
    def point(checkpoint, raw, coefficient=.01):
        return {"checkpoint": checkpoint, "aggregate_steps": checkpoint * 100,
                "worker_steps": checkpoint * 50, "updates": checkpoint,
                "last_rollout_intrinsic_mean": raw, "coefficient": coefficient}

    def fixture(self):
        return {"results": [
            {"run": "runs/native/lunar_rnd_shared2/seed_42", "status": "audited", "errors": [],
             "expected_modules": 1, "actual_total_steps": 1000,
             "modules": [{"owner_worker": 0, "shared_by_workers": [0, 1],
                          "target_unchanged": True, "predictor_changed": True}],
             "worker_traces": [
                 {"worker": 0, "missing_early_checkpoints": True,
                  "points": [self.point(10, 2.0)],
                  "final_statistics": {"updates": 5, "intrinsic_reward_mean": 3., "curiosity_coefficient": .01}},
                 {"worker": 1, "missing_early_checkpoints": False,
                  "points": [self.point(1, 7., 0), self.point(3, 5., .1)],
                  "final_statistics": {"updates": 5, "intrinsic_reward_mean": 4., "curiosity_coefficient": .1}},
             ]},
            {"run": "runs/native/lunar_rnd_shared2/seed_123", "status": "not_started", "expected_modules": 1},
            {"run": "runs/native/lunar_rnd_shared2/seed_2026", "status": "in_progress", "expected_modules": 1},
        ]}

    def test_shared_module_is_counted_once_and_workers_remain_separate(self):
        groups, traces, finals = report.prepare(self.fixture())
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["expected_modules"], 3)
        self.assertEqual(groups[0]["audited_modules"], 1)
        self.assertEqual(groups[0]["targets_unchanged"], 1)
        self.assertEqual(groups[0]["predictors_changed"], 1)
        self.assertEqual(groups[0]["status"], "PENDING")
        self.assertEqual(len(groups[0]["pending_runs"]), 2)
        self.assertEqual([t["worker"] for t in traces], [0, 1])
        self.assertEqual([f["intrinsic_reward_mean"] for f in finals], [3., 4.])

    def test_three_training_seeds_keep_six_separate_worker_traces(self):
        completed = self.fixture()["results"][0]
        source = {"results": []}
        for seed in (2026, 42, 123):
            run = copy.deepcopy(completed)
            run["run"] = f"runs/native/lunar_rnd_shared2/seed_{seed}"
            source["results"].append(run)
        groups, traces, finals = report.prepare(source)
        self.assertEqual(groups[0]["audited_modules"], 3)
        self.assertEqual(groups[0]["status"], "AUDITED")
        self.assertEqual([(t["seed"], t["worker"]) for t in traces],
                         [(42, 0), (42, 1), (123, 0), (123, 1), (2026, 0), (2026, 1)])
        self.assertEqual(len(finals), 6)

    def test_coefficient_is_applied_per_point_including_zero(self):
        _, traces, _ = report.prepare(self.fixture())
        self.assertAlmostEqual(traces[0]["points"][0]["weighted_intrinsic_reward_mean"], .02)
        self.assertEqual(traces[1]["points"][0]["weighted_intrinsic_reward_mean"], 0)
        self.assertEqual(traces[1]["points"][1]["weighted_intrinsic_reward_mean"], .5)

    def test_missing_checkpoint_breaks_line(self):
        _, traces, _ = report.prepare(self.fixture())
        pieces = report.segments(traces[1]["points"], "intrinsic_reward_mean")
        self.assertEqual([[p["checkpoint"] for p in piece] for piece in pieces], [[1], [3]])

    def test_invalid_value_breaks_line_without_filling(self):
        points = [{"checkpoint": n, "aggregate_steps": n * 100, "value": value}
                  for n, value in [(1, 1), (2, None), (3, 3), (4, 4)]]
        pieces = report.segments(points, "value")
        self.assertEqual([[p["checkpoint"] for p in piece] for piece in pieces], [[1], [3, 4]])

    def test_final_statistic_does_not_fill_missing_plot_point(self):
        _, traces, finals = report.prepare(self.fixture())
        self.assertEqual(len(traces[0]["points"]), 1)
        self.assertEqual(traces[0]["points"][0]["intrinsic_reward_mean"], 2.)
        self.assertEqual(finals[0]["intrinsic_reward_mean"], 3.)
        self.assertAlmostEqual(finals[0]["weighted_intrinsic_reward_mean"], .03)

    def test_bad_module_marks_group_failed(self):
        source = self.fixture()
        source["results"][0]["modules"][0]["predictor_changed"] = False
        groups, _, _ = report.prepare(source)
        self.assertEqual(groups[0]["status"], "FAILED")
        self.assertEqual(groups[0]["failed_modules"], 1)

    def test_duplicate_checkpoint_is_rejected(self):
        source = self.fixture()
        source["results"][0]["worker_traces"][0]["points"].append(self.point(10, 9))
        with self.assertRaises(ValueError):
            report.prepare(source)


if __name__ == "__main__":
    unittest.main()
