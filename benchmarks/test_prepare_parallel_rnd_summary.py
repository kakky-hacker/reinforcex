"""Analytic tests only; never invoke a learner, audit or campaign finalizer."""
import math
import unittest

from prepare_parallel_rnd_summary import aggregate_group, paired_comparison, seed_summary


def run(seed, means):
    return {"seed": seed, "aggregate_steps": 1000,
            "effective_agents": [{"algorithm": "ppo"} for _ in means],
            "workers": [{"algorithm": "ppo", "final_mean": value, "steps": 1000 // len(means),
                         "recorded_updates": 5, "policy_optimizer_steps": 20,
                         "policy_sample_presentations": 100} for value in means]}


class ParallelSummaryTests(unittest.TestCase):
    def test_workers_are_equal_weighted_but_not_independent_seeds(self):
        runs = [run(42, [100, 300]), run(123, [200, 400]), run(2026, [300, 500])]
        group = aggregate_group("example", "ppo", runs, runs, 200)
        self.assertEqual(group["seed_values"], {42: 200, 123: 300, 2026: 400})
        self.assertEqual(group["n"], 3)
        self.assertEqual(group["mean"], 300)
        self.assertEqual(group["sample_sd"], 100)
        self.assertAlmostEqual(group["ci95"][0], 300 - 4.302652729911275 * 100 / math.sqrt(3))
        self.assertEqual(group["classification"], "all_seeds_reached")
        self.assertEqual(group["worker_models_at_threshold"], 5)
        self.assertEqual(group["expected_worker_models"], 6)
        self.assertEqual(group["lowest_worker_mean"], 100)
        self.assertEqual(group["per_seed_counters"][0]["recorded_updates_sum"], 10)

    def test_missing_seed_does_not_turn_survivors_into_final_mean(self):
        expected = [run(42, [100]), run(123, [200]), run(2026, [300])]
        group = aggregate_group("example", "ppo", expected, expected[:2], 50)
        self.assertEqual(group["classification"], "pending")
        self.assertIsNone(group["mean"])
        self.assertIsNone(group["ci95"])
        self.assertEqual(group["seed_values"], {42: 100, 123: 200})
        self.assertEqual(group["expected_worker_models"], 3)
        self.assertEqual(group["observed_worker_models"], 2)

    def test_paired_ci_uses_seed_differences_not_unpaired_sd(self):
        left = {"condition": "a", "algorithm": "ppo", **seed_summary({42: 100, 123: 200, 2026: 300})}
        right = {"condition": "b", "algorithm": "ppo", **seed_summary({42: 90, 123: 180, 2026: 270})}
        result = paired_comparison(left, right)
        self.assertEqual(result["seed_values"], {42: 10, 123: 20, 2026: 30})
        self.assertEqual(result["mean"], 20)
        self.assertEqual(result["sample_sd"], 10)

    def test_same_seed_is_required_for_a_pair(self):
        left = {"condition": "a", "algorithm": "ppo", **seed_summary({42: 100, 123: 200})}
        right = {"condition": "b", "algorithm": "ppo", **seed_summary({123: 190, 2026: 300})}
        result = paired_comparison(left, right)
        self.assertEqual(result["seed_values"], {123: 10})
        self.assertIsNone(result["mean"])


if __name__ == "__main__":
    unittest.main()
