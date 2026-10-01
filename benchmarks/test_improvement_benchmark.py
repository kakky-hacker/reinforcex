"""Protocol regressions using fake environments/agents; no native library or learning."""
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import improvement_configs as configs
import improvement_campaign as campaign
import improvement_run as runner
import numpy as np


def write(path, value):
    path.write_text(json.dumps(value))


def write_lines(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class FakeAgent:
    def __init__(self):
        self.steps = 0
        self.rewards = []
        self.stops = []

    def act(self, obs):
        return np.array([0.0])

    def act_and_train(self, obs, reward):
        self.steps += 1
        self.rewards.append(reward)
        return self.act(obs)

    def stop_episode(self, obs, reward, terminated):
        self.stops.append((reward, terminated))

    def statistics(self):
        return {"t": float(self.steps)}


class FakeEnv:
    def __init__(self, terminal_at=None, limit=20):
        self.spec = SimpleNamespace(max_episode_steps=limit)
        self.action_space = object()
        self.terminal_at = terminal_at
        self.seeds = []
        self.closed = False

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.length = 0
        return np.zeros(1), {}

    def step(self, action):
        self.length += 1
        return (np.zeros(1), 2.0, self.length == self.terminal_at,
                self.length == self.spec.max_episode_steps, {})

    def close(self):
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(runner.rx, "gym_action", side_effect=lambda agent, action, space: action).start()

    def test_evaluation_uses_raw_rewards_deterministic_actions_and_separate_seeds(self):
        env, agent = FakeEnv(terminal_at=3), FakeAgent()
        with patch.object(runner.gym, "make", return_value=env):
            result = runner.evaluate(agent, "fake", 3, 1_100_000)
        self.assertEqual(result["returns"], [6.0] * 3)
        self.assertEqual(result["lengths"], [3] * 3)
        self.assertEqual(env.seeds, [1_100_000, 1_100_001, 1_100_002])
        self.assertEqual(agent.steps, 0)
        self.assertTrue(env.closed)
        self.assertEqual(result["reward"], "raw")

    def test_blocks_preserve_episode_and_budget_cut_is_not_a_terminal(self):
        env, agent = FakeEnv(), FakeAgent()
        counter = {"lock": threading.Lock(), "steps": 0}
        with patch.object(runner.gym, "make", return_value=env):
            worker = runner.Worker(agent, "fake", 1001, "hopper", self.root, 0, "sac", counter)
        try:
            worker.train(2)
            self.assertEqual(env.seeds, [1001])
            self.assertFalse(agent.stops)
            result = worker.train(3, final=True)
            self.assertEqual(result["steps"], 5)
            self.assertEqual(result["episodes"], 0)
            self.assertEqual(counter["steps"], 5)
            self.assertEqual(agent.rewards, [0.0, 1.0, 1.0, 1.0, 1.0])
            self.assertEqual(agent.stops, [(1.0, False)])
            self.assertEqual(env.seeds, [1001])
        finally:
            worker.close()
        episode, = campaign.read_jsonl(self.root / "train_worker0.jsonl")
        self.assertEqual((episode["length"], episode["reward"], episode["learning_reward"]), (5, 10.0, 5.0))
        self.assertTrue(episode["budget_cut"])
        self.assertFalse(episode["terminated"])
        self.assertFalse(episode["truncated"])

    def test_real_terminal_at_budget_is_a_complete_episode_without_reset(self):
        env, agent = FakeEnv(terminal_at=3), FakeAgent()
        with patch.object(runner.gym, "make", return_value=env):
            worker = runner.Worker(agent, "fake", 1001, "raw", self.root, 0, "ppo",
                                   {"lock": threading.Lock(), "steps": 0})
        try:
            result = worker.train(3, final=True)
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(agent.stops, [(2.0, True)])
            self.assertEqual(env.seeds, [1001])
        finally:
            worker.close()
        self.assertFalse(campaign.read_jsonl(self.root / "train_worker0.jsonl")[0]["budget_cut"])

    def test_cartpole_last_step_and_raw_reward_transform(self):
        self.assertEqual(runner.transformed(1.0, True, 499, 500, "cartpole"), -1.0)
        self.assertEqual(runner.transformed(1.0, True, 500, 500, "cartpole"), .01)
        self.assertEqual(runner.transformed(5.0, True, 1, 20, "raw"), 5.0)

    def test_checked_library_never_uses_fallback(self):
        binary = self.root / "selected.dylib"
        binary.write_bytes(b"test-only-not-a-library")
        manifest = {"library_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        with patch.object(runner.C, "CDLL", side_effect=OSError("cannot load")), \
             patch.object(runner.rx, "load_reinforcex") as fallback:
            with self.assertRaises(OSError):
                runner.load_checked_library(binary, manifest)
            fallback.assert_not_called()
        with patch.object(runner.C, "CDLL") as loader:
            with self.assertRaisesRegex(ValueError, "differs"):
                runner.load_checked_library(binary, {"library_sha256": "wrong"})
            loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "fallback loading is disabled"):
            runner.load_checked_library(None, manifest)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = json.loads(configs.BASELINE.read_text())
        write(self.root / "configurations.json", self.baseline)
        self.patch = patch.object(configs, "BASELINE", self.root / "configurations.json")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def override(self, value):
        path = self.root / "overrides.json"
        write(path, value)
        return path

    def test_unknown_and_unrepresentable_ctypes_settings_are_rejected(self):
        for values in ({"learning_rae": .1}, {"batch_size": -1}, {"batch_size": 2 ** 64},
                       {"learning_rate": float("inf")}, {"agent": 42}):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                configs.configuration("cartpole_dqn", self.override({"algorithms": {"dqn": values}}))

    def test_reconstructed_settings_and_replicas_are_independent(self):
        case, specs = configs.effective_configuration("cartpole_dqn",
                self.override({"algorithms": {"dqn": {"learning_rate": .0001}}}), workers=2)
        self.assertEqual(specs[0]["config"].learning_rate, .0001)
        specs[0]["config"].agent.hidden_size = 999
        self.assertNotEqual(specs[1]["config"].agent.hidden_size, 999)
        self.assertNotEqual(case["agents"][0]["config"].agent.hidden_size, 999)

    def test_schedule_is_explicit_and_cannot_silently_apply_to_other_algorithms(self):
        good = {"kind": "linear", "final_fraction": .05}
        case = configs.configuration("cartpole_dqn", self.override({"learning_rate_schedule": {"dqn": good}}))
        self.assertEqual(case["agents"][0]["learning_rate_schedule"], good)
        self.assertNotIn("learning_rate_schedule", configs.configuration("cartpole_dqn")["agents"][0])
        for bad in ({}, {"kind": "cosine", "final_fraction": .05},
                    {"kind": "linear", "final_fraction": 0},
                    {"kind": "linear", "final_fraction": True},
                    {"kind": "linear", "final_fraction": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                configs.configuration("cartpole_dqn", self.override({"learning_rate_schedule": {"dqn": bad}}))
        with self.assertRaises(ValueError):
            configs.configuration("hopper_sac", self.override({"learning_rate_schedule": {"sac": good}}))

    def test_ppo_extension_cannot_silently_overwrite_or_ignore_base_parameters(self):
        class FakePpoV2(C.Structure):
            _anonymous_ = ("base",)
            _fields_ = [("base", configs.rx.RxPpoConfig), ("model", C.c_uint32),
                        ("activation", C.c_uint32), ("initial_log_std", C.c_double),
                        ("adam_epsilon", C.c_double), ("target_kl", C.c_double)]
        with patch.object(configs.rx, "RxPpoConfigV2", FakePpoV2, create=True):
            with self.assertRaisesRegex(ValueError, "extension fields only"):
                configs.configuration("cartpole_ppo", self.override({"ppo_v2": {"learning_rate": .1}}))
            result = configs.configuration("cartpole_ppo", self.override({"ppo_v2": {"target_kl": .02},
                       "algorithms": {"ppo": {"learning_rate": .0001}}}))
            self.assertEqual(result["agents"][0]["config"].target_kl, .02)
            self.assertEqual(result["agents"][0]["config"].learning_rate, .0001)
        with self.assertRaisesRegex(ValueError, "absent"):
            configs.configuration("cartpole_dqn", self.override({"ppo_v2": {}}))

    def test_single_ppo_diagnostic_preserves_settings_without_polluting_shared_replay(self):
        before = copy.deepcopy(self.baseline["halfcheetah_hybrid"])
        isolated = configs.configuration("halfcheetah_ppo_diagnostic")
        self.assertEqual(len(isolated["agents"]), 1)
        self.assertFalse(isolated["shared_replay"])
        self.assertEqual(configs.as_dict(isolated["agents"][0]["config"]), before["agents"][0]["config"])
        override = self.override({"normalization": {"ppo": {"normalize_rewards": False}}})
        normalized = configs.configuration("halfcheetah_ppo_diagnostic", override)
        self.assertFalse(normalized["agents"][0]["normalization"]["normalize_rewards"])
        self.assertTrue(normalized["agents"][0]["normalization"]["normalize_observations"])
        with self.assertRaisesRegex(ValueError, "mix transformed and raw"):
            configs.configuration("halfcheetah_hybrid", override)
        shared = configs.configuration("halfcheetah_hybrid", self.override(
            {"normalization": {"ppo": {"preserve_replay_inputs": True}}}))
        self.assertTrue(shared["shared_replay"])
        self.assertTrue(shared["agents"][0]["normalization"]["preserve_replay_inputs"])
        self.assertNotIn("normalization", shared["agents"][2])
        with self.assertRaisesRegex(ValueError, "boolean"):
            configs.configuration("halfcheetah_ppo_diagnostic", self.override(
                {"normalization": {"ppo": {"normalize_observations": 1}}}))
        self.assertEqual(json.loads(configs.BASELINE.read_text())["halfcheetah_hybrid"], before)

    def test_reserved_evaluation_splits_and_worker_budget(self):
        case, specs = configs.effective_configuration("cartpole_dqn", workers=2)
        request = configs.study_request("cartpole_dqn", 1001, 200, case, specs)
        self.assertEqual((request["validation_seed"], request["final_seed"]), (1_100_000, 1_100_000))
        confirmation = configs.study_request("cartpole_dqn", 11, 200, case, specs, stage="confirmation")
        self.assertEqual((confirmation["validation_seed"], confirmation["final_seed"]), (1_100_000, 1_200_000))
        next_round = configs.study_request("cartpole_dqn", 17, 200, case, specs,
                                          stage="confirmation", final_seed=1_300_000)
        self.assertEqual((next_round["validation_seed"], next_round["final_seed"]), (1_100_000, 1_300_000))
        for changes in ({"final_seed": 1_300_000},
                        {"stage": "confirmation", "final_seed": 1_100_000},
                        {"stage": "confirmation", "final_seed": 1_300_001},
                        {"stage": "confirmation", "final_seed": 1_300_000.,},
                        {"stage": "confirmation", "final_seed": 1_300_000, "seed": 1_290_000}):
            values = {"case_name": "cartpole_dqn", "seed": 1001, "steps": 200, "case": case, "specs": specs}
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configs.study_request(**values)
        for changes in ({"steps": 210}, {"seed": 1_190_000}, {"validation_episodes": 100001},
                        {"eval_episodes": 100001}):
            values = {"case_name": "cartpole_dqn", "seed": 1001, "steps": 200, "case": case, "specs": specs}
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configs.study_request(**values)


class ResumeTests(ConfigurationTests):
    def setUp(self):
        super().setUp()
        self.root_patch = patch.object(campaign, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        binary = self.root / "library.dylib"
        binary.write_bytes(b"fake frozen binary")
        write(self.root / "manifest.json", {"library_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()})
        self.job = {"case": "cartpole_dqn", "seed": 1001, "steps": 200, "output": "result",
                    "library": "library.dylib", "manifest": "manifest.json",
                    "checkpoints": 2, "eval-episodes": 3, "validation-episodes": 2}
        self.plan = campaign.prepare_job(self.job)
        self.make_complete()

    def make_complete(self):
        output, request = self.plan["output"], self.plan["request"]
        output.mkdir()
        metadata = {"request": request, "library_sha256": self.plan["manifest"]["library_sha256"],
                    "library": str(self.plan["library"]), "overrides": None,
                    "case": request["case"], "seed": request["seed"], "requested_total_steps": request["steps"],
                    "worker_count": 1, "configuration": request["configuration"], "effective_agents": request["effective_agents"],
                    "study_stage": "development", "final_test_episodes": 3, "validation_episodes": 2,
                    "share_rnd": False, "concurrent_workers": True, "validation_seed": 1_100_000,
                    "final_test_seed": 1_100_000}
        write(output / "metadata.json", metadata)
        final_eval = {"worker": 0, "algorithm": "dqn", "aggregate_steps": 200, "split": "development",
                      "seed_start": 1_100_000, "episodes": 3, "returns": [6., 6., 6.], "lengths": [3, 3, 3],
                      "deterministic": True, "reward": "raw"}
        final = {"status": "complete", "case": "cartpole_dqn", "seed": 1001, "study_stage": "development",
                 "actual_total_steps": 200, "workers": [{"worker": 0, "algorithm": "dqn", "steps": 200}],
                 "test": [final_eval]}
        write(output / "final.json", final)
        write_lines(output / "progress_history.jsonl", [{"checkpoint": i, "aggregate_steps": 100 * i} for i in (1, 2)])
        validations = [{**final_eval, "aggregate_steps": 100 * i, "split": "validation", "episodes": 2,
                        "returns": [6., 6.], "lengths": [3, 3]} for i in range(3)]
        write_lines(output / "evaluations.jsonl", validations + [final_eval])

    def test_identical_and_legacy_completed_runs_resume_without_modifying_logs(self):
        self.assertEqual(campaign.verify_completed(self.plan)["status"], "cached")
        path = self.plan["output"] / "metadata.json"
        metadata = json.loads(path.read_text())
        del metadata["request"]
        write(path, metadata)
        before = path.read_bytes()
        self.assertFalse(campaign.verify_completed(self.plan)["source_hashes_recorded"])
        self.assertEqual(before, path.read_bytes())

    def test_changed_measurement_cannot_resume_a_different_request(self):
        for change in ({"workers": 2}, {"eval-episodes": 4}, {"validation-episodes": 3},
                       {"checkpoints": 4}, {"stage": "confirmation"}, {"serial-workers": True}):
            plan = campaign.prepare_job({**self.job, **change})
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "measurement identity"):
                campaign.verify_completed(plan)

    def test_changed_baseline_configuration_is_not_accepted_by_resume(self):
        self.baseline["cartpole_dqn"]["agents"][0]["config"]["learning_rate"] = .0001
        write(self.root / "configurations.json", self.baseline)
        with self.assertRaisesRegex(ValueError, "measurement identity"):
            campaign.verify_completed(campaign.prepare_job(self.job))

    def test_final_counts_and_per_worker_budgets_are_verified(self):
        path = self.plan["output"] / "final.json"
        original = json.loads(path.read_text())
        broken = copy.deepcopy(original)
        broken["workers"][0]["steps"] = 199
        write(path, broken)
        with self.assertRaisesRegex(ValueError, "worker budget"):
            campaign.verify_completed(self.plan)
        broken = copy.deepcopy(original)
        broken["test"][0]["returns"].pop()
        write(path, broken)
        with self.assertRaisesRegex(ValueError, "final episode count"):
            campaign.verify_completed(self.plan)

    def test_same_hash_different_library_path_and_unknown_job_keys_are_rejected(self):
        (self.root / "other.dylib").write_bytes(self.plan["library"].read_bytes())
        with self.assertRaisesRegex(ValueError, "library path"):
            campaign.verify_completed(campaign.prepare_job({**self.job, "library": "other.dylib"}))
        with self.assertRaisesRegex(ValueError, "unknown job fields"):
            campaign.prepare_job({**self.job, "worker": 2})

    def test_output_collisions_and_incomplete_output_are_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            campaign.validate_outputs([self.plan, self.plan])
        child = campaign.prepare_job({**self.job, "output": "result/child"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            campaign.validate_outputs([self.plan, child])
        new = campaign.prepare_job({**self.job, "output": "incomplete"})
        new["output"].mkdir()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            campaign.validate_outputs([new])

    def test_supported_worker_flags_are_forwarded_to_runner(self):
        plan = campaign.prepare_job({**self.job, "workers": 2, "serial-workers": True})
        command = campaign.command_for(plan)
        self.assertIn("--serial-workers", command)
        self.assertEqual(command[command.index("--workers") + 1], "2")
        self.assertEqual(command[command.index("--library") + 1], str(self.plan["library"]))

    def test_fresh_confirmation_seed_is_forwarded_and_part_of_resume_identity(self):
        with self.assertRaisesRegex(ValueError, "not null"):
            campaign.prepare_job({**self.job, "stage": "confirmation", "final-seed": None})
        plan = campaign.prepare_job({**self.job, "stage": "confirmation", "final-seed": 1_300_000})
        command = campaign.command_for(plan)
        self.assertEqual(command[command.index("--final-seed") + 1], "1300000")
        self.assertNotIn("--final-seed", campaign.command_for(self.plan))
        output = self.plan["output"]
        metadata = json.loads((output / "metadata.json").read_text())
        metadata.update(request=plan["request"], study_stage="confirmation", final_test_seed=1_300_000)
        write(output / "metadata.json", metadata)
        final = json.loads((output / "final.json").read_text())
        final["study_stage"] = "confirmation"
        final["test"][0].update(split="test", seed_start=1_300_000)
        write(output / "final.json", final)
        evaluations = campaign.read_jsonl(output / "evaluations.jsonl")
        evaluations[-1] = final["test"][0]
        write_lines(output / "evaluations.jsonl", evaluations)
        self.assertEqual(campaign.verify_completed(plan)["status"], "cached")
        for final_seed in (None, 1_400_000):
            job = {**self.job, "stage": "confirmation"}
            if final_seed is not None:
                job["final-seed"] = final_seed
            with self.subTest(final_seed=final_seed), self.assertRaisesRegex(ValueError, "measurement identity"):
                campaign.verify_completed(campaign.prepare_job(job))
        final["test"][0]["seed_start"] = 1_200_000
        write(output / "final.json", final)
        with self.assertRaisesRegex(ValueError, "final evaluation protocol"):
            campaign.verify_completed(plan)

    def test_new_confirmation_block_checks_later_workers_and_full_reserved_range(self):
        case, specs = configs.effective_configuration("cartpole_dqn", workers=2)
        # The first worker is outside every reserved range; only worker 1 overlaps.
        with self.assertRaisesRegex(ValueError, "overlaps"):
            configs.study_request("cartpole_dqn", 1_390_000, 200, case, specs,
                                  stage="confirmation", final_seed=1_400_000)
        case, specs = configs.effective_configuration("cartpole_dqn")
        for seed in (1_400_000, 1_499_999):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "overlaps"):
                configs.study_request("cartpole_dqn", seed, 200, case, specs,
                                      stage="confirmation", final_seed=1_400_000)
        result = configs.study_request("cartpole_dqn", 1_500_000, 200, case, specs,
                                       stage="confirmation", final_seed=1_400_000)
        self.assertEqual(result["final_seed"], 1_400_000)


if __name__ == "__main__":
    unittest.main()
