"""Fake checkpoint/inference tests. No native library or real environment is run."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_improvement_checkpoints as verify
import improvement_configs as configs
import gymnasium
import numpy as np


class FakeAgent:
    def __init__(self, *, optimizer=True, mutate=False):
        self.values = {"updates": 0.0, **({"optimizer_steps": 0.0} if optimizer else {})}
        self.mutate, self.closed = mutate, False

    def statistics(self):
        return dict(self.values)

    def act(self, obs):
        if self.mutate:
            self.values["updates"] += 1.0
        return np.zeros(1)

    def close(self):
        self.closed = True


class FakeEnv:
    spec = SimpleNamespace(max_episode_steps=3)
    action_space = object()

    def __init__(self, *, early=False):
        self.seeds, self.early, self.closed = [], early, False

    def reset(self, seed):
        self.seeds.append(seed)
        self.length = 0
        return np.zeros(1), {}

    def step(self, action):
        self.length += 1
        return np.zeros(1), 2.0, self.length == (2 if self.early else 3), False, {}

    def close(self):
        self.closed = True


class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "run"
        self.run.mkdir()
        self.library = self.root / "frozen.dylib"
        self.library.write_bytes(b"fake native library")
        self.manifest = {"library_sha256": hashlib.sha256(self.library.read_bytes()).hexdigest()}
        verify.write_json(self.root / "manifest.json", self.manifest)
        self.spec = {"algorithm": "ppo", "config": configs.as_dict(configs.rx.RxPpoConfigV2()),
                     "rnd_config": None, "coefficient": 0.0}
        self.spec["config"]["model"] = 1
        self.metadata = {"backend": "reinforcex", "case": "fake", "seed": 1001, "env_id": "fake-env",
                         "study_stage": "development", "effective_agents": [self.spec], "worker_count": 1,
                         "requested_total_steps": 20, "final_test_seed": 1_100_000, "final_test_episodes": 2,
                         "max_episode_steps": 3, "library": str(self.library), "library_sha256": self.manifest["library_sha256"],
                         "source_snapshot": str(self.root / "manifest.json"), "build_manifest": self.manifest,
                         "python": sys.version, "packages": {}, "worker_seeds": [{"policy_seed": 1001, "rnd_seed": 1001001}]}
        self.final = {"status": "complete", "backend": "reinforcex", "case": "fake", "seed": 1001,
                      "env_id": "fake-env", "study_stage": "development", "actual_total_steps": 20,
                      "workers": [{"worker": 0, "algorithm": "ppo", "steps": 20, "statistics": {"updates": 5, "optimizer_steps": 10}}],
                      "test": [{"worker": 0, "algorithm": "ppo", "aggregate_steps": 20, "split": "development",
                                "seed_start": 1_100_000, "episodes": 2, "deterministic": True, "reward": "raw",
                                "returns": [6.0, 6.0], "lengths": [3, 3], "mean": 6.0, "std": 0.0, "min": 6.0, "max": 6.0}]}
        verify.write_json(self.run / "metadata.json", self.metadata)
        verify.write_json(self.run / "final.json", self.final)
        verify.write_json(self.run.with_suffix(".launch.json"), {"command": [sys.executable],
                          "job": {"library": str(self.library), "manifest": str(self.root / "manifest.json"), "output": str(self.run)}})
        (self.run / "worker0.ot").write_bytes(b"fake policy weights")
        self.rx = SimpleNamespace(gym_action=lambda agent, action, space: action)

    def request(self):
        return verify.make_request(self.run, 1e-6, 1e-7)

    def test_reconstructs_ppo_v1_v2_dqn_and_both_sac_structs_exactly(self):
        for algorithm, typ in (("ppo", configs.rx.RxPpoConfig), ("ppo", configs.rx.RxPpoConfigV2),
                               ("dqn", configs.rx.RxDqnConfig), ("sac", configs.rx.RxSacConfig),
                               ("sac", configs.rx.RxSacConfigV2)):
            saved = configs.as_dict(typ())
            saved["agent"]["hidden_size"] = 256
            result = verify.reconstruct_config(configs.rx, algorithm, saved)
            self.assertIsInstance(result, typ)
            self.assertEqual(configs.as_dict(result), saved)
        broken = copy.deepcopy(self.spec["config"])
        del broken["adam_epsilon"]
        with self.assertRaisesRegex(ValueError, "missing/unknown"):
            verify.reconstruct_config(configs.rx, "ppo", broken)

    def test_sac_temperature_and_shared_rnd_are_in_checkpoint_fingerprint_set(self):
        spec = {"algorithm": "sac", "rnd_config": {}}
        files = verify.checkpoint_paths(self.run, 2, spec, {"share_rnd": True})
        self.assertEqual([p.name for p in files[:4]], ["worker2_actor.ot", "worker2_critic1.ot", "worker2_critic2.ot", "worker2_temperature.ot"])
        self.assertEqual([p.parent.name for p in files[4:]], ["rnd_worker0", "rnd_worker0"])

    def test_full_saved_split_is_kept_and_corrupt_reference_is_rejected(self):
        request = self.request()
        self.assertEqual([ep["seed"] for ep in request["references"][0]], [1_100_000, 1_100_001])
        metadata, final = copy.deepcopy(self.metadata), copy.deepcopy(self.final)
        metadata["study_stage"] = final["study_stage"] = "confirmation"
        metadata["final_test_seed"] = final["test"][0]["seed_start"] = 1_200_000
        final["test"][0]["split"] = "test"
        self.assertEqual(verify.references(metadata, final)[0][0]["seed"], 1_200_000)
        for field, value in (("episodes", 1), ("reward", "shaped"), ("split", "validation"), ("mean", 999.0)):
            broken = copy.deepcopy(self.final)
            broken["test"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.references(self.metadata, broken)

    def test_library_hash_and_launch_path_must_match(self):
        self.library.write_bytes(b"different library")
        with self.assertRaisesRegex(ValueError, "library hash"):
            self.request()

    def test_normalization_cannot_be_silently_ignored(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["effective_agents"][0]["normalization"] = {"normalize_observations": True}
        verify.write_json(self.run / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.request()
        verify.write_json(self.run / "metadata.json", self.metadata)
        (self.run / "worker0.normalization.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.request()

    def test_fake_inference_matches_every_return_length_and_preserves_zero_counters(self):
        expected = verify.references(self.metadata, self.final)[0]
        env, agent = FakeEnv(), FakeAgent()
        result = verify.evaluate_loaded(agent, self.rx, env, expected, {"optimizer_steps": 10}, 1e-6, 1e-7)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(env.seeds, [1_100_000, 1_100_001])
        self.assertEqual(result["max_absolute_return_difference"], 0.0)
        self.assertEqual(result["optimizer_steps"], 0)
        with self.assertRaisesRegex(RuntimeError, "training is forbidden"):
            agent.act_and_train(None, 0)
        with self.assertRaisesRegex(RuntimeError, "saving is forbidden"):
            agent.save()

    def test_mismatch_and_counter_changes_fail_instead_of_passing(self):
        expected = verify.references(self.metadata, self.final)[0]
        result = verify.evaluate_loaded(FakeAgent(), self.rx, FakeEnv(early=True), expected, {}, 1e-6, 1e-7)
        self.assertEqual(result["status"], "failed")
        with self.assertRaisesRegex(ValueError, "counter must be zero"):
            verify.evaluate_loaded(FakeAgent(mutate=True), self.rx, FakeEnv(), expected, {}, 1e-6, 1e-7)
        with self.assertRaisesRegex(ValueError, "counter exposed"):
            verify.zero_update_statistics(FakeAgent(optimizer=False), {"optimizer_steps": 10})
        old = verify.evaluate_loaded(FakeAgent(optimizer=False), self.rx, FakeEnv(), expected, {}, 1e-6, 1e-7)
        self.assertFalse(old["optimizer_counter_available"])
        self.assertIsNone(old["optimizer_steps"])

    def test_worker_loads_exact_v2_library_config_without_saving_or_training(self):
        request = self.request()
        agent, env = FakeAgent(), FakeEnv()
        fake_lib = SimpleNamespace(_name=str(self.library))
        with patch("ctypes.CDLL", return_value=fake_lib) as cdll, \
             patch.object(configs.rx, "configure_ffi"), patch.object(configs.rx, "cuda_is_available", return_value=False), \
             patch.object(configs.rx, "manual_seed"), patch.object(configs.rx, "create_ppo", return_value=agent) as create, \
             patch.object(configs.rx, "gym_action", side_effect=self.rx.gym_action), \
             patch.object(gymnasium, "make", return_value=env):
            result = verify.run_worker(request)
        self.assertEqual(result["status"], "passed")
        cdll.assert_called_once_with(str(self.library))
        self.assertIsInstance(create.call_args.args[1], configs.rx.RxPpoConfigV2)
        self.assertIsNone(create.call_args.args[2])
        self.assertEqual(create.call_args.args[3], str(self.run / "worker0.ot"))
        self.assertTrue(agent.closed and env.closed)
        self.assertTrue(result["checkpoint_unchanged"])
        self.assertFalse(result["cached"])

    def test_worker_restores_normalization_and_evaluation_does_not_change_moments(self):
        from reinforcex_normalization import NormalizedAgent
        options = {"normalize_observations": True, "normalize_rewards": True,
                   "clip_observations": 10.0, "clip_rewards": 10.0}
        self.spec["normalization"] = options
        self.spec["config"]["agent"].update(obs_size=1, gamma=.99)
        verify.write_json(self.run / "metadata.json", self.metadata)
        state = NormalizedAgent(FakeAgent(), 1, .99, **options).state_dict()
        state["observation"] = {"mean": [10.0], "var": [4.0], "count": 100.0}
        state["discounted_return"] = {"mean": 1.0, "var": 2.0, "count": 20.0}
        state_path = self.run / "worker0.normalization.json"
        verify.write_json(state_path, state)
        request = self.request()
        self.assertIn(str(state_path), request["checkpoint_fingerprints"])
        agent, env = FakeAgent(), FakeEnv()
        observations = []
        agent.act = lambda obs: observations.append(np.asarray(obs).copy()) or np.zeros(1)
        fake_lib = SimpleNamespace(_name=str(self.library))
        with patch("ctypes.CDLL", return_value=fake_lib), patch.object(configs.rx, "configure_ffi"), \
             patch.object(configs.rx, "cuda_is_available", return_value=False), patch.object(configs.rx, "manual_seed"), \
             patch.object(configs.rx, "create_ppo", return_value=agent), \
             patch.object(configs.rx, "gym_action", side_effect=self.rx.gym_action), \
             patch.object(gymnasium, "make", return_value=env):
            result = verify.run_worker(request)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(np.allclose(obs, [-5.0]) for obs in observations))
        row = result["workers"][0]
        self.assertEqual(row["normalization_state_before"], state)
        self.assertEqual(row["normalization_state_after"], state)
        self.assertEqual(row["statistics_after"]["normalization_observation_count"], 100.0)
        self.assertEqual(verify.read_json(state_path), state)

    def test_nonzero_child_exit_missing_output_and_truncated_success_are_errors(self):
        request = self.request()
        raw = self.root / "child.json"
        verify.write_json(raw, {"status": "passed"})
        result = verify.verified_child_result(SimpleNamespace(returncode=1), raw, request)
        self.assertEqual(result["status"], "error")
        result = verify.verified_child_result(SimpleNamespace(returncode=0), raw, request)
        self.assertEqual(result["status"], "error")
        raw.unlink()
        result = verify.verified_child_result(SimpleNamespace(returncode=0), raw, request)
        self.assertEqual(result["status"], "error")

    def test_changed_checkpoint_or_reference_fails_controller_verification(self):
        for path in (self.run / "worker0.ot", self.run / "final.json"):
            request = self.request()
            before = path.read_bytes()
            raw = self.root / "child.json"
            verify.write_json(raw, {"status": "failed"})
            path.write_bytes(before + b" ")
            result = verify.verified_child_result(SimpleNamespace(returncode=0), raw, request)
            self.assertEqual(result["status"], "failed")
            self.assertIn("changed", result["error"])
            path.write_bytes(before)

    def test_controller_starts_fresh_child_and_does_not_reuse_passed_cache(self):
        args = SimpleNamespace(runs=[self.run], output_dir=self.root / "verification", libtorch_dir=self.root,
                               atol=1e-6, rtol=1e-7, python=None, timeout=100)
        verify.write_json(args.output_dir / "run.json", {"status": "passed", "cached": True})
        with patch.object(verify.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as child:
            self.assertEqual(verify.controller(args), 1)
        child.assert_called_once()
        command = child.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--worker-request", command)
        environment = child.call_args.kwargs["env"]
        self.assertEqual(environment["DYLD_LIBRARY_PATH"], str(self.root))
        self.assertTrue(all(environment[name] == "1" for name in verify.THREADS))
        result = verify.read_json(args.output_dir / "run.json")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["cached"])
        self.assertEqual(verify.read_json(self.run / "final.json"), self.final)


if __name__ == "__main__":
    unittest.main()
