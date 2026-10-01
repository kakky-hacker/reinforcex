"""SB3 normalization adapter contracts; synthetic envs/fake models, no learning.

Run in the dedicated SB3 interpreter after clearing both LibTorch loader paths::

    env -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH <sb3-python> -B -m unittest discover -s benchmarks -p test_improvement_sb3.py -v
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

import improvement_sb3 as reference
from reinforcex_normalization import NormalizedAgent


class ListSink:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(copy.deepcopy(row))


class TwoStepEnv(gym.Env):
    observation_space = gym.spaces.Box(-100., 100., shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, truncated=False):
        self.truncated = truncated
        self.resets = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets.append(seed)
        self.index = 0
        return np.array([1., 2.], dtype=np.float32), {}

    def step(self, action):
        self.index += 1
        done = self.index == 2
        observation = np.array([1 + self.index * 2, 2 + self.index * 2], dtype=np.float32)
        return observation, self.index * 2., done and not self.truncated, done and self.truncated, {}

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self):
        self.training = True

    def set_training_mode(self, training):
        self.training = training


class FakeModel:
    loads = []

    def __init__(self, content=b"model-weights"):
        self.content = content
        self.policy = FakePolicy()
        self.num_timesteps = 4
        self._n_updates = 0
        self.observations = []

    def predict(self, observation, deterministic=True):
        self.observations.append(np.array(observation).copy())
        return 0, None

    def save(self, path):
        Path(path).write_bytes(self.content)

    @classmethod
    def load(cls, path, *, device):
        cls.loads.append((str(path), device))
        return cls(Path(path).read_bytes())


def document(options=None):
    spec = {"algorithm": "ppo", "rnd_config": None, "coefficient": 0.,
            "config": {"agent": {"obs_size": 2, "action_size": 2,
                                    "hidden_layers": 0, "hidden_size": 16, "gamma": .5},
                       "action_space": 0, "learning_rate": .0003, "gae_lambda": .95,
                       "update_interval": 8, "epochs": 2, "minibatch_size": 4,
                       "policy_clip_epsilon": .2, "value_clip_range": 0.,
                       "value_loss_coefficient": .5, "entropy_coefficient": 0.,
                       "standardize_gae": 1, "model": 1, "activation": 0,
                       "initial_log_std": 0., "adam_epsilon": 1e-5, "target_kl": 0.,
                       "min_action": -1., "max_action": 1., "min_variance": .01}}
    if options is not None:
        spec["normalization"] = dict(options)
    return {"env_id": "Synthetic-v0", "reward_mode": "raw", "agents": [spec], "shared_replay": False}


class ReferenceNormalizationTests(unittest.TestCase):
    def setUp(self):
        FakeModel.loads = []
        self.document = document({"normalize_observations": True, "normalize_rewards": True,
                                  "clip_observations": 10., "clip_rewards": 10.})
        self.args = argparse.Namespace(reward_transform="scale", reward_scale=.5,
                                       stage="development", validation_seed=1100000,
                                       final_eval_seed=1100000, success_threshold=None)

    def normalizer(self):
        return reference.normalization_for_config(self.document)

    def test_training_sequence_exactly_matches_native_wrapper_moments_and_outputs(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                normalizer = self.normalizer()
                sink = ListSink()
                env = reference.TrainingRecorder(TwoStepEnv(truncated), self.args, sink, normalizer)
                native_sink = reference._NormalizationSink()
                native = NormalizedAgent(native_sink, 2, .5)
                initial, _ = env.reset(seed=13)
                expected_initial, _ = native.act_and_train([1., 2.], 0.)
                np.testing.assert_array_equal(initial, expected_initial)
                obs, reward, terminated, time_limit, _ = env.step(0)
                expected_obs, expected_reward = native.act_and_train([3., 4.], 1.)
                np.testing.assert_array_equal(obs, expected_obs)
                self.assertEqual(reward, expected_reward)
                self.assertFalse(terminated or time_limit)
                obs, reward, terminated, time_limit, info = env.step(0)
                native.stop_episode([5., 6.], 2., terminated=not truncated)
                np.testing.assert_array_equal(obs, native_sink.terminal[0])
                self.assertEqual(reward, native_sink.terminal[1])
                self.assertEqual(normalizer.state_dict(), native.state_dict())
                self.assertAlmostEqual(normalizer.state_dict()["observation"]["count"], 3.0001)
                self.assertAlmostEqual(normalizer.state_dict()["discounted_return"]["count"], 2.0001)
                self.assertEqual(normalizer.agent._discounted_return, 0.)
                self.assertEqual(info["episode"]["r"], 6.)
                self.assertEqual(sink.rows[0]["return"], 6.)
                self.assertEqual(sink.rows[0]["pre_normalization_return"], 3.)
                self.assertAlmostEqual(sink.rows[0]["train_return"], expected_reward + reward)
                env.close()

    def test_dummy_vecenv_preserves_terminal_observation_and_documents_autoreset_count(self):
        normalizer = self.normalizer()
        recorder = reference.TrainingRecorder(TwoStepEnv(truncated=True), self.args, ListSink(), normalizer)
        env = reference.DummyVecEnv([lambda: recorder])
        expected = self.normalizer()
        try:
            env.reset()
            expected.reset([1., 2.])
            env.step(np.array([0]))
            expected.step([3., 4.], 1., False, False)
            _, _, done, infos = env.step(np.array([0]))
            terminal, _ = expected.step([5., 6.], 2., False, True)
            self.assertTrue(done[0])
            np.testing.assert_array_equal(infos[0]["terminal_observation"], terminal)
            expected.reset([1., 2.])
            self.assertEqual(normalizer.training_state(), expected.training_state())
            self.assertAlmostEqual(normalizer.state_dict()["observation"]["count"], 4.0001)
            self.assertAlmostEqual(normalizer.state_dict()["discounted_return"]["count"], 2.0001)
        finally:
            env.close()

    def test_100_episode_evaluation_is_raw_and_freezes_moments_and_live_return(self):
        normalizer = self.normalizer()
        normalizer.reset([1., 2.])
        normalizer.step([3., 4.], 1., False, False)
        before = normalizer.training_state()
        model, episodes, points = FakeModel(), ListSink(), ListSink()
        with patch.object(reference, "make_env", return_value=TwoStepEnv(truncated=True)):
            evaluator = reference.Evaluator(self.args, episodes, points, normalizer)
        try:
            result = evaluator.run(model, "final", 100, 4)
            self.assertEqual(result["mean_return"], 6.)
            self.assertEqual(result["std_return"], 0.)
            self.assertEqual(result["episodes"], 100)
            self.assertTrue(result["normalization_state_unchanged"])
            self.assertEqual(normalizer.training_state(), before)
            self.assertTrue(model.policy.training)
            self.assertEqual([row["seed"] for row in episodes.rows], list(range(1100000, 1100100)))
            np.testing.assert_array_equal(model.observations[0], normalizer.evaluate([1., 2.]))
        finally:
            evaluator.close()

    def test_independent_normalization_flags_and_raw_replay_option_mapping(self):
        for observations in (False, True):
            for rewards in (False, True):
                config = document({"normalize_observations": observations, "normalize_rewards": rewards,
                                   "preserve_replay_inputs": True})
                normalizer = reference.normalization_for_config(config)
                obs = normalizer.reset([1., 2.])
                final, reward = normalizer.step([3., 4.], 4., True, False)
                if not observations:
                    np.testing.assert_array_equal(obs, [1., 2.])
                    np.testing.assert_array_equal(final, [3., 4.])
                if not rewards:
                    self.assertEqual(reward, 4.)
                self.assertEqual(normalizer.agent.normalize_observations, observations)
                self.assertEqual(normalizer.agent.normalize_rewards, rewards)
                self.assertFalse(normalizer.agent.preserve_replay_inputs)
                self.assertTrue(normalizer.native_options["preserve_replay_inputs"])

    def test_normalization_spec_is_validated_instead_of_ignored(self):
        self.assertIsNone(reference.normalization_for_config(document()))
        for options in (False, [], {"unknown": True}, {"normalize_rewards": 1},
                        {"clip_observations": 0}, {"preserve_replay_inputs": "yes"}):
            config = document()
            config["agents"][0]["normalization"] = options
            with self.subTest(options=options), self.assertRaises(ValueError):
                reference.normalization_for_config(config)
        normalizer = self.normalizer()
        normalizer.reset([1., 2.])
        with self.assertRaisesRegex(RuntimeError, "previous episode"):
            normalizer.reset([1., 2.])

    def fixture(self, directory, normalized=True):
        path = Path(directory) / "input.json"
        config = copy.deepcopy(self.document) if normalized else document()
        path.write_text(json.dumps(config))
        args = argparse.Namespace(algo="ppo", env="Synthetic-v0", config=path, native_config=config)
        return args, reference.normalization_for_config(config)

    def test_checkpoint_pair_roundtrip_restores_identical_eval_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, normalizer = self.fixture(directory)
            normalizer.reset([1., 2.])
            normalizer.step([3., 4.], 3., True, False)
            before = normalizer.state_dict()
            reference.save_reference_checkpoint(FakeModel(), output, args, normalizer)
            with patch.dict(reference.ALGORITHMS, {"ppo": FakeModel}):
                model, restored = reference.load_reference_checkpoint(output, args)
            self.assertEqual(model.content, b"model-weights")
            self.assertEqual(FakeModel.loads, [(str(output / "final_model.zip"), "cpu")])
            self.assertEqual(restored.state_dict(), before)
            self.assertFalse(restored.agent._pending_action)
            np.testing.assert_array_equal(restored.evaluate([-3., 7.]), normalizer.evaluate([-3., 7.]))
            for _ in range(100):
                restored.evaluate([999., -999.])
            self.assertEqual(restored.state_dict(), before)

    def test_checkpoint_hash_or_configuration_mismatch_prevents_weight_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, normalizer = self.fixture(directory)
            reference.save_reference_checkpoint(FakeModel(), output, args, normalizer)
            state_path = output / "final_normalization.json"
            original = state_path.read_bytes()
            state_path.write_bytes(original + b" ")
            with patch.dict(reference.ALGORITHMS, {"ppo": FakeModel}), self.assertRaisesRegex(ValueError, "hash mismatch"):
                reference.load_reference_checkpoint(output, args)
            self.assertFalse(FakeModel.loads)
            state_path.write_bytes(original)
            args.native_config["agents"][0]["normalization"]["normalize_rewards"] = False
            with patch.dict(reference.ALGORITHMS, {"ppo": FakeModel}), self.assertRaisesRegex(ValueError, "options mismatch"):
                reference.load_reference_checkpoint(output, args)
            self.assertFalse(FakeModel.loads)
            args.config.write_text("{}")
            with patch.dict(reference.ALGORITHMS, {"ppo": FakeModel}), self.assertRaisesRegex(ValueError, "configuration"):
                reference.load_reference_checkpoint(output, args)
            self.assertFalse(FakeModel.loads)

    def test_corrupt_moments_rejected_even_with_updated_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, normalizer = self.fixture(directory)
            reference.save_reference_checkpoint(FakeModel(), output, args, normalizer)
            path = output / "final_normalization.json"
            state = json.loads(path.read_text())
            state["observation"]["var"] = [-1., 1.]
            path.write_text(json.dumps(state))
            manifest = output / "final_checkpoint.json"
            data = json.loads(manifest.read_text())
            data["artifacts"]["normalization"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            with patch.dict(reference.ALGORITHMS, {"ppo": FakeModel}), self.assertRaises(ValueError):
                reference.load_reference_checkpoint(output, args)
            self.assertFalse(FakeModel.loads)

    def test_plain_checkpoint_has_no_normalization_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, normalizer = self.fixture(directory, normalized=False)
            _, state_path = reference.save_reference_checkpoint(FakeModel(), output, args, normalizer)
            self.assertIsNone(state_path)
            self.assertFalse((output / "final_normalization.json").exists())
            with patch.dict(reference.ALGORITHMS, {"ppo": FakeModel}):
                _, restored = reference.load_reference_checkpoint(output, args)
            self.assertIsNone(restored)

    def test_real_sb3_ppo_weights_and_normalization_reload_without_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, normalizer = self.fixture(directory)
            normalizer.reset([1., 2.])
            normalizer.step([3., 4.], 2., True, False)
            env = TwoStepEnv()
            model = reference.PPO("MlpPolicy", env, n_steps=8, batch_size=4, n_epochs=1,
                                  policy_kwargs={"net_arch": [8]}, device="cpu", seed=7)
            try:
                self.assertEqual(model._n_updates, 0)
                reference.save_reference_checkpoint(model, output, args, normalizer)
                restored, state = reference.load_reference_checkpoint(output, args)
                self.assertEqual(restored._n_updates, 0)
                for name, parameter in model.policy.state_dict().items():
                    self.assertTrue(reference.torch.equal(parameter, restored.policy.state_dict()[name]), name)
                for observation in ([1., 2.], [-9., 4.], [100., -100.]):
                    old = model.predict(normalizer.evaluate(observation), deterministic=True)[0]
                    new = restored.predict(state.evaluate(observation), deterministic=True)[0]
                    np.testing.assert_array_equal(old, new)
                self.assertEqual(state.state_dict(), normalizer.state_dict())
            finally:
                env.close()

    def test_parse_mapping_includes_normalization_and_rejects_gamma_override(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            command = ["--config", str(args.config), "--case", "synthetic", "--seed", "1001",
                       "--steps", "80", "--output", str(Path(directory) / "unused")]
            parsed = reference.parse_args(command)
            reference.model_kwargs(parsed)
            self.assertTrue(any("float64 Welford" in note for note in parsed.comparison_notes))
            overridden = reference.parse_args(command + ["--gamma", ".9"])
            with self.assertRaisesRegex(ValueError, "normalization return discount"):
                reference.model_kwargs(overridden)
            self.assertFalse(parsed.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
