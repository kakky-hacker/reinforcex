"""Normalization contracts with fake agents; no native build, Gym, or Torch needed.

    python3 -m unittest discover -s ffi/tests -p test_normalization.py -v
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from reinforcex_normalization import NormalizedAgent, _RunningMeanStd


class RecordingAgent:
    discrete = False
    action_bounds = (-1.0, 1.0)
    output_size = 2

    def __init__(self):
        self.handle = 71
        self.lib = object()
        self.training = []
        self.ends = []
        self.evaluations = []
        self.saves = self.loads = 0
        self.fail_save = self.fail_load = False
        self.native_statistics = {"updates": 3.0}

    def act_and_train(self, observation, reward):
        # Retain the actual object to catch accidental mutation/reuse by wrapper.
        self.training.append((observation, reward))
        return np.array([.25, -.25], dtype=np.float32)

    def stop_episode(self, observation, reward, *, terminated=True):
        self.ends.append((observation, reward, terminated))

    def act(self, observation):
        self.evaluations.append(observation)
        return np.array([observation.sum(), -observation.sum()], dtype=np.float32)

    def statistics(self):
        return self.native_statistics

    def save(self):
        if self.fail_save:
            raise RuntimeError("native save failed")
        self.saves += 1

    def load(self):
        if self.fail_load:
            raise RuntimeError("native load failed")
        self.loads += 1

    def close(self):
        self.handle = 0


class ReplayRecordingAgent(RecordingAgent):
    def __init__(self):
        super().__init__()
        self.lib = SimpleNamespace(rx_agent_act_and_train_with_replay_input=object(),
                                   rx_agent_stop_episode_with_replay_input=object())
        self.replay_training = []
        self.replay_ends = []

    def act_and_train_with_replay_input(self, observation, reward, replay_observation, replay_reward):
        self.replay_training.append((observation, reward, replay_observation, replay_reward))
        return np.array([.5, -.5], dtype=np.float32)

    def stop_episode_with_replay_input(self, observation, reward, replay_observation,
                                      replay_reward, *, terminated=True):
        self.replay_ends.append((observation, reward, replay_observation, replay_reward, terminated))


def expected_moments(samples):
    samples = np.asarray(samples, dtype=np.float64)
    count = len(samples) + 1e-4
    mean = samples.sum(axis=0) / count
    variance = (np.square(samples).sum(axis=0) + 1e-4) / count - np.square(mean)
    return count, mean, variance


def episode(agent, reward=2.0, terminated=True):
    agent.act_and_train([1., 3.], 0.0)
    agent.stop_episode([2., 5.], reward, terminated=terminated)


class NormalizationTests(unittest.TestCase):
    def test_float64_welford_population_moments_with_prior(self):
        samples = [[1., 5.], [3., -2.], [2., 7.], [-4., .5]]
        rms = _RunningMeanStd((2,))
        for sample in samples:
            rms.update(sample)
        count, mean, variance = expected_moments(samples)
        self.assertEqual(rms.count, count)
        self.assertEqual(rms.mean.dtype, np.float64)
        self.assertEqual(rms.var.dtype, np.float64)
        np.testing.assert_allclose(rms.mean, mean, rtol=1e-13)
        np.testing.assert_allclose(rms.var, variance, rtol=1e-13)

    def test_dummy_excluded_terminal_counted_and_return_discounted_once(self):
        native = RecordingAgent()
        agent = NormalizedAgent(native, 2, .5)
        agent.act_and_train([1., 2.], 999.)  # No previous action: ignored dummy.
        self.assertEqual(native.training[0][1], 0)
        self.assertEqual(agent.statistics()["normalization_reward_count"], 1e-4)
        agent.act_and_train([3., 4.], 2.)
        agent.act_and_train([5., 6.], 4.)
        agent.stop_episode([7., 8.], 8., terminated=True)
        state = agent.state_dict()
        count, mean, variance = expected_moments([2., 5., 10.5])
        self.assertAlmostEqual(state["discounted_return"]["count"], count)
        self.assertAlmostEqual(state["discounted_return"]["mean"], mean)
        self.assertAlmostEqual(state["discounted_return"]["var"], variance)
        self.assertAlmostEqual(state["observation"]["count"], 4.0001)
        for index, (raw_reward, discounted) in enumerate(zip([2., 4., 8.], [2., 5., 10.5])):
            _, _, variance = expected_moments([2., 5., 10.5][:index + 1])
            expected = np.clip(raw_reward / np.sqrt(variance + 1e-8), -10, 10)
            actual = native.training[index + 1][1] if index < 2 else native.ends[0][1]
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(agent._discounted_return, 0)
        self.assertFalse(agent._pending_action)

    def test_termination_and_truncation_both_reset_return_and_dummy(self):
        native = RecordingAgent()
        agent = NormalizedAgent(native, 2, .9)
        episode(agent, reward=3., terminated=True)
        episode(agent, reward=7., terminated=False)
        self.assertEqual([row[2] for row in native.ends], [True, False])
        self.assertEqual([row[1] for row in native.training], [0., 0.])
        count, mean, variance = expected_moments([3., 7.])
        stats = agent.state_dict()["discounted_return"]
        self.assertAlmostEqual(stats["count"], count)
        self.assertAlmostEqual(stats["mean"], mean)
        self.assertAlmostEqual(stats["var"], variance)
        self.assertEqual(agent._discounted_return, 0)
        self.assertFalse(agent._pending_action)

    def test_stop_without_pending_action_does_not_count_a_reward(self):
        agent = NormalizedAgent(RecordingAgent(), 2, .9)
        agent.stop_episode([1., 2.], 99., terminated=False)
        self.assertEqual(agent.statistics()["normalization_reward_count"], 1e-4)
        episode(agent)
        before = agent.state_dict()["discounted_return"]
        agent.stop_episode([1., 2.], 99.)
        self.assertEqual(agent.state_dict()["discounted_return"], before)

    def test_100_evaluation_episodes_leave_all_training_state_unchanged(self):
        native = RecordingAgent()
        agent = NormalizedAgent(native, 2, .9)
        agent.act_and_train([1., 2.], 0.)
        agent.act_and_train([2., 4.], 3.)
        before = json.dumps(agent.state_dict(), sort_keys=True)
        statistics = agent.statistics()
        episodic = agent._discounted_return, agent._pending_action
        prediction = agent.act([.5, -.5])
        for _ in range(100):
            for observation in ([1000., -1000.], [.5, -.5], [4., 9.]):
                agent.act(observation)
        np.testing.assert_array_equal(agent.act([.5, -.5]), prediction)
        self.assertEqual(json.dumps(agent.state_dict(), sort_keys=True), before)
        self.assertEqual(agent.statistics(), statistics)
        self.assertEqual((agent._discounted_return, agent._pending_action), episodic)
        self.assertEqual(len(native.training), 2)
        self.assertFalse(native.ends)
        self.assertEqual(native.loads, 0)

    def test_observation_and_reward_enable_flags_are_independent(self):
        for obs_enabled in (False, True):
            for reward_enabled in (False, True):
                with self.subTest(obs=obs_enabled, reward=reward_enabled):
                    native = RecordingAgent()
                    agent = NormalizedAgent(native, 2, .99, normalize_observations=obs_enabled,
                                            normalize_rewards=reward_enabled)
                    episode(agent, reward=7.)
                    self.assertAlmostEqual(agent.statistics()["normalization_observation_count"], 2.0001 if obs_enabled else .0001)
                    self.assertAlmostEqual(agent.statistics()["normalization_reward_count"], 1.0001 if reward_enabled else .0001)
                    if not obs_enabled:
                        np.testing.assert_array_equal(native.training[0][0], [1., 3.])
                        np.testing.assert_array_equal(native.ends[0][0], [2., 5.])
                    if not reward_enabled:
                        self.assertEqual(native.ends[0][1], 7.)

    def test_observation_and_reward_clipping_without_mean_subtraction(self):
        native = RecordingAgent()
        agent = NormalizedAgent(native, 2, .99, clip_observations=3., clip_rewards=2.)
        agent.act([100., -100.])
        np.testing.assert_array_equal(native.evaluations[-1], [3., -3.])
        episode(agent, reward=-5.)
        self.assertEqual(native.ends[0][1], -2.)
        self.assertEqual(native.training[0][0].dtype, np.float32)
        self.assertTrue(native.training[0][0].flags.c_contiguous)

    def test_collected_observations_are_copies_and_not_renormalized_later(self):
        native = RecordingAgent()
        agent = NormalizedAgent(native, 2, .99)
        observation = np.array([1., 2.], dtype=np.float64)
        agent.act_and_train(observation, 0.)
        captured = native.training[0][0].copy()
        observation[:] = 999.
        for index in range(20):
            agent.act_and_train([index * 10., -index * 10.], 1.)
        np.testing.assert_array_equal(native.training[0][0], captured)
        self.assertFalse(np.shares_memory(observation, native.training[0][0]))
        self.assertFalse(np.shares_memory(native.training[0][0], native.training[-1][0]))

    def test_per_agent_statistics_are_independent_and_attributes_delegate(self):
        one, two = RecordingAgent(), RecordingAgent()
        first, second = NormalizedAgent(one, 2, .9), NormalizedAgent(two, 2, .9)
        before = second.state_dict()
        episode(first)
        self.assertEqual(second.state_dict(), before)
        self.assertIs(first.lib, one.lib)
        self.assertEqual((first.handle, first.discrete, first.action_bounds, first.output_size),
                         (71, False, (-1., 1.), 2))
        first.statistics()["updates"] = -1
        self.assertEqual(one.native_statistics, {"updates": 3.0})
        first.close()
        self.assertEqual(first.handle, 0)
        self.assertEqual(second.handle, 71)

    def test_save_and_constructor_restore_moments_but_start_new_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subdir/state.json"
            native = RecordingAgent()
            agent = NormalizedAgent(native, 2, .5, save_path=path)
            agent.act_and_train([1., 2.], 0.)
            agent.act_and_train([2., 3.], 2.)
            agent.save()  # The checkpoint does not serialize a live environment.
            self.assertEqual(native.saves, 1)
            self.assertEqual(json.loads(path.read_text()), agent.state_dict())
            restored_native = RecordingAgent()
            restored = NormalizedAgent(restored_native, 2, .5, load_path=path)
            self.assertEqual(restored.state_dict(), agent.state_dict())
            self.assertEqual(restored_native.loads, 0, "already-created native agent must not reload twice")
            self.assertFalse(restored._pending_action)
            self.assertEqual(restored._discounted_return, 0)
            np.testing.assert_array_equal(restored.act([2., -2.]), agent.act([2., -2.]))
            restored.act_and_train([9., 4.], 0.)
            restored.stop_episode([8., 3.], 5.)
            self.assertAlmostEqual(restored.state_dict()["discounted_return"]["mean"], expected_moments([2., 5.])[1])
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_explicit_load_restores_and_requires_an_episode_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            native = RecordingAgent()
            agent = NormalizedAgent(native, 2, .9, save_path=path, load_path=None)
            episode(agent)
            agent.save()
            saved = agent.state_dict()
            agent.load_path = path
            episode(agent, 7.)
            agent.load()
            self.assertEqual(native.loads, 1)
            self.assertEqual(agent.state_dict(), saved)
            agent.act_and_train([1., 3.], 0.)
            with self.assertRaisesRegex(RuntimeError, "between training episodes"):
                agent.load()
            self.assertEqual(native.loads, 1)

    def test_mismatched_or_corrupt_state_rejected_before_native_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            native = RecordingAgent()
            agent = NormalizedAgent(native, 2, .9, save_path=path)
            episode(agent)
            agent.save()
            saved = agent.state_dict()
            agent.load_path = path
            mutations = [
                lambda s: s.update(schema=2), lambda s: s.update(schema=True),
                lambda s: s.update(gamma=.8), lambda s: s.update(gamma=True),
                lambda s: s.update(observation_shape=[3]),
                lambda s: s["options"].update(normalize_rewards=False),
                lambda s: s["options"].update(normalize_observations=1),
                lambda s: s["options"].update(clip_rewards=5.),
                lambda s: s["options"].update(preserve_replay_inputs=True),
                lambda s: s["observation"].update(mean=[1.]),
                lambda s: s["observation"].update(var=[-1., 1.]),
                lambda s: s["observation"].update(count=0),
                lambda s: s["observation"].update(count=True),
                lambda s: s["discounted_return"].update(var=float("nan")),
                lambda s: s["discounted_return"].update(mean=[1.]),
                lambda s: s.update(load_episode_policy="resume_pending"),
            ]
            for mutate in mutations:
                bad = copy.deepcopy(saved)
                mutate(bad)
                path.write_text(json.dumps(bad))
                with self.subTest(state=bad), self.assertRaises(ValueError):
                    agent.load()
                self.assertEqual(native.loads, 0)
                self.assertEqual(agent.state_dict(), saved)
            path.write_text(json.dumps(saved))
            for kwargs in ({"gamma": .8}, {"observation_size": 3}, {"normalize_rewards": False}):
                arguments = {"observation_size": 2, "gamma": .9, "load_path": path, **kwargs}
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    NormalizedAgent(RecordingAgent(), **arguments)

    def test_atomic_json_save_keeps_old_file_on_native_or_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            native = RecordingAgent()
            agent = NormalizedAgent(native, 2, .9, save_path=path)
            episode(agent)
            agent.save()
            original = path.read_bytes()
            episode(agent, 5.)
            native.fail_save = True
            with self.assertRaisesRegex(RuntimeError, "native save failed"):
                agent.save()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            native.fail_save = False
            with patch("reinforcex_normalization.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    agent.save()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_native_load_failure_does_not_replace_normalization_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            native = RecordingAgent()
            agent = NormalizedAgent(native, 2, .9, save_path=path)
            episode(agent)
            agent.save()
            episode(agent, 8.)
            before = agent.state_dict()
            agent.load_path, native.fail_load = path, True
            with self.assertRaisesRegex(RuntimeError, "native load failed"):
                agent.load()
            self.assertEqual(agent.state_dict(), before)

    def test_missing_paths_and_bad_inputs_are_explicit(self):
        native = RecordingAgent()
        agent = NormalizedAgent(native, 2, .9)
        for operation in (agent.save, agent.load):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual((native.loads, native.saves), (0, 0))
        for observation, reward in (([1.], 0.), ([float("nan"), 0.], 0.), ([1., 2.], float("inf")), ([1., 2.], [1.])):
            before = agent.state_dict()
            with self.subTest(observation=observation, reward=reward), self.assertRaises(ValueError):
                agent.act_and_train(observation, reward)
            self.assertEqual(agent.state_dict(), before)
        for kwargs in ({"gamma": 1.1}, {"gamma": float("nan")}, {"observation_size": 0},
                       {"normalize_rewards": 1}, {"clip_rewards": 0}, {"preserve_replay_inputs": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NormalizedAgent(native, **{"observation_size": 2, "gamma": .9, **kwargs})

    def test_raw_replay_inputs_remain_separate_from_learning_inputs(self):
        native = ReplayRecordingAgent()
        agent = NormalizedAgent(native, 2, .5, preserve_replay_inputs=True)
        first = np.array([3., -4.], dtype=np.float64)
        agent.act_and_train(first, 0.)
        first[:] = 999.
        agent.act_and_train([6., -8.], 2.)
        agent.stop_episode([9., -12.], 4., terminated=False)
        self.assertFalse(native.training)
        self.assertFalse(native.ends)
        np.testing.assert_array_equal(native.replay_training[0][2], [3., -4.])
        np.testing.assert_array_equal(native.replay_training[1][2], [6., -8.])
        np.testing.assert_array_equal(native.replay_ends[0][2], [9., -12.])
        self.assertEqual([row[3] for row in native.replay_training], [0., 2.])
        self.assertEqual(native.replay_ends[0][3:], (4., False))
        self.assertNotEqual(native.replay_training[1][1], 2.)
        self.assertFalse(np.array_equal(native.replay_training[0][0], native.replay_training[0][2]))
        self.assertFalse(np.shares_memory(native.replay_training[0][0], native.replay_training[0][2]))
        self.assertTrue(agent.state_dict()["options"]["preserve_replay_inputs"])
        self.assertAlmostEqual(agent.statistics()["normalization_reward_count"], 2.0001)

    def test_raw_replay_requires_methods_and_new_native_symbols(self):
        with self.assertRaisesRegex(RuntimeError, "agent does not support"):
            NormalizedAgent(RecordingAgent(), 2, .9, preserve_replay_inputs=True)
        native = ReplayRecordingAgent()
        native.lib = object()  # Methods present in Python, old native library.
        with self.assertRaisesRegex(RuntimeError, "FFI library does not support"):
            NormalizedAgent(native, 2, .9, preserve_replay_inputs=True)
        self.assertFalse(native.replay_training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
