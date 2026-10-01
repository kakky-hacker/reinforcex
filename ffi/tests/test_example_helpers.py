"""Gymnasium integration regressions; no native build or GPU is required.

Run with ``python -m unittest discover -s ffi/tests -p test_example_helpers.py -v``.
Native learning and concurrency checks live in the CPU validation harness.
"""

import argparse
import contextlib
import io
from pathlib import Path
import sys
from threading import Event
import time
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import reinforcex_ffi as rx
import train_half_cheetah_hybrid_ffi as hybrid
import train_ant_ppo_rnd_sac_shared_ffi as ant_shared


class RecordingAgent:
    discrete = True
    action_bounds = None

    def __init__(self):
        self.transitions = []
        self.ends = []
        self.saves = 0
        self.evaluations = 0

    def act_and_train(self, observation, reward):
        self.transitions.append((np.array(observation).copy(), reward))
        return 0

    def act(self, observation):
        self.evaluations += 1
        return 0

    def stop_episode(self, observation, reward, *, terminated=True):
        self.ends.append((np.array(observation).copy(), reward, terminated))

    def statistics(self):
        return {}

    def save(self):
        self.saves += 1


class RecordingEnv:
    action_space = gym.spaces.Discrete(2)

    def __init__(self, *, terminated=False, truncated=False):
        self.terminated = terminated
        self.truncated = truncated
        self.closed = False
        self.steps = 0
        self.seeds = []

    def reset(self, *, seed):
        self.seeds.append(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.array([float(self.steps)]), 2.0, self.terminated, self.truncated, {}

    def close(self):
        self.closed = True


def train(agent, **kwargs):
    arguments = dict(agent=agent, env_id="unused", agent_id=0, seed=10,
                     episodes=1, max_steps=2, log_interval=1)
    arguments.update(kwargs)
    return rx.train_gym_agent(**arguments)


class ExampleHelperTests(unittest.TestCase):
    def test_terminal_truncation_and_manual_limit_are_distinct(self):
        for terminated, truncated, expected_steps in ((True, False, 1), (False, True, 1), (False, False, 2)):
            with self.subTest(terminated=terminated, truncated=truncated):
                agent = RecordingAgent()
                env = RecordingEnv(terminated=terminated, truncated=truncated)
                shaping_flags = []

                def shape(reward, step, terminal, max_steps):
                    shaping_flags.append(terminal)
                    return reward / 10

                with patch.object(gym, "make", return_value=env), contextlib.redirect_stdout(io.StringIO()):
                    returns = train(agent, reward_transform=shape)
                self.assertEqual(returns, [expected_steps * 2.0])
                self.assertEqual(len(agent.transitions), expected_steps)
                self.assertEqual(agent.transitions[0][1], 0.0)
                self.assertEqual(len(agent.ends), 1)
                np.testing.assert_array_equal(agent.ends[0][0], [expected_steps])
                self.assertEqual(agent.ends[0][1:], (0.2, terminated))
                self.assertEqual(shaping_flags[-1], terminated)
                self.assertTrue(env.closed)

    def test_episode_reset_does_not_reuse_previous_reward(self):
        agent = RecordingAgent()
        env = RecordingEnv(truncated=True)
        with patch.object(gym, "make", return_value=env), contextlib.redirect_stdout(io.StringIO()):
            train(agent, episodes=3)
        self.assertEqual(env.seeds, [10, 11, 12])
        self.assertEqual([reward for _, reward in agent.transitions], [0.0, 0.0, 0.0])
        self.assertEqual(len(agent.ends), 3)

    def test_continuous_action_affine_mapping_preserves_policy_array(self):
        agent = rx.Agent(None, 0, 4, False, (-1.0, 1.0))
        space = gym.spaces.Box(
            low=np.array([[-2., 0.], [10., -8.]], dtype=np.float32),
            high=np.array([[2., 6.], [20., -2.]], dtype=np.float32),
        )
        policy_action = np.array([-1., 0., 1., 0.5], dtype=np.float32)
        actual = rx.gym_action(agent, policy_action, space)
        np.testing.assert_array_equal(actual, [[-2., 3.], [20., -3.5]])
        np.testing.assert_array_equal(policy_action, [-1., 0., 1., 0.5])
        self.assertEqual(actual.shape, (2, 2))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(space.contains(actual))

    def test_ppo_custom_action_bounds_map_correctly(self):
        agent = rx.Agent(None, 0, 1, False, (-2., 2.))
        space = gym.spaces.Box(-10., 10., shape=(1,), dtype=np.float32)
        np.testing.assert_array_equal(rx.gym_action(agent, [1.], space), [5.])

    def test_discrete_action_start_is_applied(self):
        space = gym.spaces.Discrete(3, start=5)
        self.assertEqual(rx.gym_action(RecordingAgent(), 2, space), 7)
        with self.assertRaisesRegex(ValueError, "invalid discrete"):
            rx.gym_action(RecordingAgent(), 3, space)

    def test_invalid_continuous_actions_fail_before_env_step(self):
        agent = rx.Agent(None, 0, 1, False, (-1., 1.))
        space = gym.spaces.Box(-2., 2., shape=(1,), dtype=np.float32)
        for action in ([np.nan], [np.inf], [0., 1.]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                rx.gym_action(agent, action, space)
        with self.assertRaisesRegex(ValueError, "finite Gymnasium"):
            rx.gym_action(agent, [0.], gym.spaces.Box(-np.inf, np.inf, shape=(1,)))

    def test_evaluation_never_trains_or_finalizes_training(self):
        agent = RecordingAgent()
        env = RecordingEnv(truncated=True)
        with patch.object(gym, "make", return_value=env), contextlib.redirect_stdout(io.StringIO()):
            returns = rx.evaluate_gym_agent(agent=agent, env_id="unused", agent_id=0,
                                           seed=11, episodes=3, max_steps=5)
        self.assertEqual(returns, [2., 2., 2.])
        self.assertEqual(agent.evaluations, 3)
        self.assertEqual(agent.transitions, [])
        self.assertEqual(agent.ends, [])
        self.assertEqual(agent.saves, 0)
        self.assertTrue(env.closed)

    def test_training_and_evaluation_both_map_continuous_actions(self):
        class ContinuousAgent(RecordingAgent):
            discrete = False
            action_bounds = (-1., 1.)

            def act_and_train(self, observation, reward):
                return np.array([0.5])

            def act(self, observation):
                return np.array([0.5])

        class ContinuousEnv(RecordingEnv):
            action_space = gym.spaces.Box(-2., 2., shape=(1,), dtype=np.float32)

            def step(self, action):
                np.testing.assert_array_equal(action, [1.])
                return super().step(action)

        for evaluation in (False, True):
            with self.subTest(evaluation=evaluation):
                env = ContinuousEnv(truncated=True)
                with patch.object(gym, "make", return_value=env), contextlib.redirect_stdout(io.StringIO()):
                    if evaluation:
                        rx.evaluate_gym_agent(agent=ContinuousAgent(), env_id="unused", agent_id=0,
                                              seed=0, episodes=1, max_steps=2)
                    else:
                        train(ContinuousAgent())
                self.assertEqual(env.steps, 1)

    def test_later_worker_failure_cancels_earlier_training_and_closes_env(self):
        ready = Event()

        class SlowEnv(RecordingEnv):
            def step(self, action):
                ready.set()
                time.sleep(0.002)
                return super().step(action)

        env = SlowEnv()
        agent = RecordingAgent()

        def worker(index):
            if index == 0:
                return train(agent, episodes=100000, max_steps=1000)
            if not ready.wait(timeout=2):
                raise AssertionError("training did not start")
            raise RuntimeError("worker one failed")

        started = time.monotonic()
        with patch.object(gym, "make", return_value=env), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "worker one failed"):
                rx.run_parallel(2, worker)
        self.assertLess(time.monotonic() - started, 2)
        self.assertLess(env.steps, 100)
        self.assertTrue(env.closed)
        self.assertEqual(len(agent.ends), 1)
        self.assertFalse(agent.ends[0][2])
        self.assertEqual(agent.saves, 0)

    def test_parallel_results_keep_worker_order(self):
        first = Event()

        def worker(index):
            if index == 0:
                first.wait(timeout=2)
            else:
                first.set()
            return index * 10

        self.assertEqual(rx.run_parallel(2, worker), [0, 10])
        self.assertEqual(rx.run_parallel(1, lambda index: 3), [3])

    def test_invalid_loop_arguments_fail_before_creating_env(self):
        with patch.object(gym, "make") as make:
            for name in ("episodes", "max_steps", "log_interval", "solved_window"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    train(RecordingAgent(), **{name: 0})
            with self.assertRaises(ValueError):
                rx.evaluate_gym_agent(agent=RecordingAgent(), env_id="unused", agent_id=0,
                                      seed=0, episodes=1, max_steps=0)
            with self.assertRaises(ValueError):
                rx.run_parallel(0, lambda index: None)
            make.assert_not_called()

    def test_parallel_checkpoint_collision_is_rejected(self):
        args = argparse.Namespace(episodes=1, max_steps=1, log_interval=1, parallel=2,
                                  save_path="same.ot", load_path="same.ot")
        with self.assertRaisesRegex(ValueError, "agent_id"):
            rx.validate_training_args(args)
        args.save_path = "worker_{agent_id}/../same.ot"
        with self.assertRaisesRegex(ValueError, "distinct"):
            rx.validate_training_args(args)
        args.save_path = "worker_{agent_id}.ot"
        rx.validate_training_args(args)
        args.save_path = None
        rx.validate_training_args(args)

    def test_hybrid_candidate_checkpoint_collision_is_rejected(self):
        args = hybrid.parser().parse_args(["--ppo-save-path", "same.ot"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            hybrid.validate_args(args)
        hybrid.validate_args(hybrid.parser().parse_args([]))

    def test_hybrid_examples_allow_cpu_by_default(self):
        self.assertFalse(hybrid.parser().parse_args([]).require_cuda)
        self.assertFalse(ant_shared.parser().parse_args([]).require_cuda)

    def test_half_cheetah_candidates_use_the_same_evaluation_seeds(self):
        seeds = []
        args = hybrid.parser().parse_args([])

        class EvaluatedAgent:
            def close(self):
                pass

        def evaluate(**kwargs):
            seeds.append(kwargs["seed"])
            return [10., 20.]

        with patch.object(hybrid, "create_ppo", return_value=EvaluatedAgent()), \
                patch.object(hybrid, "evaluate_gym_agent", side_effect=evaluate):
            best, means, _ = hybrid.evaluate_candidates(None, args, "ppo", None, 3, "model_{agent_id}")
        self.assertEqual(best, 0)
        self.assertEqual(means, [15., 15., 15.])
        self.assertEqual(seeds, [args.seed + 20_000_000] * 3)

    def test_wrapper_forwards_terminal_flag_and_preserves_default(self):
        calls = []

        class Library:
            def rx_agent_stop_episode_with_terminal(self, handle, obs, size, reward, terminal):
                calls.append((handle, size, obs[0], reward, terminal))
                return 0

        agent = rx.Agent(Library(), 42, 1, True)
        agent.stop_episode(np.array([1.]), 2.)
        agent.stop_episode(np.array([3.]), 4., terminated=False)
        self.assertEqual(calls, [(42, 1, 1., 2., 1), (42, 1, 3., 4., 0)])


if __name__ == "__main__":
    unittest.main()
