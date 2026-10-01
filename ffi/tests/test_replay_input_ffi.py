"""Python ABI plumbing for opt-in PPO replay inputs; no native library required."""
import ctypes as C
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples'))
import reinforcex_ffi as rx

NEW_SYMBOLS = {'rx_agent_act_and_train_with_replay_input', 'rx_agent_stop_episode_with_replay_input'}


class Function:
    def __init__(self):
        self.calls = []
        self.callback = lambda *args: 0

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class Library:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.functions = {}

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(name)
        return self.functions.setdefault(name, Function())


class ReplayInputWrapperTests(unittest.TestCase):
    def test_optional_binding_accepts_old_library_and_legacy_calls_work(self):
        lib = Library(NEW_SYMBOLS)
        rx.configure_ffi(lib)
        agent = rx.Agent(lib, 17, 1, True)

        def legacy(handle, obs, count, reward, output, capacity):
            output[0] = 1.
            return 1

        lib.rx_agent_act_and_train.callback = legacy
        self.assertEqual(agent.act_and_train(np.ones(4), 0.), 1)
        with self.assertRaisesRegex(RuntimeError, 'does not support separate replay inputs'):
            agent.act_and_train_with_replay_input(np.ones(4), .1, np.ones(4), 1.)
        with self.assertRaisesRegex(RuntimeError, 'does not support separate replay inputs'):
            agent.stop_episode_with_replay_input(np.ones(4), .1, np.ones(4), 1.)
        self.assertEqual(len(lib.rx_agent_act_and_train.calls), 1)
        self.assertFalse(lib.rx_agent_stop_episode_with_terminal.calls)

    def test_new_symbol_ctypes_layout_matches_header(self):
        lib = Library()
        rx.configure_ffi(lib)
        pointer = C.POINTER(C.c_float)
        self.assertEqual(lib.rx_agent_act_and_train_with_replay_input.argtypes,
                         [C.c_uint64, pointer, C.c_uint64, C.c_float, pointer, C.c_uint64,
                          C.c_float, pointer, C.c_uint64])
        self.assertIs(lib.rx_agent_act_and_train_with_replay_input.restype, C.c_int64)
        self.assertEqual(lib.rx_agent_stop_episode_with_replay_input.argtypes,
                         [C.c_uint64, pointer, C.c_uint64, C.c_float, C.c_uint32,
                          pointer, C.c_uint64, C.c_float])
        self.assertIs(lib.rx_agent_stop_episode_with_replay_input.restype, C.c_int32)

    def test_act_preserves_distinct_streams_and_returns_action_kind(self):
        for discrete in (False, True):
            lib = Library()
            rx.configure_ffi(lib)
            agent = rx.Agent(lib, 17, 1 if discrete else 2, discrete)
            obs = np.array([[10., 20.], [30., 40.]]).T
            raw = np.array([1., 3., 2., 4.])

            def call(handle, learner, learner_len, reward, replay, replay_len, replay_reward, output, capacity):
                self.assertEqual(handle, 17)
                self.assertEqual((learner_len, replay_len, reward, replay_reward), (4, 4, .25, 2.5))
                np.testing.assert_array_equal(np.ctypeslib.as_array(learner, shape=(4,)), [10, 30, 20, 40])
                np.testing.assert_array_equal(np.ctypeslib.as_array(replay, shape=(4,)), raw)
                output[0] = 1. if discrete else .25
                if not discrete:
                    output[1] = -.5
                return capacity

            lib.rx_agent_act_and_train_with_replay_input.callback = call
            action = agent.act_and_train_with_replay_input(obs, .25, raw, 2.5)
            if discrete:
                self.assertIsInstance(action, int)
                self.assertEqual(action, 1)
            else:
                np.testing.assert_array_equal(action, [.25, -.5])
                self.assertEqual(action.dtype, np.float32)
            np.testing.assert_array_equal(obs, [[10, 30], [20, 40]])
            np.testing.assert_array_equal(raw, [1, 3, 2, 4])

    def test_stop_passes_common_terminal_flag_and_both_rewards(self):
        for terminated in (False, True):
            lib = Library()
            rx.configure_ffi(lib)
            agent = rx.Agent(lib, 17, 1, True)

            def call(handle, learner, learner_len, reward, flag, replay, replay_len, replay_reward):
                self.assertEqual((handle, learner_len, replay_len, reward, flag, replay_reward),
                                 (17, 4, 4, .25, int(terminated), 2.5))
                np.testing.assert_array_equal(np.ctypeslib.as_array(learner, shape=(4,)), [10.] * 4)
                np.testing.assert_array_equal(np.ctypeslib.as_array(replay, shape=(4,)), [1.] * 4)
                return 0

            lib.rx_agent_stop_episode_with_replay_input.callback = call
            agent.stop_episode_with_replay_input(np.full(4, 10.), .25, np.ones(4), 2.5, terminated=terminated)
            self.assertEqual(len(lib.rx_agent_stop_episode_with_replay_input.calls), 1)

    def test_native_rejection_and_unexpected_output_count_are_exposed(self):
        lib = Library()
        rx.configure_ffi(lib)
        agent = rx.Agent(lib, 17, 2, False)
        lib.rx_agent_act_and_train_with_replay_input.callback = lambda *args: -2
        lib.rx_agent_stop_episode_with_replay_input.callback = lambda *args: -2
        with self.assertRaisesRegex(RuntimeError, 'status -2'):
            agent.act_and_train_with_replay_input(np.ones(4), 0., np.ones(4), 0.)
        with self.assertRaisesRegex(RuntimeError, 'status -2'):
            agent.stop_episode_with_replay_input(np.ones(4), 0., np.ones(4), 0.)
        lib.rx_agent_act_and_train_with_replay_input.callback = lambda *args: 1
        with self.assertRaisesRegex(RuntimeError, 'expected 2 action values, got 1'):
            agent.act_and_train_with_replay_input(np.ones(4), 0., np.ones(4), 0.)


if __name__ == '__main__':
    unittest.main()
