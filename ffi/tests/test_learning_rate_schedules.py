"""Schedule contracts with fake agents; standard library only, no native build."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from reinforcex_schedules import LinearLearningRateAgent


class RecordingAgent:
    discrete = False
    action_bounds = (-1., 1.)

    def __init__(self):
        self.events = []
        self.rate = 9.
        self.native_stats = {"updates": 7.}
        self.action = object()
        self.fail = None
        self.saves = self.loads = 0

    def event(self, operation, *args):
        self.events.append((operation, *args))
        if self.fail == operation:
            raise RuntimeError("native " + operation + " failed")

    def set_learning_rate(self, value):
        self.event("set_learning_rate", value)
        self.rate = value

    def act_and_train(self, obs, reward):
        self.event("act_and_train", obs, reward)
        return self.action

    def act_and_train_with_replay_input(self, obs, reward, raw_obs, raw_reward):
        self.event("act_and_train_with_replay_input", obs, reward, raw_obs, raw_reward)
        return self.action

    def stop_episode(self, obs, reward, *, terminated=True):
        self.event("stop_episode", obs, reward, terminated)

    def stop_episode_with_replay_input(self, obs, reward, raw_obs, raw_reward, *, terminated=True):
        self.event("stop_episode_with_replay_input", obs, reward, raw_obs, raw_reward, terminated)

    def act(self, obs):
        self.event("act", obs)
        return self.action

    def statistics(self):
        return self.native_stats

    def save(self):
        self.event("save")
        self.saves += 1

    def load(self):
        self.event("load")
        self.loads += 1

    def close(self):
        self.event("close")


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "worker0.schedule.json"
        self.native = RecordingAgent()

    def wrapper(self, native=None, **options):
        return LinearLearningRateAgent(native or self.native, .01, total_steps=4,
                                       final_fraction=.2, **options)

    def rates(self, native=None):
        return [row[1] for row in (native or self.native).events if row[0] == "set_learning_rate"]

    def test_known_linear_sequence_stop_floor_and_extra_calls(self):
        agent = self.wrapper()
        self.assertEqual(self.native.events, [])
        obs = object()
        for index in range(4):
            self.assertIs(agent.act_and_train(obs, index), self.native.action)
        self.assertEqual(agent.current_steps, 4)
        for expected, actual in zip((.01, .008, .006, .004), self.rates()):
            self.assertAlmostEqual(expected, actual)
        agent.stop_episode(obs, 4., terminated=False)
        agent.stop_episode(obs, 4., terminated=False)
        agent.act_and_train(obs, 0.)
        agent.stop_episode(obs, 1.)
        self.assertEqual(self.rates()[-4:], [.002] * 4)
        self.assertEqual(agent.current_steps, 5)
        for index, event in enumerate(self.native.events):
            if event[0] in {"act_and_train", "stop_episode"}:
                self.assertEqual(self.native.events[index - 1][0], "set_learning_rate")

    def test_mid_episode_stop_does_not_advance_and_next_episode_uses_same_count(self):
        agent = self.wrapper()
        agent.act_and_train([1.], 0.)
        agent.stop_episode([2.], 1.)
        self.assertEqual(agent.current_steps, 1)
        self.assertAlmostEqual(self.rates()[-1], .008)
        agent.act_and_train([3.], 0.)
        self.assertAlmostEqual(self.rates()[-1], .008)
        self.assertEqual(agent.current_steps, 2)

    def test_eval_has_no_setter_or_schedule_mutation_and_delegates_attributes(self):
        agent = self.wrapper()
        agent.act_and_train(object(), 0.)
        state, stats = agent.state_dict(), agent.statistics()
        before = len(self.native.events)
        obs = object()
        for _ in range(3):
            self.assertIs(agent.act(obs), self.native.action)
        self.assertEqual(self.native.events[before:], [("act", obs)] * 3)
        self.assertEqual(state, agent.state_dict())
        self.assertEqual(stats, agent.statistics())
        self.assertEqual(agent.action_bounds, (-1., 1.))
        agent.close()
        self.assertEqual(self.native.events[-1], ("close",))

    def test_statistics_distinguish_next_rate_from_applied_rate_and_do_not_mutate_native(self):
        agent = self.wrapper()
        self.assertNotIn("learning_rate_last_applied", agent.statistics())
        self.assertEqual(agent.statistics()["scheduled_learning_rate"], .01)
        agent.act_and_train(None, 0.)
        stats = agent.statistics()
        self.assertEqual(stats["learning_rate_last_applied"], .01)
        self.assertAlmostEqual(stats["scheduled_learning_rate"], .008)
        self.assertEqual((stats["learning_rate_steps"], stats["learning_rate_total_steps"]), (1, 4))
        self.assertEqual(self.native.native_stats, {"updates": 7.})

    def test_separate_replay_inputs_and_terminal_flags_are_delegated_without_changes(self):
        agent = self.wrapper()
        normalized, raw = object(), object()
        self.assertIs(agent.act_and_train_with_replay_input(normalized, .25, raw, 25.), self.native.action)
        self.assertEqual(self.native.events[-1], ("act_and_train_with_replay_input", normalized, .25, raw, 25.))
        agent.stop_episode_with_replay_input(normalized, -.1, raw, -10., terminated=False)
        self.assertEqual(self.native.events[-1], ("stop_episode_with_replay_input", normalized, -.1, raw, -10., False))
        self.assertEqual(agent.current_steps, 1)
        self.assertAlmostEqual(self.rates()[-1], .008)

    def test_failed_setter_prevents_native_training_and_failed_calls_never_advance(self):
        agent = self.wrapper()
        self.native.fail = "set_learning_rate"
        with self.assertRaisesRegex(RuntimeError, "set_learning_rate"):
            agent.act_and_train(None, 0.)
        self.assertEqual(agent.current_steps, 0)
        self.assertNotIn("learning_rate_last_applied", agent.statistics())
        self.assertEqual([e[0] for e in self.native.events], ["set_learning_rate"])
        for name, call in (
            ("act_and_train", lambda: agent.act_and_train(None, 0.)),
            ("act_and_train_with_replay_input", lambda: agent.act_and_train_with_replay_input(None, 0., None, 0.)),
            ("stop_episode", lambda: agent.stop_episode(None, 0.)),
            ("stop_episode_with_replay_input", lambda: agent.stop_episode_with_replay_input(None, 0., None, 0.)),
        ):
            self.native.fail = name
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                call()
            self.assertEqual(agent.current_steps, 0)
        self.native.fail = None
        agent.act_and_train(None, 0.)
        self.assertEqual(agent.current_steps, 1)
        self.assertEqual(self.rates()[-1], .01)

    def test_invalid_parameters_missing_setter_and_underflow_floor(self):
        cases = [{"initial_learning_rate": v} for v in (True, 0, -1., float("inf"), float("nan"), "0.1", 10 ** 1000)]
        cases += [{"final_fraction": v} for v in (True, 0, -1., 1.01, float("nan"), "0.1")]
        cases += [{"total_steps": v} for v in (True, 0, -1, 1., "4")]
        cases += [{"initial_learning_rate": 5e-324, "final_fraction": .05}]
        for case in cases:
            options = {"initial_learning_rate": .01, "total_steps": 4, "final_fraction": .2, **case}
            with self.subTest(case=case), self.assertRaises(ValueError):
                LinearLearningRateAgent(self.native, **options)
        with self.assertRaisesRegex(RuntimeError, "set_learning_rate"):
            LinearLearningRateAgent(object(), .01, total_steps=4)
        self.assertEqual(self.native.events, [])

    def test_tiny_nonzero_floor_and_constant_schedule(self):
        agent = LinearLearningRateAgent(self.native, 1., total_steps=1, final_fraction=1e-300)
        agent.act_and_train(None, 0.)
        agent.stop_episode(None, 1.)
        self.assertEqual(self.rates(), [1., 1e-300])
        other = RecordingAgent()
        constant = LinearLearningRateAgent(other, .125, total_steps=2, final_fraction=1.)
        for _ in range(4):
            constant.act_and_train(None, 0.)
        self.assertEqual(self.rates(other), [.125] * 4)

    def test_save_constructor_load_and_explicit_load_preserve_state_without_setting_native_lr(self):
        agent = self.wrapper(save_path=self.path)
        for _ in range(3):
            agent.act_and_train(None, 0.)
        agent.save()
        self.assertEqual(self.native.saves, 1)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved, agent.state_dict())
        other = RecordingAgent()
        restored = self.wrapper(other, load_path=self.path)
        self.assertEqual(other.events, [])  # Native weights were loaded by the caller already.
        self.assertEqual(restored.state_dict(), saved)
        self.assertNotIn("learning_rate_last_applied", restored.statistics())
        restored.act(None)
        self.assertEqual(other.rate, 9.)  # Schedule JSON is not optimizer state.
        restored.act_and_train(None, 0.)
        self.assertAlmostEqual(self.rates(other)[-1], .004)
        self.assertEqual(restored.current_steps, 4)
        before_setters = len(self.rates(other))
        restored.load()
        self.assertEqual(other.loads, 1)
        self.assertEqual(restored.current_steps, 3)
        self.assertEqual(len(self.rates(other)), before_setters)
        self.assertNotIn("learning_rate_last_applied", restored.statistics())
        self.assertEqual(restored.state_dict(), saved)
        copy_of_state = restored.state_dict()
        copy_of_state["current_steps"] = 999
        self.assertEqual(restored.current_steps, 3)

    def test_malformed_or_mismatched_checkpoint_is_rejected_before_native_load(self):
        agent = self.wrapper(save_path=self.path, load_path=None)
        agent.save()
        agent.load_path = self.path
        original = json.loads(self.path.read_text())
        changes = [{"total_steps": 8}, {"initial_learning_rate": .02}, {"final_fraction": .3},
                   {"current_steps": -1}, {"current_steps": True}, {"current_steps": 1.},
                   {"schema": True}, {"schema": 2}, {"final_fraction": float("nan")},
                   {"initial_learning_rate": True}, {"checkpoint_scope": "full_training"}, {"extra": 1}]
        for change in changes:
            self.path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                agent.load()
            self.assertEqual(agent.current_steps, 0)
        missing = copy.deepcopy(original)
        del missing["total_steps"]
        self.path.write_text(json.dumps(missing))
        with self.assertRaises(ValueError):
            agent.load()
        self.assertEqual(self.native.loads, 0)
        self.assertEqual(self.rates(), [])

    def test_over_budget_checkpoint_keeps_counter_and_clamps_next_rate(self):
        agent = self.wrapper(save_path=self.path)
        for _ in range(7):
            agent.act_and_train(None, 0.)
        agent.save()
        loaded = self.wrapper(RecordingAgent(), load_path=self.path)
        self.assertEqual(loaded.current_steps, 7)
        self.assertEqual(loaded.scheduled_learning_rate, .002)

    def test_failed_native_save_keeps_published_json_and_failed_load_keeps_schedule(self):
        agent = self.wrapper(save_path=self.path)
        agent.save()
        before = self.path.read_bytes()
        agent.act_and_train(None, 0.)
        self.native.fail = "save"
        with self.assertRaisesRegex(RuntimeError, "save"):
            agent.save()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
        agent.load_path = self.path
        self.native.fail = "load"
        state, stats = agent.state_dict(), agent.statistics()
        with self.assertRaisesRegex(RuntimeError, "load"):
            agent.load()
        self.assertEqual(agent.state_dict(), state)
        self.assertEqual(agent.statistics(), stats)

    def test_missing_paths_do_not_trigger_native_weight_operations(self):
        agent = self.wrapper()
        with self.assertRaisesRegex(ValueError, "save_path"):
            agent.save()
        with self.assertRaisesRegex(ValueError, "load_path"):
            agent.load()
        self.assertEqual(self.native.events, [])


if __name__ == "__main__":
    unittest.main()
