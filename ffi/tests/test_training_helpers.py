"""New fixed-budget helper contracts with fake environments/agents, no LibTorch.

Run: python -m unittest discover -s ffi/tests -p test_training_helpers.py -v
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import reinforcex_training as rt
from reinforcex_schedules import LinearLearningRateAgent


class Agent:
    def __init__(self):
        self.calls = []
        self.updates = 0
        self.rate = .01
        self.save_calls = 0
        self.eval_calls = 0
        self.closed = False

    def act_and_train(self, observation, reward):
        self.calls.append(("train", list(observation), reward))
        self.updates += 1
        return 0

    def stop_episode(self, observation, reward, *, terminated=True):
        self.calls.append(("stop", list(observation), reward, terminated))
        self.updates += 1

    def act(self, observation):
        self.eval_calls += 1
        return 0

    def statistics(self):
        return {"n_updates": self.updates, "learning_rate": self.rate}

    def set_learning_rate(self, value):
        self.calls.append(("set_lr", value))
        self.rate = value

    def save(self):
        self.save_calls += 1

    def close(self):
        self.closed = True


class Env:
    action_space = object()

    def __init__(self, lengths=(1000,), *, ending="terminated", reward=2., on_step=None):
        self.lengths = lengths
        self.ending = ending
        self.reward = reward
        self.on_step = on_step
        self.seeds = []
        self.actions = []
        self.steps = self.local_steps = 0
        self.closed = False

    def reset(self, **kwargs):
        self.seeds.append(kwargs)
        self.local_steps = 0
        return [0.], {}

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        self.local_steps += 1
        if self.on_step is not None:
            self.on_step(self)
        end = self.local_steps == self.lengths[min(len(self.seeds) - 1, len(self.lengths) - 1)]
        return ([float(self.local_steps)], self.reward,
                end and self.ending == "terminated", end and self.ending == "truncated", {})

    def close(self):
        self.closed = True


def direct_action(_agent, action, _space):
    return action


def train(agent, env, **kwargs):
    options = dict(budget=rt.TrainingBudget(steps_per_agent=5), seed=42, max_steps=500,
                   make_env=lambda _: env, action_adapter=direct_action, log_interval=None)
    options.update(kwargs)
    return rt.train_agent(agent, "Fake-v0", **options)


def evaluate(agent, env, **kwargs):
    options = dict(seed=900000, max_steps=500, make_env=lambda _: env, action_adapter=direct_action)
    options.update(kwargs)
    return rt.evaluate_agent(agent, "Fake-v0", **options)


class BudgetTests(unittest.TestCase):
    def parser(self):
        return rt.training_parser("example", steps_per_agent=204800, max_steps=500)

    def test_default_exact_per_worker_budget_and_explicit_episodes(self):
        parser = self.parser()
        args = parser.parse_args(["--parallel", "4"])
        budget = rt.resolve_budget(args, scheduled=True)
        self.assertEqual(budget.steps_per_agent, 204800)  # Not divided by 4.
        self.assertEqual(budget.schedule_horizon_steps, 204800)
        self.assertEqual(args.eval_episodes, 100)
        args = parser.parse_args(["--episodes", "10"])
        budget = rt.resolve_budget(args)
        self.assertEqual((budget.mode, budget.episodes, budget.steps_per_agent), ("episodes", 10, None))
        with self.assertRaisesRegex(ValueError, "explicit --schedule-steps"):
            rt.resolve_budget(args, scheduled=True)
        args = parser.parse_args(["--episodes", "10", "--schedule-steps", "1000"])
        self.assertEqual(rt.resolve_budget(args, scheduled=True).schedule_horizon_steps, 1000)

    def test_budget_conflicts_and_invalid_values(self):
        parser = self.parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--episodes", "10", "--steps-per-agent", "100"])
        for options in (["--steps-per-agent", "0"], ["--max-steps", "-1"], ["--parallel", "0"],
                        ["--eval-episodes", "0"], ["--eval-seed", "-1"],
                        ["--schedule-steps", "100"], ["--episodes", "0"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                rt.resolve_budget(parser.parse_args(options), scheduled=True)
        for values in ({}, {"steps_per_agent": 2, "episodes": 1}, {"steps_per_agent": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                rt.TrainingBudget(**values)

    def test_evaluation_only_requires_model_but_no_schedule_horizon(self):
        parser = self.parser()
        with self.assertRaisesRegex(ValueError, "requires --load-path"):
            rt.resolve_budget(parser.parse_args(["--eval-only"]))
        args = parser.parse_args(["--eval-only", "--load-path", "old.ot", "--episodes", "2"])
        self.assertEqual(rt.resolve_budget(args, scheduled=True).mode, "episodes")

    def test_parallel_weight_output_must_be_unique(self):
        parser = self.parser()
        with self.assertRaisesRegex(ValueError, "distinct"):
            rt.resolve_budget(parser.parse_args(["--parallel", "2", "--save-path", "same.ot"]))
        rt.resolve_budget(parser.parse_args(["--parallel", "2", "--save-path", "worker{agent_id}.ot"]))


class OutputAndCleanupTests(unittest.TestCase):
    def test_distinct_outputs_and_none_are_accepted_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rt.validate_output_paths([None, root / "w0.ot", root / "w0.learning_rate.json",
                                      root / "w1.ot", root / "results.json"])
            self.assertEqual(list(root.iterdir()), [])
            (root / "existing.ot").write_text("existing")
            rt.validate_output_paths([root / "existing.ot"])
            self.assertEqual((root / "existing.ot").read_text(), "existing")

    def test_identical_aliases_parent_child_and_normcase_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for paths in ([root / "w.ot", root / "sub/../w.ot"],
                          [root / "w.ot", root / "w.ot/results.json"],
                          [root / "w.ot/results.json", root / "w.ot"]):
                with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "overlap"):
                    rt.validate_output_paths(paths)
            with patch.object(rt.os.path, "normcase", side_effect=lambda value: value.casefold()):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    rt.validate_output_paths([root / "Worker.ot", root / "worker.ot"])

    def test_symlink_alias_and_existing_file_parent_fail_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "overlap"):
                rt.validate_output_paths([real / "worker.ot", alias / "worker.ot"])
            (root / "file").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "non-directory ancestor"):
                rt.validate_output_paths([root / "file/subdir/result.json"])
            self.assertEqual((root / "file").read_text(), "preserve")

    def test_darwin_rejects_case_and_nfc_nfd_aliases_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aliases = [("Model.ot", "model.ot"),
                       ("caf\u00e9.ot", "cafe\u0301.ot"),
                       ("CAF\u00c9.ot", "cafe\u0301.ot")]
            with patch.object(rt.sys, "platform", "darwin"):
                for left, right in aliases:
                    with self.subTest(left=left, right=right), self.assertRaisesRegex(ValueError, "overlap"):
                        rt.validate_output_paths([root / left, root / right])
            self.assertEqual(list(root.iterdir()), [])

    def test_darwin_alias_ancestor_overlap_is_rejected_in_both_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "CAF\u00c9", root / "cafe\u0301/weights.ot"
            with patch.object(rt.sys, "platform", "darwin"):
                for paths in ([parent, child], [child, parent]):
                    with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "overlap"):
                        rt.validate_output_paths(paths)
                # Canonical normalization does not strip accents or combine
                # distinct worker numbers; ordinary distinct paths still work.
                rt.validate_output_paths([root / "resume.ot", root / "r\u00e9sum\u00e9.ot",
                                          root / "worker0.ot", root / "worker1.ot"])
            self.assertEqual(list(root.iterdir()), [])

    def test_non_darwin_retains_platform_normcase_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(rt.sys, "platform", "linux"), \
                    patch.object(rt.os.path, "normcase", side_effect=lambda value: value):
                rt.validate_output_paths([root / "Model.ot", root / "model.ot",
                                          root / "caf\u00e9.ot", root / "cafe\u0301.ot"])
            self.assertEqual(list(root.iterdir()), [])

    def closers(self, failures):
        closed = []
        class Closer:
            def __init__(self, index):
                self.index = index
            def close(self):
                closed.append(self.index)
                if self.index in failures:
                    raise failures[self.index]
        return [Closer(i) for i in range(3)], closed

    def test_close_attempts_all_reverse_order_and_raises_first_cleanup_error(self):
        first, later = RuntimeError("last agent failed"), ValueError("first agent failed")
        agents, closed = self.closers({2: first, 0: later})
        with self.assertRaises(RuntimeError) as caught:
            rt.close_agents(agents)
        self.assertIs(caught.exception, first)
        self.assertEqual(closed, [2, 1, 0])
        self.assertEqual(first.cleanup_errors, (first, later))
        self.assertIsNotNone(later.__traceback__)
        if callable(getattr(first, "add_note", None)):
            self.assertIn("agent[2].close()", first.__notes__[0])
            self.assertIn("agent[0].close()", first.__notes__[1])

    def test_finally_preserves_original_exception_and_keeps_all_close_failures(self):
        primary = ValueError("training failed")
        cleanup = RuntimeError("close failed")
        agents, closed = self.closers({1: cleanup})
        with self.assertRaises(ValueError) as caught:
            try:
                raise primary
            finally:
                rt.close_agents(agents, sys.exc_info()[1])
        self.assertIs(caught.exception, primary)
        self.assertEqual(closed, [2, 1, 0])
        self.assertEqual(primary.cleanup_errors, (cleanup,))

    def test_cleanup_retention_without_add_note_support_and_clean_shutdown(self):
        class Python310StyleError(Exception):
            add_note = None
        primary, cleanup = Python310StyleError("original"), RuntimeError("cleanup")
        agents, closed = self.closers({2: cleanup})
        self.assertEqual(rt.close_agents(agents, primary), (cleanup,))
        self.assertEqual(primary.cleanup_errors, (cleanup,))
        self.assertEqual(closed, [2, 1, 0])
        agents, closed = self.closers({})
        self.assertEqual(rt.close_agents(agents), ())
        self.assertEqual(closed, [2, 1, 0])
        self.assertEqual(rt.close_agents([], primary), ())


class TrainingTests(unittest.TestCase):
    def test_exact_steps_seed_only_first_reset_and_final_reward_once(self):
        agent, env = Agent(), Env((2, 1000))
        flags = []

        def shape(reward, step, terminated, max_steps):
            flags.append((step, terminated, max_steps))
            return -.5 if terminated else reward / 10

        result = train(agent, env, reward_transform=shape)
        self.assertEqual(result["actual_steps"], 5)
        self.assertEqual(env.seeds, [{"seed": 42}, {}])
        self.assertEqual([c[2] for c in agent.calls if c[0] == "train"], [0., .2, 0., .2, .2])
        stops = [c for c in agent.calls if c[0] == "stop"]
        self.assertEqual(stops, [("stop", [2.], -.5, True), ("stop", [3.], .2, False)])
        records = result["episode_records"]
        self.assertEqual([r["return"] for r in records], [4., 6.])
        self.assertEqual([r["budget_cut"] for r in records], [False, True])
        self.assertEqual([r["natural_episode"] for r in records], [True, False])
        self.assertTrue(records[-1]["truncated"])
        self.assertEqual((result["completed_episodes"], result["natural_episodes"], result["budget_cuts"]), (1, 1, 1))
        self.assertEqual(flags, [(1, False, 500), (2, True, 500), (1, False, 500),
                                 (2, False, 500), (3, False, 500)])
        self.assertTrue(env.closed)
        self.assertFalse(agent.closed)

    def test_terminal_and_gym_truncation_at_exact_budget_are_complete(self):
        for ending, terminal in (("terminated", True), ("truncated", False)):
            with self.subTest(ending=ending):
                agent, env = Agent(), Env((5,), ending=ending)
                result = train(agent, env)
                record = result["episode_records"][0]
                self.assertTrue(record["natural_episode"])
                self.assertFalse(record["budget_cut"])
                self.assertEqual(agent.calls[-1], ("stop", [5.], 2., terminal))
                self.assertEqual(result["completed_episodes"], 1)
                self.assertEqual(len(env.seeds), 1)  # No reset after the final step.

    def test_explicit_episode_cap_is_distinct_from_natural_and_budget_cut(self):
        agent, env = Agent(), Env()
        result = train(agent, env, max_steps=2)
        self.assertEqual([r["length"] for r in result["episode_records"]], [2, 2, 1])
        self.assertEqual([r["max_steps_cut"] for r in result["episode_records"]], [True, True, False])
        self.assertEqual(result["natural_episodes"], 0)
        self.assertEqual(result["completed_episodes"], 2)
        self.assertTrue(all(not c[-1] for c in agent.calls if c[0] == "stop"))

    def test_episode_compatibility_mode_has_no_inferred_step_cap(self):
        result = train(Agent(), Env((1, 3, 2)), budget=rt.TrainingBudget(episodes=3), max_steps=10)
        self.assertEqual(result["actual_steps"], 6)
        self.assertEqual(result["natural_episodes"], 3)
        self.assertEqual(result["budget_cuts"], 0)

    def test_no_solved_stop_and_save_only_last_model_when_requested(self):
        agent, env = Agent(), Env((1,), reward=500.)
        result = train(agent, env, budget=rt.TrainingBudget(steps_per_agent=105), save_enabled=True)
        self.assertEqual(result["actual_steps"], 105)
        self.assertEqual(agent.save_calls, 1)
        self.assertEqual(len([c for c in agent.calls if c[0] == "stop"]), 105)

    def test_schedule_without_save_path_runs_and_applies_final_floor(self):
        native, env = Agent(), Env()
        wrapped = LinearLearningRateAgent(native, .01, total_steps=5, final_fraction=.05)
        train(wrapped, env)
        self.assertEqual(wrapped.current_steps, 5)
        self.assertEqual(native.rate, .0005)
        self.assertEqual(native.save_calls, 0)
        rates = [c[1] for c in native.calls if c[0] == "set_lr"]
        self.assertEqual(len(rates), 6)  # Five actions + one final stop.
        self.assertEqual(rates[0], .01)
        self.assertGreater(rates[-2], .0005)

    def test_callback_gets_copy_and_action_adapter_is_used(self):
        def callback(record):
            record["return"] = -999
        env = Env((5,))
        result = train(Agent(), env, on_episode=callback,
                       action_adapter=lambda agent, action, space: action + 3)
        self.assertEqual(result["episode_records"][0]["return"], 10.)
        self.assertEqual(env.actions, [3] * 5)

    def test_env_step_failure_closes_env_without_save_or_fabricated_stop(self):
        def fail(env):
            raise RuntimeError("environment failure")
        agent, env = Agent(), Env(on_step=fail)
        with self.assertRaisesRegex(RuntimeError, "environment failure"):
            train(agent, env, save_enabled=True)
        self.assertTrue(env.closed)
        self.assertFalse(agent.closed)
        self.assertEqual(agent.save_calls, 0)
        self.assertEqual([c for c in agent.calls if c[0] == "stop"], [])

    def test_reset_and_save_failures_still_close_environment(self):
        class BadReset(Env):
            def reset(self, **kwargs):
                raise RuntimeError("reset failure")
        class BadSave(Agent):
            def save(self):
                raise RuntimeError("save failure")
        for agent, env, message in ((Agent(), BadReset(), "reset failure"),
                                     (BadSave(), Env(), "save failure")):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                train(agent, env, save_enabled=True)
            self.assertTrue(env.closed)


class EvaluationTests(unittest.TestCase):
    def test_final_evaluation_defaults_100_raw_and_has_no_learning_calls(self):
        native = Agent()
        wrapped = LinearLearningRateAgent(native, .01, total_steps=5)
        train(wrapped, Env((5,)), reward_transform=lambda *args: -.25)
        before = list(native.calls), wrapped.state_dict(), native.statistics()
        env = Env((1,))
        result = evaluate(wrapped, env)
        self.assertEqual(result["mean_return"], 2.)
        self.assertEqual(result["episodes"], 100)
        self.assertEqual(result["std_return"], 0.)
        self.assertEqual(env.seeds, [{"seed": 900000 + i} for i in range(100)])
        self.assertEqual(before, (native.calls, wrapped.state_dict(), native.statistics()))
        self.assertEqual(native.save_calls, 0)
        self.assertEqual(native.eval_calls, 100)
        self.assertTrue(env.closed)
        self.assertTrue(result["observable_state_unchanged"])
        self.assertIn("native optimizer not serialized", result["freeze_scope"])

    def test_nested_real_normalization_and_schedule_remain_frozen(self):
        from reinforcex_normalization import NormalizedAgent
        native = Agent()
        normalized = NormalizedAgent(native, 1, .99)
        scheduled = LinearLearningRateAgent(normalized, .01, total_steps=5)
        train(scheduled, Env((2, 1000)))
        before = rt._observable_state(scheduled)
        result = evaluate(scheduled, Env((2,)), episodes=3)
        self.assertEqual(result["mean_return"], 4.)
        self.assertEqual(before, rt._observable_state(scheduled))
        self.assertGreater(normalized.state_dict()["observation"]["count"], 5)

    def test_detects_changed_native_statistics_and_closes_environment(self):
        class Mutating(Agent):
            def act(self, observation):
                self.updates += 1
                return 0
        env = Env((1,))
        with self.assertRaisesRegex(RuntimeError, "evaluation changed observable"):
            evaluate(Mutating(), env, episodes=1)
        self.assertTrue(env.closed)

    def test_detects_inner_wrapper_mutation_even_with_unchanged_outer_stats(self):
        class HiddenMutation:
            def __init__(self, agent):
                self._agent = agent
                self.count = 0
            def __getattr__(self, key):
                return getattr(self._agent, key)
            def state_dict(self):
                return {"count": self.count}
            def act(self, observation):
                self.count += 1
                return self._agent.act(observation)
        wrapped = LinearLearningRateAgent(HiddenMutation(Agent()), .01, total_steps=5)
        with self.assertRaisesRegex(RuntimeError, "evaluation changed observable"):
            evaluate(wrapped, Env((1,)), episodes=2)

    def test_eval_exception_cleanup_and_sample_standard_deviation(self):
        def fail(env):
            raise RuntimeError("eval step failed")
        agent, env = Agent(), Env(on_step=fail)
        with self.assertRaisesRegex(RuntimeError, "eval step failed"):
            evaluate(agent, env, episodes=1)
        self.assertTrue(env.closed)
        self.assertEqual(agent.calls, [])
        result = evaluate(agent, Env((1, 2)), episodes=2)
        self.assertAlmostEqual(result["std_return"], 2 ** .5)
        self.assertEqual(result["std_ddof"], 1)
        self.assertIsNone(evaluate(agent, Env((1,)), episodes=1)["std_return"])


class CancellationTests(unittest.TestCase):
    def test_pre_cancel_creates_no_environment_or_agent_activity(self):
        cancel = Event()
        cancel.set()
        agent = Agent()
        def forbidden(_):
            self.fail("environment created after cancellation")
        with self.assertRaises(rt.CancelledError):
            rt.train_agent(agent, "unused", budget=rt.TrainingBudget(steps_per_agent=1),
                           seed=42, max_steps=2, cancel_event=cancel, make_env=forbidden)
        with self.assertRaises(rt.CancelledError):
            rt.evaluate_agent(agent, "unused", seed=42, max_steps=2,
                              cancel_event=cancel, make_env=forbidden)
        self.assertEqual(agent.calls, [])

    def test_cancel_after_known_transition_flushes_once_and_does_not_save(self):
        cancel = Event()
        agent, records = Agent(), []
        env = Env(on_step=lambda _: cancel.set())
        with self.assertRaises(rt.CancelledError):
            train(agent, env, cancel_event=cancel, save_enabled=True, on_episode=records.append)
        self.assertEqual(agent.calls, [("train", [0.], 0.), ("stop", [1.], 2., False)])
        self.assertEqual(agent.save_calls, 0)
        self.assertTrue(env.closed)
        self.assertTrue(records[0]["cancelled"])

    def test_parallel_failure_joins_cleanup_and_preserves_originating_error(self):
        peer_in_step, allow_failure = Event(), Event()
        finished = []
        agents = [Agent(), Agent()]
        envs = []

        def work(index, cancel):
            try:
                if index == 0:
                    if not peer_in_step.wait(2):
                        raise AssertionError("peer did not enter environment")
                    allow_failure.set()
                    raise RuntimeError("originating worker failure")
                def on_step(_):
                    peer_in_step.set()
                    if not allow_failure.wait(2) or not cancel.wait(2):
                        raise AssertionError("failure did not cancel peer")
                env = Env(on_step=on_step)
                envs.append(env)
                return train(agents[index], env, cancel_event=cancel, save_enabled=True)
            finally:
                finished.append(index)

        with self.assertRaisesRegex(RuntimeError, "originating worker failure"):
            rt.run_workers(2, work)
        self.assertEqual(sorted(finished), [0, 1])  # Both finally blocks finished before raise.
        self.assertTrue(envs[0].closed)
        self.assertEqual(agents[1].calls[-1], ("stop", [1.], 2., False))
        self.assertEqual(agents[1].save_calls, 0)
        self.assertFalse(any(a.closed for a in agents))  # Caller owns teardown.

    def test_parallel_results_keep_worker_order_and_training_eval_barrier(self):
        done = [False, False]
        result = rt.run_workers(2, lambda i, cancel: done.__setitem__(i, True) or i * 10)
        self.assertEqual(result, [0, 10])
        self.assertTrue(all(done))
        evaluated = rt.run_workers(2, lambda i, cancel: (all(done), i))
        self.assertEqual(evaluated, [(True, 0), (True, 1)])

    def test_submission_failure_cancels_and_joins_already_started_worker(self):
        entered, cleaned_up = Event(), Event()
        observed_cancellation = []
        real_executor = rt.ThreadPoolExecutor

        class FailingSubmit(real_executor):
            submissions = 0

            def submit(self, *args, **kwargs):
                self.submissions += 1
                if self.submissions == 2:
                    if not entered.wait(2):
                        raise AssertionError("first worker did not start")
                    raise RuntimeError("cannot start new thread")
                return super().submit(*args, **kwargs)

        def worker(index, cancel):
            try:
                entered.set()
                observed_cancellation.append(cancel.wait(2))
            finally:
                cleaned_up.set()

        with patch.object(rt, "ThreadPoolExecutor", FailingSubmit):
            with self.assertRaisesRegex(RuntimeError, "cannot start new thread"):
                rt.run_workers(2, worker)
        self.assertTrue(cleaned_up.is_set())
        self.assertEqual(observed_cancellation, [True])


if __name__ == "__main__":
    unittest.main()
