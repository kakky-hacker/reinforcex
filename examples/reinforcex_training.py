"""Fixed-budget training and final raw-return evaluation for CPU FFI examples.

This module imports only the standard library. Gymnasium and the existing action
adapter are imported lazily; native libraries are never loaded here. Agents are
duck typed and remain owned by the caller. Environments are always closed here.

Typical integration::

    parser = training_parser(__doc__, steps_per_agent=204800, max_steps=500)
    args = parser.parse_args()
    budget = resolve_budget(args, scheduled=True)
    # Construct each agent and its schedule using budget.schedule_horizon_steps.
    trained = run_workers(args.parallel, lambda i, cancel: train_agent(
        agents[i], "CartPole-v1", budget=budget, seed=worker_seeds[i],
        max_steps=args.max_steps, cancel_event=cancel,
        save_enabled=bool(args.save_path), agent_id=i))
    # This barrier matters when agents share replay or RND state.
    evaluated = run_workers(args.parallel, lambda i, cancel: evaluate_agent(
        agents[i], "CartPole-v1", seed=args.eval_seed,
        episodes=args.eval_episodes, max_steps=args.max_steps,
        cancel_event=cancel, agent_id=i))

There is no solved early stop or best-model selection. Evaluation uses the last
model and raw environment rewards. Its freeze check covers exposed statistics
and Python wrapper state, not inaccessible native optimizer/parameter storage.
Only call evaluation after every training worker has finished. Loading models,
choosing legacy models, seeding model initialization, and exact-resume semantics
are deliberately left to the examples. A budget counts this invocation's steps.
Parallel workers do not imply independent LibTorch global RNG or fully seeded
Rust sampling; identical seeds alone do not guarantee identical trajectories.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
from threading import Event
import time
import unicodedata


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TrainingBudget:
    """Exactly one stopping criterion, shared by neither workers nor agents.

    Episode mode is a compatibility convenience, not the validated step mode.
    A schedule horizon cannot be inferred from episode count: when needed it
    must be supplied explicitly. It is a schedule setting, not a step bound.
    """
    steps_per_agent: int | None = None
    episodes: int | None = None
    schedule_steps: int | None = None

    def __post_init__(self):
        if (self.steps_per_agent is None) == (self.episodes is None):
            raise ValueError("choose exactly one of steps_per_agent and episodes")
        for name in ("steps_per_agent", "episodes", "schedule_steps"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(value, name)
        if (self.steps_per_agent is not None and self.schedule_steps is not None
                and self.schedule_steps != self.steps_per_agent):
            raise ValueError("step mode schedule horizon must equal steps_per_agent")

    @property
    def mode(self):
        return "steps" if self.steps_per_agent is not None else "episodes"

    @property
    def schedule_horizon_steps(self):
        if self.steps_per_agent is not None:
            return self.steps_per_agent
        if self.schedule_steps is None:
            raise ValueError("episode mode with a schedule requires explicit --schedule-steps")
        return self.schedule_steps


def training_parser(description, *, steps_per_agent, max_steps, log_interval=25):
    """New opt-in parser; the old reinforcex_ffi parser is unaffected.

    An explicit --episodes switches mode without retaining the default step cap.
    Eval seed 900000 is a public/reused example block, not a new heldout claim.
    Example-specific model, reward, and load options may be added by the caller.
    """
    for name, value in (("steps_per_agent", steps_per_agent), ("max_steps", max_steps),
                        ("log_interval", log_interval)):
        _positive_int(value, name)
    parser = argparse.ArgumentParser(description=description)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--steps-per-agent", type=int,
                         help=f"exact environment steps per worker (default: {steps_per_agent})")
    budgets.add_argument("--episodes", type=int,
                         help="legacy episode budget; not equivalent to a fixed step budget")
    parser.set_defaults(_default_steps_per_agent=steps_per_agent)
    parser.add_argument("--schedule-steps", type=int,
                        help="explicit LR horizon for episode mode; never inferred from episode count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=max_steps)
    parser.add_argument("--log-interval", type=int, default=log_interval)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--save-path", help="weights path; replace {agent_id} per worker")
    parser.add_argument("--load-path", help="model loading policy is defined by the example")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=900000)
    return parser


def resolve_budget(args, *, scheduled=False):
    """Validate common CLI fields and resolve a per-worker budget.

    Evaluation-only mode needs no schedule horizon or learning-rate capability.
    Model/schedule restoration and save-sidecar collision checks remain caller
    responsibilities; this checks the common weight-path template.
    """
    for name in ("max_steps", "log_interval", "parallel", "eval_episodes"):
        _positive_int(getattr(args, name), "--" + name.replace("_", "-"))
    for name in ("seed", "eval_seed"):
        value = getattr(args, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a nonnegative integer")
    if args.steps_per_agent is not None and args.episodes is not None:
        raise ValueError("--steps-per-agent and --episodes are mutually exclusive")
    steps = args.steps_per_agent
    if steps is None and args.episodes is None:
        steps = args._default_steps_per_agent
    budget = TrainingBudget(steps, args.episodes, args.schedule_steps)
    if args.eval_only:
        if not args.load_path:
            raise ValueError("--eval-only requires --load-path")
    elif scheduled:
        budget.schedule_horizon_steps  # Reject an implicit episode-based estimate.
    elif args.schedule_steps is not None:
        raise ValueError("--schedule-steps requires a scheduled agent")
    if args.parallel > 1 and args.save_path:
        paths = [os.path.normcase(str(Path(args.save_path.replace("{agent_id}", str(i))).resolve()))
                 for i in range(args.parallel)]
        if len(set(paths)) != len(paths):
            raise ValueError("--save-path must resolve to distinct paths per worker")
    return budget


def validate_output_paths(paths):
    """Reject overlapping output names before creating agents or artifacts.

    Pass every resolved worker weight, sidecar, and result path together. None
    entries are ignored. Paths are resolved (including existing symlinks) and
    normcase'd; identical names and either direction of ancestor overlap fail.
    On Darwin, NFC normalization and casefold also reject Unicode/case aliases
    conservatively, even on a case-sensitive volume where both could exist.
    An existing non-directory ancestor also fails. Existing output files are
    allowed here: the caller decides whether overwriting them is appropriate.
    This preflight is not a lock against subsequent filesystem changes.
    """
    resolved = []
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"output path {value!s} has a non-directory ancestor: {parent}")
        name = os.path.normcase(str(path))
        if sys.platform == "darwin":
            name = unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())
        normalized = Path(name)
        for previous, previous_name in resolved:
            if (normalized == previous or normalized in previous.parents
                    or previous in normalized.parents):
                raise ValueError(f"output paths overlap: {previous_name!s} and {value!s}")
        resolved.append((normalized, value))


def close_agents(agents, primary_error=None):
    """Attempt every close in reverse order without hiding an earlier failure.

    Intended for ``finally: close_agents(agents, sys.exc_info()[1])``. If there
    is a primary error this function returns and lets the caller propagate it;
    otherwise it raises the first cleanup error, after trying every agent.
    The preserved/raised exception's ``cleanup_errors`` tuple retains all close
    exceptions, including tracebacks, on Python 3.10+. Python 3.11+ also receives
    readable exception notes. The return value is the tuple of cleanup errors
    (empty on clean shutdown). Agents must not be closed concurrently elsewhere.
    """
    errors = []
    for index, agent in reversed(list(enumerate(agents))):
        try:
            agent.close()
        except BaseException as error:
            errors.append((index, error))
    if not errors:
        return ()
    target = primary_error if primary_error is not None else errors[0][1]
    failures = tuple(error for _, error in errors)
    target.cleanup_errors = getattr(target, "cleanup_errors", ()) + failures
    add_note = getattr(target, "add_note", None)
    if callable(add_note):
        for index, error in errors:
            add_note(f"agent[{index}].close() failed: {type(error).__name__}: {error}")
    if primary_error is None:
        raise target
    return failures


def _make_env(env_id):
    import gymnasium as gym
    return gym.make(env_id)


def _action(agent, value, space):
    from reinforcex_ffi import gym_action
    return gym_action(agent, value, space)


def _cancelled(event):
    return event is not None and event.is_set()


def _check_cancelled(event):
    if _cancelled(event):
        raise CancelledError("another worker failed or training was cancelled")


def _reward(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("environment/transformed reward must be finite")
    return result


def train_agent(agent, env_id, *, budget, seed, max_steps, reward_transform=None,
                save_enabled=False, agent_id=0, log_interval=25, cancel_event=None,
                on_episode=None, make_env=None, action_adapter=None):
    """Train one stream, flushing every known final transition exactly once.

    The first reset receives seed; subsequent training resets have no seed.
    Records separate Gym episode ends, explicit per-episode time caps, partial
    budget cuts, and cancellation. Raw and transformed returns are both logged.
    Saving is opt-in and happens once, after successful training, never on error.
    A cancelled stream flushes its last *known* transition with bootstrap unless
    the environment terminated; failed env/native calls cannot be rolled back.
    This function closes its environment but never closes the caller's agent.
    """
    if not isinstance(budget, TrainingBudget):
        raise TypeError("budget must be TrainingBudget")
    _positive_int(max_steps, "max_steps")
    if log_interval is not None:
        _positive_int(log_interval, "log_interval")
    if type(save_enabled) is not bool:
        raise ValueError("save_enabled must be boolean")
    _check_cancelled(cancel_event)
    env = (make_env or _make_env)(env_id)
    adapt = action_adapter or _action
    records, steps = [], 0
    natural_returns = deque(maxlen=100)
    started = time.monotonic()
    try:
        while True:
            _check_cancelled(cancel_event)
            observation, _ = env.reset(seed=seed) if not records else env.reset()
            length, raw_return, train_return, previous_reward = 0, 0.0, 0.0, 0.0
            terminated = truncated = capped = budget_cut = False
            while True:
                # Cancellation may arrive between the previous step's check and
                # this action. Flush that previous transition, without a new act.
                if _cancelled(cancel_event):
                    if length == 0:
                        _check_cancelled(cancel_event)
                    break
                action = agent.act_and_train(observation, previous_reward)
                observation, reward, terminated, truncated, _ = env.step(
                    adapt(agent, action, env.action_space))
                terminated, truncated = bool(terminated), bool(truncated)
                steps += 1
                length += 1
                raw = _reward(reward)
                previous_reward = _reward(reward_transform(raw, length, terminated, max_steps)
                                          if reward_transform is not None else raw)
                raw_return += raw
                train_return += previous_reward
                natural_end = terminated or truncated
                capped = length == max_steps and not natural_end
                budget_cut = (budget.steps_per_agent == steps and not natural_end and not capped)
                if natural_end or capped or budget_cut or _cancelled(cancel_event):
                    break
            agent.stop_episode(observation, previous_reward, terminated=terminated)
            record = {"episode": len(records) + 1, "environment_steps": steps,
                      "length": length, "return": raw_return, "train_return": train_return,
                      "terminated": terminated, "environment_truncated": truncated,
                      "truncated": truncated or capped or budget_cut or _cancelled(cancel_event),
                      "max_steps_cut": capped, "budget_cut": budget_cut,
                      "cancelled": _cancelled(cancel_event),
                      "natural_episode": terminated or truncated}
            records.append(record)
            if record["natural_episode"]:
                natural_returns.append(raw_return)
            if on_episode is not None:
                on_episode(dict(record))
            _check_cancelled(cancel_event)
            if log_interval is not None and (len(records) == 1 or len(records) % log_interval == 0):
                recent = statistics.mean(natural_returns) if natural_returns else None
                print(f"agent={agent_id} episode={len(records)} environment_steps={steps} "
                      f"return={raw_return:.2f} train_return={train_return:.2f} "
                      f"budget_cut={budget_cut} natural_mean100={recent}", flush=True)
            if steps == budget.steps_per_agent or len(records) == budget.episodes:
                break
    finally:
        env.close()
    _check_cancelled(cancel_event)
    if save_enabled:
        agent.save()
    return {"mode": budget.mode, "actual_steps": steps, "episode_records": records,
            "completed_episodes": sum(not r["budget_cut"] and not r["cancelled"] for r in records),
            "natural_episodes": sum(r["natural_episode"] for r in records),
            "budget_cuts": sum(r["budget_cut"] for r in records),
            "seconds": time.monotonic() - started}


def _observable_state(agent):
    """Canonical JSON snapshot of the supported FFI/Python wrapper contract.

    Following owned _agent links captures inner normalization/schedule wrappers
    even when an outer wrapper's state_dict hides them. The small runtime fields
    below are not persisted by NormalizedAgent but must also stay fixed in eval.
    This is evidence about exposed state, not a native optimizer serialization.
    """
    result = {"statistics": agent.statistics(), "wrappers": []}
    current, seen = agent, set()
    while current is not None:
        if id(current) in seen:
            raise ValueError("cyclic agent wrapper chain")
        seen.add(id(current))
        own = getattr(current, "__dict__", {})
        state_method = getattr(type(current), "state_dict", None)
        if callable(state_method):
            result["wrappers"].append({"class": type(current).__name__, "state": state_method(current),
                "runtime": {key: own[key] for key in
                            ("_discounted_return", "_pending_action", "_last_applied_learning_rate")
                            if key in own}})
        current = own.get("_agent")
    return json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":"))


def evaluate_agent(agent, env_id, *, seed, episodes=100, max_steps,
                   agent_id=0, cancel_event=None, make_env=None, action_adapter=None):
    """Evaluate the last model deterministically on raw rewards, with no training.

    Each episode resets with seed+i. Agent.act is the deterministic FFI path.
    No reward transform, save, LR setter, train action or stop_episode is used.
    A successful return asserts unchanged exposed statistics/wrapper state; it
    cannot prove that unexposed optimizer tensors have not changed internally.
    Returned SD is sample SD (ddof=1), None for a single evaluation episode.
    """
    _positive_int(episodes, "episodes")
    _positive_int(max_steps, "max_steps")
    _check_cancelled(cancel_event)
    before = _observable_state(agent)
    env = (make_env or _make_env)(env_id)
    adapt = action_adapter or _action
    records = []
    try:
        for index in range(episodes):
            _check_cancelled(cancel_event)
            observation, _ = env.reset(seed=seed + index)
            raw_return = 0.0
            for length in range(1, max_steps + 1):
                _check_cancelled(cancel_event)
                observation, reward, terminated, truncated, _ = env.step(
                    adapt(agent, agent.act(observation), env.action_space))
                raw_return += _reward(reward)
                if terminated or truncated:
                    break
            records.append({"seed": seed + index, "return": raw_return, "length": length,
                            "terminated": bool(terminated), "environment_truncated": bool(truncated),
                            "max_steps_cut": not (terminated or truncated) and length == max_steps})
    finally:
        env.close()
    _check_cancelled(cancel_event)
    if _observable_state(agent) != before:
        raise RuntimeError("evaluation changed observable agent statistics or wrapper state")
    returns = [r["return"] for r in records]
    return {"agent_id": agent_id, "episodes": episodes, "episode_records": records,
            "mean_return": statistics.mean(returns),
            "std_return": statistics.stdev(returns) if episodes > 1 else None,
            "std_ddof": 1, "min_return": min(returns), "max_return": max(returns),
            "deterministic": True, "reward": "raw", "observable_state_unchanged": True,
            "freeze_scope": "exposed statistics and Python wrapper state; native optimizer not serialized"}


def run_workers(worker_count, worker):
    """Call worker(index, cancel_event), preserving order and joining on failure.

    A worker failure signals peers immediately. The first non-cancellation error
    is propagated only after every started worker's finally block has finished.
    Agent/replay destruction is therefore safe in the caller's outer finally.
    Separate calls for training and evaluation provide a shared-state barrier.
    """
    _positive_int(worker_count, "worker_count")
    cancel = Event()

    def invoke(index):
        try:
            _check_cancelled(cancel)
            return worker(index, cancel)
        except BaseException:
            cancel.set()
            raise

    if worker_count == 1:
        return [invoke(0)]
    results, failure = [None] * worker_count, None
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        try:
            # Thread creation/submission can fail after a previous worker has
            # started. Signal that worker before the executor joins it, too.
            for index in range(worker_count):
                futures[executor.submit(invoke, index)] = index
            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                except BaseException as error:
                    if failure is None or (isinstance(failure, CancelledError)
                                           and not isinstance(error, CancelledError)):
                        failure = error
                    cancel.set()
                    for pending in futures:
                        pending.cancel()
        except BaseException:
            cancel.set()
            for future in futures:
                future.cancel()
            raise
    if failure is not None:
        raise failure
    return results
