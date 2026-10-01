"""Opt-in learning-rate schedules for agents exposing ``set_learning_rate``.

No native library, NumPy, or ML runtime is imported here. One wrapper belongs to
one training stream. Each successful training action advances the schedule once;
the caller is responsible for taking one environment step per action.

Saving delegates native weight saving and publishes schedule JSON separately.
These files are not a multi-file atomic checkpoint. Restoring the schedule does
not restore optimizer moments, replay/rollout buffers, RNG, or a live episode,
and does not claim to resume the original training process exactly.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile


_SCHEMA = 1
_STATE_FIELDS = {"schema", "initial_learning_rate", "final_fraction", "total_steps",
                 "current_steps", "checkpoint_scope"}


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _parameters(initial_learning_rate, final_fraction, total_steps):
    initial = _positive_float(initial_learning_rate, "initial_learning_rate")
    fraction = _positive_float(final_fraction, "final_fraction")
    if fraction > 1.0:
        raise ValueError("final_fraction must be at most 1")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    # The native setter requires strictly positive rates, including the floor.
    if initial * fraction <= 0.0:
        raise ValueError("the final learning rate underflows to zero")
    return initial, fraction, total_steps


class LinearLearningRateAgent:
    """Linearly anneal native learning rate on training calls only.

    ``current_steps`` counts successful ``act_and_train`` calls, including the
    separate-replay-input variant. At 0-based action index ``s``, the rate is
    linearly interpolated from ``initial_learning_rate`` to its product with
    ``final_fraction`` using ``min(s / total_steps, 1)``. ``stop_episode`` applies
    the rate at the current completed count without advancing that count. Thus
    the final stop at ``total_steps`` applies the exact floor. Repeated stops or
    calls beyond the budget do not push the rate below the floor.

    If setting the rate succeeds but the following native training call fails,
    the counter stays unchanged and the new rate remains applied. This wrapper
    does not roll back native state or make a training call atomic.

    Construction, loading, and evaluation never call the native rate setter.
    A constructor ``load_path`` restores schedule JSON only: the supplied native
    agent must already have loaded its weights. Explicit ``load()`` validates
    JSON before delegating native loading, then restores the schedule counter.
    It leaves the optimizer's rate alone until the next training call.

    ``scheduled_learning_rate`` is the next training call's planned rate.
    ``learning_rate_last_applied`` appears only after a successful setter call
    in this wrapper and is cleared by loading; it is not an optimizer snapshot.
    Native statistics, including any native rate report, remain available.
    """

    def __init__(self, agent, initial_learning_rate, *, total_steps,
                 final_fraction=.05, save_path=None, load_path=None):
        initial, fraction, total = _parameters(initial_learning_rate, final_fraction, total_steps)
        if not callable(getattr(agent, "set_learning_rate", None)):
            raise RuntimeError("agent does not support set_learning_rate")
        self._agent = agent
        self.initial_learning_rate = initial
        self.final_fraction = fraction
        self.total_steps = total
        self.current_steps = 0
        self.save_path = Path(save_path) if save_path is not None else None
        self.load_path = Path(load_path) if load_path is not None else None
        self._last_applied_learning_rate = None
        if self.load_path is not None:
            self.current_steps = self._read_state()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_agent"), name)

    @property
    def scheduled_learning_rate(self):
        floor = self.initial_learning_rate * self.final_fraction
        if self.current_steps >= self.total_steps:
            return floor
        progress = self.current_steps / self.total_steps
        # This convex form avoids cancellation to zero at very small floors.
        return (1.0 - progress) * self.initial_learning_rate + progress * floor

    def _before_training(self):
        rate = self.scheduled_learning_rate
        self._agent.set_learning_rate(rate)
        self._last_applied_learning_rate = rate

    def act_and_train(self, observation, reward):
        self._before_training()
        result = self._agent.act_and_train(observation, reward)
        self.current_steps += 1
        return result

    def act_and_train_with_replay_input(self, observation, reward,
                                       replay_observation, replay_reward):
        operation = self._agent.act_and_train_with_replay_input
        self._before_training()
        result = operation(observation, reward, replay_observation, replay_reward)
        self.current_steps += 1
        return result

    def stop_episode(self, observation, reward, *, terminated=True):
        self._before_training()
        return self._agent.stop_episode(observation, reward, terminated=terminated)

    def stop_episode_with_replay_input(self, observation, reward,
                                      replay_observation, replay_reward, *, terminated=True):
        operation = self._agent.stop_episode_with_replay_input
        self._before_training()
        return operation(observation, reward, replay_observation, replay_reward, terminated=terminated)

    def act(self, observation):
        return self._agent.act(observation)

    def statistics(self):
        result = dict(self._agent.statistics())
        result.update(scheduled_learning_rate=self.scheduled_learning_rate,
                      learning_rate_steps=self.current_steps,
                      learning_rate_total_steps=self.total_steps)
        if self._last_applied_learning_rate is not None:
            result["learning_rate_last_applied"] = self._last_applied_learning_rate
        return result

    def state_dict(self):
        return {"schema": _SCHEMA, "initial_learning_rate": self.initial_learning_rate,
                "final_fraction": self.final_fraction, "total_steps": self.total_steps,
                "current_steps": self.current_steps, "checkpoint_scope": "schedule_state_only"}

    def _read_state(self):
        if self.load_path is None:
            raise ValueError("schedule load_path is required")
        state = json.loads(self.load_path.read_text())
        if not isinstance(state, dict) or set(state) != _STATE_FIELDS:
            raise ValueError("invalid schedule checkpoint fields")
        if type(state["schema"]) is not int or state["schema"] != _SCHEMA:
            raise ValueError("unsupported schedule schema")
        options = _parameters(state["initial_learning_rate"], state["final_fraction"], state["total_steps"])
        if options != (self.initial_learning_rate, self.final_fraction, self.total_steps):
            raise ValueError("schedule checkpoint parameters do not match")
        steps = state["current_steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("schedule current_steps must be a nonnegative integer")
        if state["checkpoint_scope"] != "schedule_state_only":
            raise ValueError("unsupported schedule checkpoint scope")
        return steps

    def load(self):
        steps = self._read_state()  # Reject mismatches before touching native weights.
        self._agent.load()
        self.current_steps = steps
        self._last_applied_learning_rate = None

    def save(self):
        if self.save_path is None:
            raise ValueError("schedule save_path is required")
        encoded = json.dumps(self.state_dict(), allow_nan=False, indent=2) + "\n"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.save_path.parent,
                                             prefix=self.save_path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._agent.save()
            os.replace(temporary, self.save_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
