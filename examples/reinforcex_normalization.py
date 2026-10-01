"""Per-agent observation/return normalization for the Python FFI adapter.

This wrapper stores each observation after normalization at collection time;
the underlying agent owns that transition and no old observation is rewritten.
``act`` is evaluation only and never updates the normalization or episode state.
``stop_episode`` is a training operation, including for time-limit truncations.

The running moments use the same float64 population-moment update and 1e-4
prior as SB3 RunningMeanStd. Unlike VecNormalize's autoreset vector environment,
this agent API explicitly observes terminal states, so terminal observations
are included in its observation moments. No SB3 or Torch import is required.

One wrapper belongs to one agent/environment stream. It is not a mechanism for
sharing normalization statistics across parallel workers or shared RND models.
Do not apply the default wrapper to an agent exporting to shared replay: its
normalized transitions would mix with other agents' raw transitions. Shared
replay requires ``preserve_replay_inputs=True`` and a supporting raw-input FFI.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


_SCHEMA = 1
_INITIAL_COUNT = 1e-4
_EPSILON = 1e-8


class _RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = _INITIAL_COUNT

    def update(self, value):
        """Merge one sample with the existing population moments (Welford)."""
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.mean.shape or not np.isfinite(value).all():
            raise ValueError("running moment sample has wrong shape or nonfinite values")
        delta = value - self.mean
        count = self.count + 1.0
        mean = self.mean + delta / count
        variance = (self.var * self.count + np.square(delta) * self.count / count) / count
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("running moments overflowed")
        self.mean, self.var, self.count = mean, variance, count

    def state(self):
        return {"count": self.count, "mean": self.mean.tolist(), "var": self.var.tolist()}

    @classmethod
    def from_state(cls, state, shape):
        if not isinstance(state, dict) or set(state) != {"count", "mean", "var"}:
            raise ValueError("invalid running moment fields")
        count = state["count"]
        if isinstance(count, bool) or not isinstance(count, (float, int)) or not math.isfinite(count) or count < _INITIAL_COUNT:
            raise ValueError("invalid running moment count")
        mean, variance = np.asarray(state["mean"], dtype=np.float64), np.asarray(state["var"], dtype=np.float64)
        if mean.shape != shape or variance.shape != shape:
            raise ValueError("normalization moment shape mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError("invalid normalization mean or variance")
        result = cls(shape)
        result.mean, result.var, result.count = mean.copy(), variance.copy(), float(count)
        return result


class NormalizedAgent:
    """Wrap an already-created FFI Agent without changing its model or replay.

    ``save_path`` and ``load_path`` refer only to normalization JSON, independently
    of the native agent's weight paths. A constructor load restores moments only:
    the supplied agent is responsible for its own initial weight loading.

    Native checkpoints contain weights, not a live environment/rollout. Loading
    normalization therefore starts a new episode with zero discounted return and
    no pending action. Call explicit ``load`` only between episodes; it validates
    the JSON first, invokes native ``load``, and then replaces the moments.
    Saving the JSON is atomic, but the weights plus JSON are not one transaction.
    """

    def __init__(self, agent, observation_size: int, gamma: float, *,
                 normalize_observations=True, normalize_rewards=True,
                 clip_observations=10.0, clip_rewards=10.0,
                 preserve_replay_inputs=False,
                 save_path: Path | str | None = None,
                 load_path: Path | str | None = None):
        if isinstance(observation_size, bool) or not isinstance(observation_size, int) or observation_size <= 0:
            raise ValueError("observation_size must be a positive integer")
        if isinstance(gamma, bool) or not isinstance(gamma, (float, int)) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and within [0, 1]")
        if any(type(value) is not bool for value in
               (normalize_observations, normalize_rewards, preserve_replay_inputs)):
            raise ValueError("normalization enable options must be booleans")
        if preserve_replay_inputs:
            methods = ("act_and_train_with_replay_input", "stop_episode_with_replay_input")
            symbols = ("rx_agent_act_and_train_with_replay_input", "rx_agent_stop_episode_with_replay_input")
            if not all(callable(getattr(agent, name, None)) for name in methods):
                raise RuntimeError("agent does not support preserving raw replay inputs")
            library = getattr(agent, "lib", None)
            if library is not None and not all(hasattr(library, name) for name in symbols):
                raise RuntimeError("FFI library does not support preserving raw replay inputs; rebuild ReinforceX")
        for name, value in (("clip_observations", clip_observations), ("clip_rewards", clip_rewards)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self._agent = agent
        self.observation_size = observation_size
        self.gamma = float(gamma)
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.preserve_replay_inputs = preserve_replay_inputs
        self.clip_observations = float(clip_observations)
        self.clip_rewards = float(clip_rewards)
        self.save_path = Path(save_path) if save_path is not None else None
        self.load_path = Path(load_path) if load_path is not None else None
        self._obs_rms = _RunningMeanStd((observation_size,))
        self._return_rms = _RunningMeanStd(())
        self._discounted_return = 0.0
        self._pending_action = False
        if self.load_path is not None:
            self._restore(self._read_state())

    def __getattr__(self, name):
        # Guard partially constructed instances instead of recursively resolving
        # _agent through __getattr__. lib/handle/action_bounds/close delegate here.
        return getattr(object.__getattribute__(self, "_agent"), name)

    def _options(self):
        return {"normalize_observations": self.normalize_observations,
                "normalize_rewards": self.normalize_rewards,
                "clip_observations": self.clip_observations,
                "clip_rewards": self.clip_rewards,
                "preserve_replay_inputs": self.preserve_replay_inputs,
                "epsilon": _EPSILON, "initial_count": _INITIAL_COUNT}

    def state_dict(self):
        """Return independent JSON-compatible persistent moments, with no live rollout."""
        return {"schema": _SCHEMA, "observation_shape": [self.observation_size],
                "gamma": self.gamma, "options": self._options(),
                "observation": self._obs_rms.state(), "discounted_return": self._return_rms.state(),
                "load_episode_policy": "reset_return_and_pending_action"}

    def _observation(self, observation):
        value = np.asarray(observation, dtype=np.float64).reshape(-1)
        if value.shape != (self.observation_size,) or not np.isfinite(value).all():
            raise ValueError("observation has wrong size or nonfinite values")
        return value

    @staticmethod
    def _reward(reward):
        if isinstance(reward, (bool, np.bool_)):
            raise ValueError("reward must be a finite scalar")
        value = np.asarray(reward)
        if value.ndim != 0:
            raise ValueError("reward must be a finite scalar")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("reward must be a finite scalar")
        return result

    def _normalized_observation(self, observation):
        if self.normalize_observations:
            observation = np.clip((observation - self._obs_rms.mean) /
                                  np.sqrt(self._obs_rms.var + _EPSILON),
                                  -self.clip_observations, self.clip_observations)
        result = np.array(observation, dtype=np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("observation cannot be represented as finite float32")
        return result

    def _training_reward(self, reward):
        # act_and_train's first reward belongs to no action. It is a dummy and
        # must not enter the return moments. stop_episode completes the last
        # action, so its actual reward is processed by this same path once.
        if not self._pending_action:
            return 0.0
        if not self.normalize_rewards:
            return reward
        discounted_return = self.gamma * self._discounted_return + reward
        self._return_rms.update(discounted_return)
        self._discounted_return = discounted_return
        return float(np.clip(reward / np.sqrt(self._return_rms.var + _EPSILON),
                             -self.clip_rewards, self.clip_rewards))

    def act_and_train(self, observation, reward):
        observation, reward = self._observation(observation), self._reward(reward)
        if self.normalize_observations:
            self._obs_rms.update(observation)
        normalized_reward = self._training_reward(reward)
        normalized_observation = self._normalized_observation(observation)
        if self.preserve_replay_inputs:
            # The caller may already have applied environment reward shaping;
            # "raw" here means before this normalization wrapper, not before it.
            result = self._agent.act_and_train_with_replay_input(
                normalized_observation, normalized_reward,
                np.array(observation, dtype=np.float32, copy=True), reward)
        else:
            result = self._agent.act_and_train(normalized_observation, normalized_reward)
        self._pending_action = True
        return result

    def act(self, observation):
        # No RMS update, return accumulation, pending flag change, or episode end.
        return self._agent.act(self._normalized_observation(self._observation(observation)))

    def stop_episode(self, observation, reward, *, terminated=True):
        observation, reward = self._observation(observation), self._reward(reward)
        if type(terminated) is not bool:
            raise ValueError("terminated must be a boolean")
        if self.normalize_observations:
            self._obs_rms.update(observation)
        normalized_reward = self._training_reward(reward)
        normalized_observation = self._normalized_observation(observation)
        if self.preserve_replay_inputs:
            self._agent.stop_episode_with_replay_input(
                normalized_observation, normalized_reward,
                np.array(observation, dtype=np.float32, copy=True), reward, terminated=terminated)
        else:
            self._agent.stop_episode(normalized_observation, normalized_reward, terminated=terminated)
        self._discounted_return = 0.0
        self._pending_action = False

    def statistics(self):
        result = dict(self._agent.statistics())
        result.update(normalization_observation_count=self._obs_rms.count,
                      normalization_reward_count=self._return_rms.count)
        return result

    def _read_state(self):
        if self.load_path is None:
            raise ValueError("normalization load_path is required")
        state = json.loads(self.load_path.read_text())
        if not isinstance(state, dict) or type(state.get("schema")) is not int or state["schema"] != _SCHEMA:
            raise ValueError("unsupported normalization state schema")
        shape, gamma = state.get("observation_shape"), state.get("gamma")
        if (not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int
                or shape != [self.observation_size] or isinstance(gamma, bool)
                or not isinstance(gamma, (float, int)) or gamma != self.gamma):
            raise ValueError("normalization observation shape or gamma mismatch")
        options, expected = state.get("options"), self._options()
        if not isinstance(options, dict) or set(options) != set(expected) or any(
                type(options[name]) is not bool if isinstance(value, bool)
                else isinstance(options[name], bool) or not isinstance(options[name], (int, float))
                for name, value in expected.items()) or options != expected:
            raise ValueError("normalization options mismatch")
        if state.get("load_episode_policy") != "reset_return_and_pending_action":
            raise ValueError("normalization episode restoration policy mismatch")
        return (_RunningMeanStd.from_state(state.get("observation"), (self.observation_size,)),
                _RunningMeanStd.from_state(state.get("discounted_return"), ()))

    def _restore(self, moments):
        self._obs_rms, self._return_rms = moments
        self._discounted_return = 0.0
        self._pending_action = False

    def load(self):
        if self._pending_action:
            raise RuntimeError("load normalization only between training episodes")
        moments = self._read_state()  # Reject mismatch before touching native weights.
        self._agent.load()
        self._restore(moments)

    def save(self):
        if self.save_path is None:
            raise ValueError("normalization save_path is required")
        encoded = json.dumps(self.state_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
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
