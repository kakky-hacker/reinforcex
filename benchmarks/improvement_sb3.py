#!/usr/bin/env python3
"""CPU SB3 reference for the October improvement study, one native agent per run.

The training budget is a lower bound, as documented by SB3: PPO finishes its
current rollout and off-policy algorithms finish their collection interval.
No artificial terminal transition is inserted at the requested budget.

Development uses seeds 1100000+ for all evaluation. Confirmation uses those
validation seeds and reserves 1200000+ for its final evaluation. Neither stage
uses the old 800000/900000 evaluation seeds. Run in the dedicated SB3 Python
process after unsetting DYLD_LIBRARY_PATH and LD_LIBRARY_PATH. --describe only
validates and prints the constructor mapping; --self-test runs mapping tests.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback
from typing import Any

# Do not silently import a Python torch installation against native LibTorch.
# Clearing these after process startup is insufficient on all supported loaders.
if any(os.environ.get(name) for name in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")):
    raise RuntimeError("Unset DYLD_LIBRARY_PATH and LD_LIBRARY_PATH before launching this SB3 process")

# Set limits before importing numerical libraries. CPU selection is also explicit
# in the SB3 constructor; an available MPS device is never selected automatically.
for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
# This module imports NumPy only: no native agent, ctypes library, or LibTorch.
from reinforcex_normalization import NormalizedAgent


ALGORITHMS = {"dqn": DQN, "ppo": PPO, "sac": SAC}
VALIDATION_SEED = 1_100_000
CONFIRMATION_SEED = 1_200_000
EVALUATION_SEED_BLOCK = 100_000


class _NormalizationSink:
    """Python-only sink adapting the agent normalizer to Gym observations/rewards."""

    def act_and_train(self, observation, reward):
        return observation, reward

    def stop_episode(self, observation, reward, *, terminated=True):
        self.terminal = observation, reward

    def act(self, observation):
        return observation

    def statistics(self):
        return {}

    def save(self):
        pass

    def load(self):
        pass


class ReferenceNormalizer:
    """Use the exact native wrapper's moments/update order without a native DLL.

    The first reset is a dummy reward, each step is one real reward, and terminal
    observations enter the moments before resetting the return. Evaluation calls
    only the frozen observation transform. An SB3 autoreset's initial observation
    is consumed immediately; native normally consumes it at its next action call.
    """

    def __init__(self, config, options, load_path=None):
        if not isinstance(options, dict):
            raise ValueError("spec.normalization must be an options object")
        allowed = {"normalize_observations", "normalize_rewards", "clip_observations",
                   "clip_rewards", "preserve_replay_inputs"}
        if set(options) - allowed:
            raise ValueError(f"unsupported normalization options: {sorted(set(options) - allowed)}")
        self.native_options = dict(options)
        effective = dict(options)
        preserve = effective.pop("preserve_replay_inputs", False)
        if type(preserve) is not bool:
            raise ValueError("preserve_replay_inputs must be a boolean")
        self.sink = _NormalizationSink()
        self.agent = NormalizedAgent(self.sink, config["agent"]["obs_size"],
                                     config["agent"]["gamma"], load_path=load_path,
                                     preserve_replay_inputs=False, **effective)

    def reset(self, observation):
        if self.agent._pending_action:
            raise RuntimeError("training environment reset before its previous episode ended")
        normalized_observation, _dummy_reward = self.agent.act_and_train(observation, 0.0)
        return normalized_observation

    def step(self, observation, reward, terminated, truncated):
        if terminated or truncated:
            self.agent.stop_episode(observation, reward, terminated=bool(terminated))
            return self.sink.terminal
        return self.agent.act_and_train(observation, reward)

    def evaluate(self, observation):
        return self.agent.act(observation)

    def state_dict(self):
        return self.agent.state_dict()

    def training_state(self):
        # Include transient return/pending state as well as persistent moments.
        return {"moments": self.state_dict(), "return": self.agent._discounted_return,
                "pending": self.agent._pending_action}

    def save(self, path):
        self.agent.save_path = Path(path)
        self.agent.save()


def normalization_for_config(native_config, load_path=None):
    spec = native_config["agents"][0]
    if spec.get("normalization") is None:
        if load_path is not None:
            raise ValueError("normalization checkpoint supplied for an unnormalized configuration")
        return None
    return ReferenceNormalizer(spec["config"], spec["normalization"], load_path)


def save_reference_checkpoint(model, output, args, normalizer):
    """Save a model/moments pair and publish its hash manifest only after both exist."""
    checkpoint = output / "final_model.zip"
    model.save(checkpoint)
    normalization_path = None
    if normalizer is not None:
        normalization_path = output / "final_normalization.json"
        normalizer.save(normalization_path)
    artifacts = {"model": {"name": checkpoint.name,
                           "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    if normalization_path is not None:
        artifacts["normalization"] = {"name": normalization_path.name,
                                      "sha256": hashlib.sha256(normalization_path.read_bytes()).hexdigest()}
    bundle = {"schema": 1, "algorithm": args.algo, "environment": args.env,
              "native_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "normalization_enabled": normalizer is not None, "artifacts": artifacts,
              "normalization_source_sha256": hashlib.sha256(
                  (ROOT / "examples/reinforcex_normalization.py").read_bytes()).hexdigest()}
    write_json(output / "final_checkpoint.json", bundle)
    return checkpoint, normalization_path


def load_reference_checkpoint(directory, args):
    """Verify configuration and the complete saved pair before loading CPU weights.

    Returns ``(SB3 model, ReferenceNormalizer | None)``. Apply
    ``normalizer.evaluate(raw_observation)`` before deterministic prediction and
    keep evaluation rewards raw. This helper does not restore a live environment.
    """
    directory = Path(directory)
    bundle = json.loads((directory / "final_checkpoint.json").read_text())
    enabled = args.native_config["agents"][0].get("normalization") is not None
    expected = {"schema": 1, "algorithm": args.algo, "environment": args.env,
                "native_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "normalization_enabled": enabled,
                "normalization_source_sha256": hashlib.sha256(
                    (ROOT / "examples/reinforcex_normalization.py").read_bytes()).hexdigest()}
    if any(bundle.get(name) != value for name, value in expected.items()):
        raise ValueError("checkpoint configuration or normalization implementation mismatch")
    names = {"model": "final_model.zip"}
    if enabled:
        names["normalization"] = "final_normalization.json"
    artifacts = bundle.get("artifacts", {})
    if set(artifacts) != set(names):
        raise ValueError("checkpoint model/normalization pair is incomplete")
    for kind, filename in names.items():
        record = artifacts[kind]
        if record.get("name") != filename or hashlib.sha256((directory / filename).read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError(f"checkpoint {kind} path or hash mismatch")
    normalizer = normalization_for_config(args.native_config, directory / names["normalization"] if enabled else None)
    model = ALGORITHMS[args.algo].load(directory / names["model"], device="cpu")
    return model, normalizer


class LinearProgressLearningRate:
    """Serializable counterpart of the native wrapper's per-environment-step ramp."""

    def __init__(self, initial, final_fraction):
        if (type(initial) not in (int, float) or not math.isfinite(initial) or initial <= 0
                or type(final_fraction) not in (int, float) or not math.isfinite(final_fraction)
                or not 0 < final_fraction <= 1):
            raise ValueError("invalid linear learning rate schedule")
        self.initial, self.final_fraction = float(initial), float(final_fraction)

    def __call__(self, progress_remaining):
        remaining = max(0.0, min(1.0, float(progress_remaining)))
        return self.initial * (self.final_fraction + (1.0 - self.final_fraction) * remaining)


def jsonable(value: Any) -> Any:
    if isinstance(value, LinearProgressLearningRate):
        return {"kind": "linear", "initial": value.initial, "final_fraction": value.final_fraction}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class Jsonl:
    def __init__(self, path: Path):
        self.stream = path.open("x", buffering=1)

    def write(self, value: Any) -> None:
        self.stream.write(json.dumps(jsonable(value), allow_nan=False) + "\n")

    def close(self) -> None:
        self.stream.close()


def transform_reward(reward: float, terminated: bool, kind: str, scale: float,
                     step: int = 0, limit: int | None = None) -> float:
    if kind == "cartpole":
        return -1.0 if terminated and (limit is None or step < limit) else 0.01 * reward
    if kind == "scale":
        return scale * reward
    if kind == "hopper":
        return reward - 1.0
    if kind == "ant":
        return scale * (reward - 1.0)
    return reward


class TrainingRecorder(gym.Wrapper):
    """Record raw returns while returning the explicitly chosen learning reward."""

    def __init__(self, env: gym.Env, args: argparse.Namespace, sink: Jsonl,
                 normalizer: ReferenceNormalizer | None = None):
        super().__init__(env)
        self.args = args
        self.sink = sink
        self.normalizer = normalizer
        self.steps = 0
        self.episodes = 0
        self.episode_steps = 0
        self.raw_return = 0.0
        self.training_return = 0.0
        self.pre_normalization_return = 0.0
        self.recent_returns: list[float] = []
        self.start = time.monotonic()

    def reset(self, **kwargs: Any):
        self.episode_steps = 0
        self.raw_return = 0.0
        self.training_return = 0.0
        self.pre_normalization_return = 0.0
        observation, info = self.env.reset(**kwargs)
        if self.normalizer is not None:
            observation = self.normalizer.reset(observation)
        return observation, info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(reward)
        learning_reward = transform_reward(
            raw_reward, bool(terminated), self.args.reward_transform, self.args.reward_scale,
            self.episode_steps + 1, self.spec.max_episode_steps if self.spec else None,
        )
        if not math.isfinite(raw_reward) or not math.isfinite(learning_reward):
            raise FloatingPointError("environment returned a non-finite reward")
        self.pre_normalization_return += learning_reward
        if self.normalizer is not None:
            observation, learning_reward = self.normalizer.step(
                observation, learning_reward, terminated, truncated)
        self.steps += 1
        self.episode_steps += 1
        self.raw_return += raw_reward
        self.training_return += learning_reward
        if terminated or truncated:
            self.episodes += 1
            self.recent_returns.append(self.raw_return)
            self.recent_returns = self.recent_returns[-100:]
            elapsed = time.monotonic() - self.start
            self.sink.write({
                "episode": self.episodes,
                "env_steps": self.steps,
                "length": self.episode_steps,
                "return": self.raw_return,
                "train_return": self.training_return,
                "pre_normalization_return": self.pre_normalization_return,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "elapsed_seconds": elapsed,
            })
            # Passing an already-vectorized env avoids SB3's automatic Monitor,
            # which would otherwise replace this raw-return episode information.
            info = dict(info)
            info["episode"] = {"r": self.raw_return, "l": self.episode_steps, "t": elapsed}
        return observation, learning_reward, terminated, truncated, info

    def partial_episode(self) -> dict[str, Any] | None:
        if self.episode_steps == 0:
            return None
        return {
            "episode": self.episodes + 1,
            "length": self.episode_steps,
            "return": self.raw_return,
            "train_return": self.training_return,
            "pre_normalization_return": self.pre_normalization_return,
            "complete": False,
            "note": "Budget stopped collection without forcing termination or reset.",
        }


def make_env(args: argparse.Namespace) -> gym.Env:
    options = {}
    if args.max_episode_steps is not None:
        options["max_episode_steps"] = args.max_episode_steps
    return gym.make(args.env, **options)


class Evaluator:
    def __init__(self, args: argparse.Namespace, episodes: Jsonl, points: Jsonl,
                 normalizer: ReferenceNormalizer | None = None):
        self.args = args
        self.env = make_env(args)
        self.episodes = episodes
        self.points = points
        self.seconds = 0.0
        self.point_index = 0
        self.normalizer = normalizer

    def run(self, model: Any, phase: str, count: int, requested_step: int) -> dict[str, Any]:
        start = time.monotonic()
        base_seed = self.args.final_eval_seed if phase == "final" else self.args.validation_seed
        split = ("test" if self.args.stage == "confirmation" else "development") if phase == "final" else "validation"
        returns, lengths = [], []
        # Keep evaluation from changing the training RNG streams or policy mode.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        was_training = model.policy.training
        normalization_before = self.normalizer.training_state() if self.normalizer is not None else None
        self.point_index += 1
        try:
            for index in range(count):
                episode_seed = base_seed + index
                observation, _ = self.env.reset(seed=episode_seed)
                raw_return, length = 0.0, 0
                while True:
                    model_observation = self.normalizer.evaluate(observation) if self.normalizer is not None else observation
                    action, _ = model.predict(model_observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = self.env.step(action)
                    reward = float(reward)
                    if not math.isfinite(reward):
                        raise FloatingPointError("non-finite evaluation reward")
                    raw_return += reward
                    length += 1
                    if terminated or truncated:
                        break
                returns.append(raw_return)
                lengths.append(length)
                self.episodes.write({
                    "point": self.point_index,
                    "phase": phase,
                    "split": split,
                    "study_stage": self.args.stage,
                    "env_steps": int(model.num_timesteps),
                    "requested_step": requested_step,
                    "episode": index + 1,
                    "seed": episode_seed,
                    "return": raw_return,
                    "length": length,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            model.policy.set_training_mode(was_training)
            self.seconds += time.monotonic() - start
            if self.normalizer is not None and self.normalizer.training_state() != normalization_before:
                raise AssertionError("evaluation changed normalization moments or training episode state")
        std = float(np.std(returns, ddof=1)) if count > 1 else 0.0
        mean = float(np.mean(returns))
        margin = 1.96 * std / math.sqrt(count)
        result = {
            "point": self.point_index,
            "phase": phase,
            "split": split,
            "study_stage": self.args.stage,
            "env_steps": int(model.num_timesteps),
            "requested_step": requested_step,
            "updates": int(model._n_updates),
            "episodes": count,
            "seed_start": base_seed,
            "mean_return": mean,
            "std_return": std,
            "std_convention": "sample ddof=1",
            "min_return": min(returns),
            "max_return": max(returns),
            "mean_length": float(np.mean(lengths)),
            "normal_approx_ci95": [mean - margin, mean + margin],
            "ci_unit": "evaluation episodes of this trained model, not training seeds",
            "evaluation_seconds": time.monotonic() - start,
            "deterministic": True,
            "normalization_state_unchanged": True if self.normalizer is not None else None,
        }
        if phase == "final":
            result["success_threshold"] = self.args.success_threshold
            result["passed"] = (
                mean >= self.args.success_threshold
                if self.args.success_threshold is not None else None
            )
        self.points.write(result)
        print(json.dumps({"phase": phase, "env_steps": model.num_timesteps,
                          "mean_return": mean, "episodes": count}), flush=True)
        return result

    def close(self) -> None:
        self.env.close()


class EvaluationCallback(BaseCallback):
    def __init__(self, args: argparse.Namespace, evaluator: Evaluator):
        super().__init__()
        self.args = args
        self.evaluator = evaluator
        self.thresholds = sorted({
            math.ceil(args.steps * i / args.eval_checkpoints)
            for i in range(1, args.eval_checkpoints + 1)
        }) if args.eval_checkpoints else []
        self.next_threshold = 0

    def _on_step(self) -> bool:
        while (self.next_threshold < len(self.thresholds)
               and self.num_timesteps >= self.thresholds[self.next_threshold]):
            self.evaluator.run(
                self.model, "progress", self.args.progress_eval_episodes,
                self.thresholds[self.next_threshold],
            )
            self.next_threshold += 1
        return True


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Explicit single-agent native study configuration JSON; no example defaults are read.")
    parser.add_argument("--case", required=True, help="Study case label recorded in metadata.")
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--describe", action="store_true", help="Print resolved mapping without creating output, environment, or model.")
    parser.add_argument("--env")
    parser.add_argument("--algo", choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Final evaluation episodes (default: 100); held out only in confirmation.")
    parser.add_argument("--progress-eval-episodes", type=int, default=10,
                        help="Episodes at initialization and each progress checkpoint.")
    parser.add_argument("--eval-checkpoints", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--final-eval-seed", type=int,
                        help="Must match the study stage; defaults to 1100000 development / 1200000 confirmation.")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--reward-transform",
                        choices=["raw", "cartpole", "scale", "hopper", "ant"])
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--net-arch", type=int, nargs="+")
    parser.add_argument("--activation", choices=["relu", "tanh", "elu"])
    parser.add_argument("--initial-log-std", type=float)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--hyperparams", type=json.loads, default={},
                        help="JSON object of additional SB3 constructor arguments.")
    for name, kind in {
        "learning-rate": float, "gamma": float, "batch-size": int,
        "buffer-size": int, "learning-starts": int, "train-freq": int,
        "gradient-steps": int, "tau": float, "n-steps": int, "n-epochs": int,
        "gae-lambda": float, "clip-range": float, "vf-coef": float,
        "max-grad-norm": float, "target-kl": float, "target-update-interval": int,
        "exploration-fraction": float, "exploration-initial-eps": float,
        "exploration-final-eps": float,
    }.items():
        parser.add_argument("--" + name, type=kind)
    parser.add_argument("--ent-coef", help="PPO: number; SAC: number, auto, or auto_0.05.")
    parser.add_argument("--target-entropy", help="SAC: number or auto.")
    args = parser.parse_args(argv)
    expected_final_seed = CONFIRMATION_SEED if args.stage == "confirmation" else VALIDATION_SEED
    if args.final_eval_seed is None:
        args.final_eval_seed = expected_final_seed
    if args.validation_seed != VALIDATION_SEED or args.final_eval_seed != expected_final_seed:
        parser.error("evaluation seeds must match the reserved development/confirmation protocol")
    if VALIDATION_SEED <= args.seed < CONFIRMATION_SEED + EVALUATION_SEED_BLOCK:
        parser.error("training seed overlaps a reserved evaluation seed block")
    args.native_config = None
    if args.config is not None:
        args.native_config = json.loads(args.config.read_text())
        specs = args.native_config["agents"]
        if len(specs) != 1:
            parser.error("the SB3 reference accepts exactly one native agent")
        spec = specs[0]
        cfg = spec["config"]
        if "base" in cfg:
            raise ValueError("config must flatten the V2 base fields, as improvement_configs.as_dict does")
        if spec.get("rnd_config") is not None or spec.get("coefficient", 0):
            parser.error("SB3 reference does not implement RND; use the matched plain PPO case")
        env_id = args.native_config["env_id"]
        algorithm = spec["algorithm"]
        if args.env is not None and args.env != env_id:
            parser.error("--env disagrees with --config")
        if args.algo is not None and args.algo != algorithm:
            parser.error("--algo disagrees with --config")
        args.env, args.algo = env_id, algorithm
        if args.reward_transform is None:
            args.reward_transform = {
                "raw": "raw", "cartpole": "cartpole", "scale0.1": "scale",
                "hopper": "hopper", "ant_shared": "ant",
            }[args.native_config["reward_mode"]]
    if args.env is None or args.algo is None:
        parser.error("provide --config or both --env and --algo")
    if args.reward_transform is None:
        args.reward_transform = "raw"
    if args.steps <= 0 or args.eval_episodes <= 0 or args.progress_eval_episodes <= 0:
        parser.error("steps and evaluation episode counts must be positive")
    if min(args.eval_episodes, args.progress_eval_episodes) < 2:
        parser.error("study evaluations require at least two episodes")
    if max(args.eval_episodes, args.progress_eval_episodes) > EVALUATION_SEED_BLOCK:
        parser.error("evaluation count leaves its reserved seed block")
    if args.seed < 0 or args.validation_seed < 0 or args.final_eval_seed < 0:
        parser.error("seeds must be nonnegative")
    if args.eval_checkpoints < 0:
        parser.error("eval-checkpoints must be nonnegative")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("max-episode-steps must be positive")
    if not isinstance(args.hyperparams, dict):
        parser.error("hyperparams must be a JSON object")
    if args.net_arch is not None and any(width <= 0 for width in args.net_arch):
        parser.error("net-arch widths must be positive")
    if not math.isfinite(args.reward_scale):
        parser.error("reward-scale must be finite")
    # Validate even in --describe mode; never silently ignore a normalization spec.
    normalization_for_config(args.native_config)
    return args


def native_kwargs(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Map documented knobs, explicitly recording unavoidable implementation differences."""
    if args.native_config is None:
        return {}, []
    cfg = args.native_config["agents"][0]["config"]
    agent = cfg["agent"]
    # Every native FC constructor has an input->hidden layer followed by
    # hidden_layers *additional* hidden->hidden layers, passed through FFI unchanged.
    result = {
        "gamma": agent["gamma"],
        "policy_kwargs": {
            "net_arch": [agent["hidden_size"]] * (agent["hidden_layers"] + 1),
            "activation_fn": torch.nn.ReLU,
        },
    }
    notes = [
        "The reference Python torch version is recorded in packages; native LibTorch is 2.7.0. These implementations and RNG streams are not bitwise replicas.",
        "Native hidden_layers counts additional layers after input->hidden; reference depth is hidden_layers+1.",
        "Only one native agent configuration is accepted. This reference has one environment and no cross-agent replay or RND.",
        "Identical training seeds do not imply identical weights, exploration, replay samples, or minibatch permutations across implementations.",
        "Native and SB3 transition/optimizer timing may differ by an episode boundary or warmup step.",
        "SB3 progress callbacks run during collection before the pending rollout/gradient update. The last progress evaluation may therefore differ from the saved final policy at the same environment step.",
        "Final means use the last saved policy, never a best checkpoint. Evaluation episode SD/CI describe one model, not between-training-seed uncertainty.",
        "Development final evaluation reuses development seeds and is not held out. Only confirmation final evaluation uses the separate 1200000 seed block.",
    ]
    if args.native_config["agents"][0].get("normalization") is not None:
        normalization_for_config(args.native_config)
        notes += [
            "Observation/return normalization reuses the native Python NormalizedAgent through a Python-only sink, without loading its FFI. This matches float64 Welford moments, count prior 1e-4, variance epsilon 1e-8, enabled flags, clips and native gamma.",
            "Each real transition updates return moments once; the reset dummy reward is excluded. Terminal observations update observation moments, and both termination and truncation reset the discounted return. Evaluation freezes all moments and training episode state and sums raw environment rewards.",
            "SB3 DummyVecEnv normalizes the next reset observation immediately after a done step; native consumes it at the next action call. At a final done boundary SB3 may therefore include one unused reset observation, and intermediate callbacks can see a one-transition-later normalization snapshot than the native action/reward API.",
            "preserve_replay_inputs only controls native shared-replay export; it is recorded in the source config but ignored for SB3 preprocessing. SB3 PPO has no shared replay export. This reference does not reproduce hybrid-worker interactions.",
            "Episode return remains raw. train_return records rewards after shaping and normalization; pre_normalization_return records shaping only. Native learning_reward episode logs currently record shaping before normalization, so those two training columns are not interchangeable.",
            "final_model.zip and final_normalization.json are a pair verified by final_checkpoint.json hashes. Reload through load_reference_checkpoint; loading weights without the normalization state changes the policy input distribution.",
        ]
    if args.algo == "ppo":
        model = cfg.get("model", 0)
        activation = cfg.get("activation", 0)
        epsilon = cfg.get("adam_epsilon", 1e-8)
        target_kl = cfg.get("target_kl", 0.0)
        initial_log_std = cfg.get("initial_log_std", 0.0)
        if model not in (0, 1) or activation not in (0, 1):
            raise ValueError("unsupported native PPO model or activation enum")
        if (not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(target_kl)
                or target_kl < 0 or not math.isfinite(initial_log_std)):
            raise ValueError("invalid native PPO optimizer, KL, or initial log-std setting")
        result.update(
            learning_rate=cfg["learning_rate"], gae_lambda=cfg["gae_lambda"],
            n_steps=cfg["update_interval"], n_epochs=cfg["epochs"],
            batch_size=cfg["minibatch_size"], clip_range=cfg["policy_clip_epsilon"],
            clip_range_vf=cfg["value_clip_range"] if cfg["value_clip_range"] > 0 else None,
            vf_coef=cfg["value_loss_coefficient"], ent_coef=cfg["entropy_coefficient"],
            normalize_advantage=bool(cfg["standardize_gae"]),
            max_grad_norm=0.5, target_kl=target_kl if target_kl > 0 else None,
        )
        result["policy_kwargs"]["optimizer_kwargs"] = {"eps": epsilon}
        notes += [
            "When enabled, native PPO normalizes advantages over the rollout using population SD (ddof=0); SB3 normalizes each minibatch using sample SD (ddof=1). Returns remain unnormalized in both.",
            "With value clipping >0, native PPO takes max(unclipped squared error, clipped squared error); SB3 takes only the clipped-prediction squared error. Mapping clip_range_vf matches the range, not the loss. Zero disables value clipping in the new native core and maps to SB3 None.",
            "Native PPO clamps policy log ratios to [-8,8] before exponentiation; SB3 does not use that numerical clamp. Both KL guards use the unclipped reverse-KL approximation and a 1.5*target cutoff before the current optimizer step.",
            "PPO maps Adam epsilon and max gradient norm=0.5 explicitly. Optimizer numerical details, shuffling RNG and floating point reduction order remain implementation specific.",
            "Native PPO updates after completed transitions and can retain one boundary action sampled before the previous update; SB3 collects a fresh rollout after each update. SB3 completes a rollout beyond a non-divisible step budget; native can leave a final partial rollout unoptimized. Compare actual steps and optimizer counts.",
            "SB3 PPO _n_updates counts attempted epochs (including a KL-stopped epoch); native updates counts rollouts and optimizer_steps counts minibatch steps. These update fields are not interchangeable.",
        ]
        if model == 1:
            result["policy_kwargs"].update(
                activation_fn=torch.nn.Tanh if activation == 0 else torch.nn.ReLU,
                ortho_init=True,
            )
            notes += [
                "Native PPO V2 model=1 and SB3 use separate actor/value MLPs with matched width, depth, activation and orthogonal gains (hidden sqrt(2), actor .01, value 1). Same shape/gains do not imply identical random weights. Native checkpoint configuration is a nontrainable descriptor.",
            ]
            if cfg["action_space"] == 1:
                if initial_log_std < 0.5 * math.log(cfg["min_variance"]):
                    raise ValueError("native initial_log_std is below its variance floor")
                result["policy_kwargs"]["log_std_init"] = initial_log_std
                notes += [
                    "Both separate continuous PPO policies use unbounded linear means and one learned state-independent log_std per action, initialized from native initial_log_std. Native additionally floors log_std/variance at min_variance; SB3 has no equivalent floor.",
                    "Both evaluate Gaussian likelihoods on unclipped sampled actions and clip executed actions. Native policy bounds are mapped into the environment Box by the wrapper; SB3 clips directly to the Box. This mapping is identical only when native policy bounds equal environment bounds.",
                ]
        else:
            notes += [
                "Legacy native PPO model=0 uses a shared ReLU actor/value trunk; SB3 uses separate ReLU MLPs at matched width/depth. Parameter count and initialization differ; this is not an architecture-matched comparison.",
                "V2 activation and initial_log_std are ignored by native model=0; the adapter does not apply them. Adam epsilon and target KL remain active.",
            ]
            if cfg["action_space"] == 1:
                notes += [
                    "Legacy continuous PPO has state-dependent variance with a variance floor and bounded mean. SB3 retains its state-independent log_std and unbounded mean; these policy parameterizations differ.",
                ]
    elif args.algo == "dqn":
        result.update(
            learning_rate=cfg["learning_rate"], batch_size=cfg["batch_size"],
            buffer_size=cfg["replay_capacity"], learning_starts=cfg["batch_size"],
            n_steps=cfg["replay_n_steps"], train_freq=cfg["update_interval"],
            target_update_interval=cfg["target_update_interval"], gradient_steps=1,
            exploration_fraction=cfg["epsilon_decay_steps"] / args.steps,
            exploration_initial_eps=cfg["epsilon_start"],
            exploration_final_eps=cfg["epsilon_end"],
        )
        notes += [
            "Native implements Double-DQN targets; official SB3 DQN is vanilla DQN. Both use the requested n-step return setting, but replay storage/sampling differ.",
            "Native DQN begins once replay has a minibatch; SB3 learning_starts is set to batch_size and its condition is strictly greater than learning_starts.",
        ]
    elif args.algo == "sac":
        if cfg["action_space"] != 1:
            raise ValueError("Official SB3 SAC supports continuous actions only")
        if cfg["actor_learning_rate"] != cfg["critic_learning_rate"]:
            raise ValueError("Single SB3 learning_rate cannot match unequal native actor/critic learning rates")
        if not cfg["squash_action"]:
            raise ValueError("This SAC reference requires native squash_action=1")
        ratio = cfg["update_interval"] / cfg["target_update_interval"]
        if ratio < 1:
            raise ValueError("This reference adapter requires target interval <= training interval")
        # Collapse nominal per-env-step retention to one reference update. This
        # matches retention only: the timing/order of changing critics differs.
        effective_tau = 1.0 - (1.0 - cfg["tau"]) ** ratio
        result.update(
            learning_rate=cfg["actor_learning_rate"], batch_size=cfg["batch_size"],
            buffer_size=cfg["replay_capacity"], learning_starts=cfg["replay_start_size"],
            n_steps=cfg["replay_n_steps"], train_freq=cfg["update_interval"],
            gradient_steps=1, target_update_interval=1, tau=effective_tau,
            ent_coef=f"auto_{cfg['alpha']}", target_entropy="auto",
        )
        notes += [
            "Native continuous SAC samples its policy during replay warmup; SB3 uses uniform random actions before learning_starts.",
            "Native SAC variance uses its variance floor; SB3 uses clipped log_std. Actor parameterization and gradient clipping differ.",
            "Native SAC clips each actor/critic gradient norm at 10 and SB3 SAC does not clip these gradients. Native uses its FC initialization; SB3 uses PyTorch Linear initialization.",
            "Native SAC starts when its replay reaches replay_start_size; SB3 requires num_timesteps > learning_starts and finishes train_freq collection before training. This may offset the first update even at identical nominal settings.",
            "SAC reference tau = 1-(1-native_tau)^(native_train_interval/native_target_interval). This aligns nominal retention per environment step, not target update ordering.",
            "SB3's continuous entropy target is -action_dimension. Native discrete_target_entropy_ratio is ignored for continuous SAC in both this mapping and the native implementation.",
        ]
    schedule = args.native_config["agents"][0].get("learning_rate_schedule")
    if schedule is not None:
        if (args.algo not in {"dqn", "ppo"} or not isinstance(schedule, dict)
                or set(schedule) != {"kind", "final_fraction"} or schedule["kind"] != "linear"):
            raise ValueError("only a linear DQN/PPO learning rate schedule is supported")
        result["learning_rate"] = LinearProgressLearningRate(cfg["learning_rate"], schedule["final_fraction"])
        notes.append("Linear learning rate uses the same initial value, final fraction and per-learner budget. SB3 applies its progress fraction before each training batch; native sets the rate before training calls and final stop, so update-boundary timing can differ by one environment step.")
    return result, notes


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    signature = inspect.signature(ALGORITHMS[args.algo].__init__)
    reserved = {"self", "policy", "env", "device", "seed", "_init_setup_model"}
    if reserved.intersection(args.hyperparams):
        raise ValueError("hyperparams cannot override policy, env, device, seed, or model setup")
    unknown = set(args.hyperparams) - set(signature.parameters)
    if unknown:
        raise ValueError(f"Unsupported {args.algo} parameters: {sorted(unknown)}")
    kwargs, args.comparison_notes = native_kwargs(args)
    schedule = args.native_config["agents"][0].get("learning_rate_schedule")
    if schedule is not None and ("learning_rate" in args.hyperparams or args.learning_rate is not None):
        raise ValueError("a learning_rate override would replace the configured schedule")
    base_policy_kwargs = dict(kwargs.get("policy_kwargs", {}))
    kwargs.update(args.hyperparams)
    if "policy_kwargs" in args.hyperparams:
        if not isinstance(args.hyperparams["policy_kwargs"], dict):
            raise ValueError("policy_kwargs override must be an object")
        base_policy_kwargs.update(args.hyperparams["policy_kwargs"])
        kwargs["policy_kwargs"] = base_policy_kwargs
    candidate_names = {
        "learning_rate", "gamma", "batch_size", "buffer_size", "learning_starts",
        "train_freq", "gradient_steps", "tau", "n_steps", "n_epochs", "gae_lambda",
        "clip_range", "vf_coef", "max_grad_norm", "target_kl", "target_update_interval",
        "exploration_fraction", "exploration_initial_eps", "exploration_final_eps",
        "ent_coef", "target_entropy",
    }
    for name in candidate_names:
        value = getattr(args, name)
        if value is not None:
            if name not in signature.parameters:
                raise ValueError(f"{name} is not supported by {args.algo}")
            if name in {"ent_coef", "target_entropy"}:
                value = value if value.startswith("auto") else float(value)
            if name == "target_kl" and value == 0:
                value = None
            kwargs[name] = value
    policy_kwargs = dict(kwargs.get("policy_kwargs", {}))
    if args.net_arch is not None:
        policy_kwargs["net_arch"] = args.net_arch
    if args.activation is not None:
        policy_kwargs["activation_fn"] = {
            "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "elu": torch.nn.ELU,
        }[args.activation]
    if args.initial_log_std is not None:
        policy_kwargs["log_std_init"] = args.initial_log_std
    kwargs.update(device="cpu", seed=args.seed, verbose=args.verbose)
    if args.native_config["agents"][0].get("normalization") is not None and kwargs["gamma"] != args.native_config["agents"][0]["config"]["agent"]["gamma"]:
        raise ValueError("a gamma override would disagree with the normalization return discount")
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    if args.hyperparams or any(getattr(args, name) is not None for name in candidate_names) or any(
            getattr(args, name) is not None for name in ("activation", "net_arch", "initial_log_std")):
        args.comparison_notes.append(
            "Explicit CLI/hyperparams overrides were supplied; inspect arguments and resolved_constructor. These may deliberately differ from the native configuration mapping.")
    return kwargs


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    kwargs = model_kwargs(args)
    normalizer = normalization_for_config(args.native_config)
    if args.describe:
        print(json.dumps(jsonable({
            "case": args.case, "study_stage": args.stage, "native_config": args.native_config,
            "constructor": kwargs, "validation_seed": args.validation_seed,
            "final_eval_seed": args.final_eval_seed, "comparison_notes": args.comparison_notes,
            "requested_steps": args.steps, "training_seed": args.seed,
            "normalization": None if normalizer is None else normalizer.state_dict(),
        }), indent=2, allow_nan=False))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned_files = [
        "metadata.json", "config.json", "final.json", "train_episodes.jsonl",
        "eval_episodes.jsonl", "evaluations.jsonl", "final_model.zip",
        "final_normalization.json", "final_checkpoint.json",
    ]
    if any((output / name).exists() for name in owned_files):
        raise FileExistsError(f"Refusing to overwrite benchmark output: {output}")
    packages = {
        package: importlib.metadata.version(package)
        for package in ["stable-baselines3", "torch", "gymnasium", "numpy", "mujoco", "Box2D"]
    }
    metadata = {
        "status": "running",
        "case": args.case, "study_stage": args.stage,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "device": "cpu", "threads": 1, "n_envs": 1,
        "native_libtorch_comparison_version": "2.7.0",
        "comparison_notes": args.comparison_notes,
        "native_config_path": str(args.config.resolve()) if args.config else None,
        "native_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_seed_protocol": "Seed the first reset only; natural unseeded resets thereafter.",
        "validation_seed": args.validation_seed, "final_eval_seed": args.final_eval_seed,
        "evaluation_seed_protocol": "Initial/progress use 1100000+; development final reuses 1100000+, confirmation final uses disjoint 1200000+. Development results are not held-out evidence.",
        "budget_protocol": "SB3 lower-bound step budget; finish rollout/collection interval without forced terminal.",
        "reward_metric": "Unmodified environment episode return, excluding incomplete episodes.",
        "training_reward_metric": "train_return is after shaping/normalization; pre_normalization_return is shaping only.",
        "normalization_native_options": args.native_config["agents"][0].get("normalization"),
        "normalization_effective_options": None if normalizer is None else normalizer.state_dict()["options"],
        "normalization_source_sha256": hashlib.sha256((ROOT / "examples/reinforcex_normalization.py").read_bytes()).hexdigest(),
    }
    if args.native_config is not None and args.algo == "sac":
        cfg = args.native_config["agents"][0]["config"]
        metadata["sac_target_update_mapping"] = {
            "native_tau": cfg["tau"],
            "native_train_env_interval": cfg["update_interval"],
            "native_target_env_interval": cfg["target_update_interval"],
            "sb3_tau": kwargs["tau"],
            "sb3_target_gradient_interval": kwargs["target_update_interval"],
            "sb3_train_env_interval": kwargs["train_freq"],
            "sb3_gradient_steps": kwargs["gradient_steps"],
            "equivalence": "Nominal retention only; update ordering is not identical.",
        }
    write_json(output / "metadata.json", metadata)
    train_sink = Jsonl(output / "train_episodes.jsonl")
    eval_sink = Jsonl(output / "eval_episodes.jsonl")
    point_sink = Jsonl(output / "evaluations.jsonl")
    train_env = evaluator = None
    start = time.monotonic()
    try:
        recorder = TrainingRecorder(make_env(args), args, train_sink, normalizer)
        native = args.native_config["agents"][0]["config"]
        if int(np.prod(recorder.observation_space.shape)) != native["agent"]["obs_size"]:
            raise ValueError("native observation dimension does not match environment")
        if args.algo == "dqn" or native.get("action_space") == 0:
            if not isinstance(recorder.action_space, gym.spaces.Discrete) or recorder.action_space.n != native["agent"]["action_size"]:
                raise ValueError("native discrete action dimension does not match environment")
        elif not isinstance(recorder.action_space, gym.spaces.Box) or int(np.prod(recorder.action_space.shape)) != native["agent"]["action_size"]:
            raise ValueError("native continuous action dimension does not match environment")
        train_env = DummyVecEnv([lambda: recorder])
        model = ALGORITHMS[args.algo]("MlpPolicy", train_env, **kwargs)
        # The model's seed also seeds DummyVecEnv's next (first) reset. Automatic
        # episode resets inside DummyVecEnv subsequently pass no seed.
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(ALGORITHMS[args.algo].__init__).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        defaults.update(kwargs)
        write_json(output / "config.json", {
            "arguments": vars(args),
            "resolved_constructor": defaults,
            "policy": str(model.policy),
            "policy_parameter_count": sum(p.numel() for p in model.policy.parameters()),
            "environment_spec": {
                "id": recorder.spec.id,
                "max_episode_steps": recorder.spec.max_episode_steps,
                "reward_threshold": recorder.spec.reward_threshold,
                "observation_space": repr(recorder.observation_space),
                "action_space": repr(recorder.action_space),
            },
        })
        evaluator = Evaluator(args, eval_sink, point_sink, normalizer)
        evaluator.run(model, "initial", args.progress_eval_episodes, 0)
        callback = EvaluationCallback(args, evaluator)
        learn_started = time.monotonic()
        previous_eval_seconds = evaluator.seconds
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
        learn_elapsed = time.monotonic() - learn_started
        training_seconds = learn_elapsed - (evaluator.seconds - previous_eval_seconds)
        checkpoint, normalization_path = save_reference_checkpoint(model, output, args, normalizer)
        if args.save_replay_buffer and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(output / "final_replay_buffer.pkl")
        final_evaluation = evaluator.run(model, "final", args.eval_episodes, args.steps)
        final = {
            "status": "complete", "case": args.case, "study_stage": args.stage,
            "engine": "stable-baselines3", "algorithm": args.algo,
            "environment": args.env, "seed": args.seed,
            "requested_steps": args.steps,
            "actual_steps": int(model.num_timesteps),
            "completed_episodes": recorder.episodes,
            "updates": int(model._n_updates),
            "last100_training_mean_return": (
                float(np.mean(recorder.recent_returns)) if recorder.recent_returns else None
            ),
            "last100_training_episode_count": len(recorder.recent_returns),
            "partial_episode_excluded": recorder.partial_episode(),
            "training_seconds_excluding_evaluation": training_seconds,
            "evaluation_seconds": evaluator.seconds,
            "total_seconds": time.monotonic() - start,
            "steps_per_training_second": int(model.num_timesteps) / training_seconds,
            "checkpoint": str(checkpoint),
            "checkpoint_manifest": str(output / "final_checkpoint.json"),
            "normalization_checkpoint": str(normalization_path) if normalization_path is not None else None,
            "normalization_state": normalizer.state_dict() if normalizer is not None else None,
            "final_evaluation": final_evaluation,
            "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "max_rss_unit": "bytes",
        }
        if recorder.steps != model.num_timesteps:
            raise AssertionError("Recorder and SB3 step counts disagree")
        write_json(output / "final.json", final)
        metadata.update(status="complete", actual_steps=int(model.num_timesteps))
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        metadata["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(output / "metadata.json", metadata)
        if train_env is not None:
            train_env.close()
        if evaluator is not None:
            evaluator.close()
        train_sink.close()
        eval_sink.close()
        point_sink.close()


def self_test() -> None:
    """Small mapping/protocol tests. Never construct a model or run an environment."""
    import contextlib
    import io
    import tempfile
    import unittest

    class MappingTests(unittest.TestCase):
        def setUp(self):
            snapshot = Path(__file__).resolve().parents[1] / "reports/core_improvements_20261001/baseline/configurations.json"
            self.cases = json.loads(snapshot.read_text())

        def arguments(self, case, **changes):
            native = copy.deepcopy(self.cases[case])
            args = argparse.Namespace(native_config=native, algo=native["agents"][0]["algorithm"], steps=2_048_000)
            args.native_config["agents"][0]["config"].update(changes)
            return args

        def test_hopper_raw_mapping(self):
            args = self.arguments("hopper_sac")
            args.native_config["reward_mode"] = "raw"
            result, notes = native_kwargs(args)
            self.assertEqual(result["learning_rate"], 3e-4)
            self.assertEqual(result["policy_kwargs"]["net_arch"], [128, 128])
            self.assertIs(result["policy_kwargs"]["activation_fn"], torch.nn.ReLU)
            self.assertEqual((result["learning_starts"], result["batch_size"], result["buffer_size"]), (512, 128, 300000))
            self.assertEqual((result["n_steps"], result["train_freq"], result["gradient_steps"]), (1, 1, 1))
            self.assertAlmostEqual(result["tau"], .005)
            self.assertEqual((result["ent_coef"], result["target_entropy"]), ("auto_0.05", "auto"))
            self.assertEqual(transform_reward(2.5, False, "raw", .1), 2.5)
            self.assertEqual(transform_reward(2.5, False, "hopper", .1), 1.5)
            self.assertTrue(any("gradient norm at 10" in note for note in notes))

        def test_learning_rate_schedule_mapping_and_serialization(self):
            for name in ("cartpole_dqn", "cartpole_ppo"):
                args = self.arguments(name)
                args.native_config["agents"][0]["learning_rate_schedule"] = {"kind": "linear", "final_fraction": .05}
                result, notes = native_kwargs(args)
                schedule = result["learning_rate"]
                initial = args.native_config["agents"][0]["config"]["learning_rate"]
                self.assertAlmostEqual(schedule(1), initial)
                self.assertAlmostEqual(schedule(.5), initial * .525)
                self.assertAlmostEqual(schedule(0), initial * .05)
                self.assertEqual(schedule(-1), schedule(0))
                self.assertEqual(jsonable(schedule), {"kind": "linear", "initial": initial, "final_fraction": .05})
                self.assertTrue(any("update-boundary" in note for note in notes))
            args = self.arguments("hopper_sac")
            args.native_config["agents"][0]["learning_rate_schedule"] = {"kind": "linear", "final_fraction": .05}
            with self.assertRaises(ValueError):
                native_kwargs(args)

        def test_target_retention_mapping(self):
            result, _ = native_kwargs(self.arguments("hopper_sac", update_interval=4, target_update_interval=1))
            self.assertAlmostEqual(result["tau"], 1 - .995 ** 4)
            self.assertEqual(result["target_update_interval"], 1)

        def test_dqn_remains_double_vs_vanilla_reference(self):
            result, notes = native_kwargs(self.arguments("cartpole_dqn"))
            self.assertEqual(result["n_steps"], 3)
            self.assertEqual(result["target_update_interval"], 250)
            self.assertEqual(result["learning_starts"], 64)
            self.assertTrue(any("Double-DQN" in note for note in notes))

        def test_separate_ppo_tanh_options(self):
            args = self.arguments("halfcheetah_hybrid", model=1, activation=0,
                                  initial_log_std=-.5, adam_epsilon=2e-5, target_kl=.03)
            result, notes = native_kwargs(args)
            self.assertEqual(result["policy_kwargs"]["net_arch"], [256, 256])
            self.assertIs(result["policy_kwargs"]["activation_fn"], torch.nn.Tanh)
            self.assertTrue(result["policy_kwargs"]["ortho_init"])
            self.assertEqual(result["policy_kwargs"]["optimizer_kwargs"], {"eps": 2e-5})
            self.assertEqual(result["policy_kwargs"]["log_std_init"], -.5)
            self.assertEqual(result["target_kl"], .03)
            self.assertEqual(result["max_grad_norm"], .5)
            self.assertTrue(any("ddof=0" in note and "ddof=1" in note for note in notes))
            self.assertTrue(any("max(unclipped squared error" in note for note in notes))

        def test_separate_ppo_relu_and_discrete(self):
            result, _ = native_kwargs(self.arguments("cartpole_ppo", model=1, activation=1, adam_epsilon=1e-5))
            self.assertIs(result["policy_kwargs"]["activation_fn"], torch.nn.ReLU)
            self.assertNotIn("log_std_init", result["policy_kwargs"])

        def test_legacy_keeps_architecture_difference_explicit(self):
            result, notes = native_kwargs(self.arguments("cartpole_ppo", model=0, activation=0, initial_log_std=-1))
            self.assertIs(result["policy_kwargs"]["activation_fn"], torch.nn.ReLU)
            self.assertEqual(result["policy_kwargs"]["optimizer_kwargs"], {"eps": 1e-8})
            self.assertNotIn("log_std_init", result["policy_kwargs"])
            self.assertTrue(any("shared ReLU" in note for note in notes))

        def test_zero_disables_value_clip_and_kl(self):
            result, _ = native_kwargs(self.arguments("cartpole_ppo", model=1, value_clip_range=0, target_kl=0))
            self.assertIsNone(result["clip_range_vf"])
            self.assertIsNone(result["target_kl"])

        def test_invalid_ppo_options_rejected(self):
            for changes in ({"model": 2}, {"activation": 3}, {"adam_epsilon": 0},
                            {"adam_epsilon": math.nan}, {"initial_log_std": math.inf}, {"target_kl": -1}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    native_kwargs(self.arguments("cartpole_ppo", **changes))

        def test_unmatched_sac_settings_rejected(self):
            for changes in ({"action_space": 0}, {"squash_action": 0},
                            {"critic_learning_rate": .001}, {"target_update_interval": 2}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    native_kwargs(self.arguments("hopper_sac", **changes))

        def test_stages_seeds_and_single_agent_input(self):
            with tempfile.TemporaryDirectory(prefix="rx-reference-mapping-") as directory:
                config = Path(directory) / "input.json"
                document = copy.deepcopy(self.cases["hopper_sac"])
                document["reward_mode"] = "raw"
                config.write_text(json.dumps(document))
                argv = ["--config", str(config), "--case", "hopper_sac", "--seed", "1001",
                        "--steps", "2048000", "--output", str(Path(directory) / "unused")]
                development = parse_args(argv)
                confirmation = parse_args(argv + ["--stage", "confirmation"])
                self.assertEqual((development.validation_seed, development.final_eval_seed), (1100000, 1100000))
                self.assertEqual((confirmation.validation_seed, confirmation.final_eval_seed), (1100000, 1200000))
                self.assertEqual(development.reward_transform, "raw")
                self.assertEqual(model_kwargs(development)["device"], "cpu")
                for extra in (["--final-eval-seed", "900000"], ["--validation-seed", "800000"],
                              ["--seed", "1100001"], ["--eval-episodes", "100001"]):
                    with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_args(argv + extra)
                document["agents"].append(copy.deepcopy(document["agents"][0]))
                config.write_text(json.dumps(document))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(argv)
                self.assertFalse((Path(directory) / "unused").exists())

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MappingTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
