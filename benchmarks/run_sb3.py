#!/usr/bin/env python3
"""CPU-only, single-environment SB3 reference runs with complete reward logs.

The training budget is a lower bound, as documented by SB3: PPO finishes its
current rollout and off-policy algorithms finish their collection interval.
No artificial terminal transition is inserted at the requested budget.
"""
from __future__ import annotations

import argparse
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


ALGORITHMS = {"dqn": DQN, "ppo": PPO, "sac": SAC}


def jsonable(value: Any) -> Any:
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

    def __init__(self, env: gym.Env, args: argparse.Namespace, sink: Jsonl):
        super().__init__(env)
        self.args = args
        self.sink = sink
        self.steps = 0
        self.episodes = 0
        self.episode_steps = 0
        self.raw_return = 0.0
        self.training_return = 0.0
        self.recent_returns: list[float] = []
        self.start = time.monotonic()

    def reset(self, **kwargs: Any):
        self.episode_steps = 0
        self.raw_return = 0.0
        self.training_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(reward)
        learning_reward = transform_reward(
            raw_reward, bool(terminated), self.args.reward_transform, self.args.reward_scale,
            self.episode_steps + 1, self.spec.max_episode_steps if self.spec else None,
        )
        if not math.isfinite(raw_reward) or not math.isfinite(learning_reward):
            raise FloatingPointError("environment returned a non-finite reward")
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
            "complete": False,
            "note": "Budget stopped collection without forcing termination or reset.",
        }


def make_env(args: argparse.Namespace) -> gym.Env:
    options = {}
    if args.max_episode_steps is not None:
        options["max_episode_steps"] = args.max_episode_steps
    return gym.make(args.env, **options)


class Evaluator:
    def __init__(self, args: argparse.Namespace, episodes: Jsonl, points: Jsonl):
        self.args = args
        self.env = make_env(args)
        self.episodes = episodes
        self.points = points
        self.seconds = 0.0
        self.point_index = 0

    def run(self, model: Any, phase: str, count: int, requested_step: int) -> dict[str, Any]:
        start = time.monotonic()
        base_seed = self.args.final_eval_seed if phase == "final" else self.args.validation_seed
        returns, lengths = [], []
        # Keep evaluation from changing the training RNG streams or policy mode.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        was_training = model.policy.training
        self.point_index += 1
        try:
            for index in range(count):
                episode_seed = base_seed + index
                observation, _ = self.env.reset(seed=episode_seed)
                raw_return, length = 0.0, 0
                while True:
                    action, _ = model.predict(observation, deterministic=True)
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
        std = float(np.std(returns, ddof=1)) if count > 1 else 0.0
        mean = float(np.mean(returns))
        margin = 1.96 * std / math.sqrt(count)
        result = {
            "point": self.point_index,
            "phase": phase,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="JSON exported by native_configs.as_dict(config(...)).")
    parser.add_argument("--env")
    parser.add_argument("--algo", choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Held-out final evaluation episodes (default: 100).")
    parser.add_argument("--progress-eval-episodes", type=int, default=10,
                        help="Episodes at initialization and each progress checkpoint.")
    parser.add_argument("--eval-checkpoints", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=800000)
    parser.add_argument("--final-eval-seed", type=int, default=900000)
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
    args = parser.parse_args()
    args.native_config = None
    if args.config is not None:
        args.native_config = json.loads(args.config.read_text())
        specs = args.native_config["agents"]
        if len(specs) != 1:
            parser.error("the SB3 reference accepts exactly one native agent")
        spec = specs[0]
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
        "PyTorch reference 2.14.0 and native LibTorch 2.7.0 differ; this is an implementation comparison, not a bitwise replica.",
        "Native hidden_layers counts additional layers after input->hidden; reference depth is hidden_layers+1.",
        "Network initialization and optimizer numerical details retain each library's implementation.",
        "Native and SB3 transition/optimizer timing may differ by an episode boundary or warmup step.",
    ]
    if args.algo == "ppo":
        result.update(
            learning_rate=cfg["learning_rate"], gae_lambda=cfg["gae_lambda"],
            n_steps=cfg["update_interval"], n_epochs=cfg["epochs"],
            batch_size=cfg["minibatch_size"], clip_range=cfg["policy_clip_epsilon"],
            clip_range_vf=cfg["value_clip_range"] if cfg["value_clip_range"] > 0 else None,
            vf_coef=cfg["value_loss_coefficient"], ent_coef=cfg["entropy_coefficient"],
            normalize_advantage=bool(cfg["standardize_gae"]),
        )
        notes += [
            "Native PPO shares its hidden trunk between actor and value; SB3 uses separate actor/value MLPs of the matched width and depth, so total parameter counts differ.",
            "SB3 PPO normalizes advantages per minibatch; native standardizes the rollout.",
        ]
        if cfg["action_space"] == 1:
            notes += [
                "Native continuous PPO uses state-dependent variance with min_variance and bounded mean; SB3 uses a state-independent log_std and clips executed actions. No policy replacement was made.",
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
            "SAC reference tau = 1-(1-native_tau)^(native_train_interval/native_target_interval). This aligns nominal retention per environment step, not target update ordering.",
        ]
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
    kwargs.update(args.hyperparams)
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
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    return kwargs


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    kwargs = model_kwargs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned_files = [
        "metadata.json", "config.json", "final.json", "train_episodes.jsonl",
        "eval_episodes.jsonl", "evaluations.jsonl", "final_model.zip",
    ]
    if any((output / name).exists() for name in owned_files):
        raise FileExistsError(f"Refusing to overwrite benchmark output: {output}")
    packages = {
        package: importlib.metadata.version(package)
        for package in ["stable-baselines3", "torch", "gymnasium", "numpy", "mujoco", "Box2D"]
    }
    metadata = {
        "status": "running",
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
        "evaluation_seed_protocol": "Fixed validation seeds for initial/progress, disjoint final seeds.",
        "budget_protocol": "SB3 lower-bound step budget; finish rollout/collection interval without forced terminal.",
        "reward_metric": "Unmodified environment episode return, excluding incomplete episodes.",
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
        recorder = TrainingRecorder(make_env(args), args, train_sink)
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
        evaluator = Evaluator(args, eval_sink, point_sink)
        evaluator.run(model, "initial", args.progress_eval_episodes, 0)
        callback = EvaluationCallback(args, evaluator)
        learn_started = time.monotonic()
        previous_eval_seconds = evaluator.seconds
        model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
        learn_elapsed = time.monotonic() - learn_started
        training_seconds = learn_elapsed - (evaluator.seconds - previous_eval_seconds)
        checkpoint = output / "final_model.zip"
        model.save(checkpoint)
        if args.save_replay_buffer and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(output / "final_replay_buffer.pkl")
        final_evaluation = evaluator.run(model, "final", args.eval_episodes, args.steps)
        final = {
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


if __name__ == "__main__":
    main()
