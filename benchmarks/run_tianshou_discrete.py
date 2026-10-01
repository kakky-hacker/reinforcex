#!/usr/bin/env python3
"""Supplementary official Tianshou Discrete SAC control; CPU, one environment.

The algorithm implementation is used unchanged. In particular Tianshou's MSE
critic loss and lack of SAC gradient clipping remain documented differences.
"""
from __future__ import annotations

import argparse
import dataclasses
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

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/reinforcex-tianshou-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/reinforcex-tianshou-numba")

import gymnasium as gym
import numpy as np
import torch
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSAC, DiscreteSACPolicy
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils.torch_utils import policy_within_training_step


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Jsonl:
    def __init__(self, path):
        self.stream = path.open("x", buffering=1)

    def write(self, value):
        self.stream.write(json.dumps(value, allow_nan=False) + "\n")

    def close(self):
        self.stream.close()


def learning_reward(reward, terminated, episode_step, limit):
    return -1.0 if terminated and episode_step < limit else .01 * float(reward)


def statistics(returns, lengths):
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(returns))
    return {"mean_return": mean, "std_return": std, "std_convention": "sample ddof=1",
            "min_return": min(returns), "max_return": max(returns),
            "mean_length": float(np.mean(lengths)),
            "normal_approx_ci95": [mean - margin, mean + margin],
            "ci_unit": "evaluation episodes of this trained model, not training seeds"}


class Evaluator:
    def __init__(self, args, episodes, points):
        self.args, self.episodes, self.points = args, episodes, points
        self.env = gym.make("CartPole-v1")
        self.seconds, self.point = 0.0, 0

    def run(self, policy, phase, steps, updates):
        count = self.args.eval_episodes if phase == "final" else self.args.progress_eval_episodes
        base_seed = 900000 if phase == "final" else 800000
        start = time.monotonic()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng, training = torch.random.get_rng_state(), policy.training
        within_training = policy.is_within_training_step
        policy.is_within_training_step = False
        policy.eval()
        returns, lengths = [], []
        self.point += 1
        try:
            with torch.no_grad():
                for episode in range(count):
                    obs, _ = self.env.reset(seed=base_seed + episode)
                    total, length = 0.0, 0
                    while True:
                        action = int(policy(Batch(obs=np.asarray([obs]), info={})).act.item())
                        obs, reward, terminated, truncated, _ = self.env.step(action)
                        total += float(reward)
                        length += 1
                        if terminated or truncated:
                            break
                    returns.append(total)
                    lengths.append(length)
                    self.episodes.write({"point": self.point, "phase": phase,
                                         "env_steps": steps, "requested_step": steps,
                                         "episode": episode + 1, "seed": base_seed + episode,
                                         "return": total, "length": length,
                                         "terminated": bool(terminated), "truncated": bool(truncated)})
        finally:
            policy.train(training)
            policy.is_within_training_step = within_training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(torch_rng)
        duration = time.monotonic() - start
        self.seconds += duration
        result = {"point": self.point, "phase": phase, "env_steps": steps,
                  "requested_step": steps, "updates": updates, "episodes": count,
                  "seed_start": base_seed, "deterministic": True,
                  "evaluation_seconds": duration, **statistics(returns, lengths)}
        if phase == "final":
            result.update(success_threshold=self.args.success_threshold,
                          passed=result["mean_return"] >= self.args.success_threshold)
        self.points.write(result)
        print(json.dumps({"phase": phase, "env_steps": steps,
                          "mean_return": result["mean_return"], "episodes": count}), flush=True)
        return result


def build_algorithm(config, env):
    c = config["agents"][0]["config"]
    a = c["agent"]
    hidden = [a["hidden_size"]] * (a["hidden_layers"] + 1)
    actor = DiscreteActor(preprocess_net=Net(state_shape=a["obs_size"], hidden_sizes=hidden),
                          action_shape=a["action_size"], softmax_output=False).to("cpu")
    critics = [DiscreteCritic(
        preprocess_net=Net(state_shape=a["obs_size"], hidden_sizes=hidden),
        last_size=a["action_size"]).to("cpu") for _ in range(2)]
    policy = DiscreteSACPolicy(actor=actor, action_space=env.action_space,
                               observation_space=env.observation_space, deterministic_eval=True)
    alpha = AutoAlpha(target_entropy=c["discrete_target_entropy_ratio"] * math.log(a["action_size"]),
                      log_alpha=math.log(c["alpha"]), optim=AdamOptimizerFactory(lr=3e-4))
    algorithm = DiscreteSAC(
        policy=policy, policy_optim=AdamOptimizerFactory(lr=c["actor_learning_rate"]),
        critic=critics[0], critic_optim=AdamOptimizerFactory(lr=c["critic_learning_rate"]),
        critic2=critics[1], critic2_optim=AdamOptimizerFactory(lr=c["critic_learning_rate"]),
        alpha=alpha, tau=c["tau"], gamma=a["gamma"], n_step_return_horizon=c["replay_n_steps"])
    resolved = {"net_arch": hidden, "activation": "ReLU", "actor_softmax_output": False,
                "critic_initialization": "two independently initialized networks",
                "gamma": a["gamma"], "actor_learning_rate": c["actor_learning_rate"],
                "critic_learning_rate": c["critic_learning_rate"], "alpha_learning_rate": 3e-4,
                "target_entropy": alpha.target_entropy if hasattr(alpha, "target_entropy") else
                                  c["discrete_target_entropy_ratio"] * math.log(a["action_size"]),
                "initial_alpha": c["alpha"], "tau": c["tau"],
                "n_step_return_horizon": c["replay_n_steps"], "batch_size": c["batch_size"],
                "replay_capacity": c["replay_capacity"], "learning_starts": c["replay_start_size"],
                "update_interval": 1, "target_update_interval": 1,
                "critic_loss": "MSE", "max_grad_norm": None,
                "actor_parameters": sum(p.numel() for p in actor.parameters()),
                "critic_parameters_each": [sum(p.numel() for p in q.parameters()) for q in critics]}
    return algorithm, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=204800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--progress-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=475.0)
    args = parser.parse_args()
    if min(args.steps, args.eval_episodes, args.progress_eval_episodes, args.checkpoints) <= 0:
        parser.error("steps, evaluation episodes and checkpoint count must be positive")
    config = json.loads(args.config.read_text())
    if (config["env_id"] != "CartPole-v1" or config["reward_mode"] != "cartpole"
            or len(config["agents"]) != 1 or config["agents"][0]["algorithm"] != "sac"
            or config["agents"][0].get("rnd_config") is not None):
        parser.error("this supplementary control accepts only single-agent CartPole discrete SAC")
    c = config["agents"][0]["config"]
    if c["action_space"] != 0 or c["update_interval"] != 1 or c["target_update_interval"] != 1:
        parser.error("only discrete actions and update/target intervals of one are supported")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(config["env_id"])
    env.action_space.seed(args.seed)
    algorithm, resolved = build_algorithm(config, env)
    buffer = ReplayBuffer(size=c["replay_capacity"])
    metadata = {
        "status": "running", "provider": "tianshou", "supplementary": True,
        "case": "cartpole_sac", "algorithm": "discrete_sac", "seed": args.seed,
        "requested_steps": args.steps, "device": "cpu", "torch_num_threads": 1,
        "torch_num_interop_threads": 1, "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("tianshou", "torch", "gymnasium", "numpy", "numba")},
        "native_libtorch_version": "2.7.0", "command": sys.argv,
        "runner_sha256": sha256(__file__), "config_sha256": sha256(args.config),
        "official_source_sha256": {str(Path(inspect.getfile(cls))): sha256(inspect.getfile(cls))
                                   for cls in (DiscreteSAC, AutoAlpha, ReplayBuffer, Net)},
        "evaluation": {"initial_episodes": args.progress_eval_episodes,
                       "progress_points": args.checkpoints,
                       "progress_episodes": args.progress_eval_episodes,
                       "validation_seed_start": 800000, "final_seed_start": 900000,
                       "final_episodes": args.eval_episodes, "deterministic": True},
        "comparison_caveats": [
            "Official Tianshou DiscreteSAC and native SAC share the method but are not identical implementations.",
            "Native discrete critics use Huber loss(delta=1); official Tianshou uses MSE.",
            "Native actor and critic gradient norm is clipped at 10; official Tianshou SAC has no clipping constructor argument and is used unchanged.",
            "Native exposes only completed n-step sequences in replay; Tianshou stores transitions immediately and shortens the return at the current buffer tail. Warmup512 therefore differs by a few transitions.",
            "Native chooses actions before its current update; this conventional reference collects one transition then updates. Native skips optimization at episode stop; this reference updates after all transitions including final ones.",
            "Replay sampling, random generator streams, initializers and target-update ordering are implementation differences.",
            "Intermediate evaluation occurs after this step's update. No intermediate score changes training or selects the final checkpoint.",
            "This supplementary control matches Gymnasium1.3 and native LibTorch2.7 using PyTorch2.7; SB3's separate controls use PyTorch2.14.",
        ],
    }
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "config.json", {
        "source_config": config, "resolved": resolved, "env_id": config["env_id"],
        "seed": args.seed, "requested_steps": args.steps,
        "reward_transform": "cartpole", "max_episode_steps": env.spec.max_episode_steps})
    train_sink = Jsonl(args.output / "train_episodes.jsonl")
    eval_sink = Jsonl(args.output / "eval_episodes.jsonl")
    point_sink = Jsonl(args.output / "evaluations.jsonl")
    update_sink = Jsonl(args.output / "update_diagnostics.jsonl")
    evaluator = Evaluator(args, eval_sink, point_sink)
    started = time.monotonic()
    steps, episodes, updates = 0, 0, 0
    length, raw_return, train_return = 0, 0.0, 0.0
    returns = []
    final_eval = None
    try:
        evaluator.run(algorithm.policy, "initial", 0, 0)
        obs, _ = env.reset(seed=args.seed)
        thresholds = {math.ceil(args.steps * i / args.checkpoints)
                      for i in range(1, args.checkpoints + 1)}
        with policy_within_training_step(algorithm.policy):
            for steps in range(1, args.steps + 1):
                with torch.no_grad():
                    if len(buffer) < c["replay_start_size"]:
                        action = int(env.action_space.sample())
                    else:
                        action = int(algorithm.policy(Batch(obs=np.asarray([obs]), info={})).act.item())
                next_obs, reward, terminated, truncated, _ = env.step(action)
                length += 1
                shaped = learning_reward(reward, terminated, length, env.spec.max_episode_steps)
                raw_return += float(reward)
                train_return += shaped
                buffer.add(Batch(obs=obs, act=action, rew=shaped, obs_next=next_obs,
                                 terminated=terminated, truncated=truncated, info={}))
                if len(buffer) >= max(c["replay_start_size"], c["batch_size"]):
                    stats = algorithm.update(buffer=buffer, sample_size=c["batch_size"])
                    updates += 1
                    losses = dataclasses.asdict(stats)
                    # Check every update without generating hundreds of thousands of log rows.
                    for key, value in losses.items():
                        if isinstance(value, (int, float)) and not math.isfinite(value):
                            raise FloatingPointError(f"non-finite update metric: {key}")
                    if updates == 1 or steps in thresholds:
                        update_sink.write({"env_steps": steps, "updates": updates,
                                           "alpha": algorithm.alpha.value, "metrics": losses})
                if terminated or truncated:
                    episodes += 1
                    returns.append(raw_return)
                    train_sink.write({"episode": episodes, "env_steps": steps, "length": length,
                                      "return": raw_return, "train_return": train_return,
                                      "terminated": bool(terminated), "truncated": bool(truncated),
                                      "elapsed_seconds": time.monotonic() - started})
                    obs, _ = env.reset()
                    length, raw_return, train_return = 0, 0.0, 0.0
                else:
                    obs = next_obs
                if steps in thresholds:
                    evaluator.run(algorithm.policy, "progress", steps, updates)
        torch.save(algorithm.state_dict(), args.output / "final_model.pt")
        final_eval = evaluator.run(algorithm.policy, "final", steps, updates)
        partial = ({"episode": episodes + 1, "length": length, "return": raw_return,
                    "train_return": train_return, "complete": False,
                    "note": "Budget ended without inserting a terminal transition or resetting."}
                   if length else None)
        total = time.monotonic() - started
        final = {"case": "cartpole_sac", "algorithm": "discrete_sac", "seed": args.seed,
                 "requested_steps": args.steps, "actual_steps": steps,
                 "completed_episodes": episodes, "updates": updates,
                 "last100_training_mean_return": float(np.mean(returns[-100:])) if returns else None,
                 "partial_episode_excluded": partial, "final_evaluation": final_eval,
                 "final_alpha": algorithm.alpha.value,
                 "training_seconds_excluding_evaluation": total - evaluator.seconds,
                 "evaluation_seconds": evaluator.seconds, "total_seconds": total,
                 "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                            (1 if sys.platform == "darwin" else 1024), "max_rss_unit": "bytes",
                 "checkpoint_sha256": sha256(args.output / "final_model.pt")}
        metadata.update(status="complete", actual_steps=steps, updates=updates,
                        max_rss=final["max_rss"], max_rss_unit="bytes")
        write_json(args.output / "metadata.json", metadata)
        write_json(args.output / "final.json", final)
    except BaseException:
        metadata.update(status="failed", actual_steps=steps, error=traceback.format_exc())
        write_json(args.output / "metadata.json", metadata)
        raise
    finally:
        env.close()
        evaluator.env.close()
        for sink in (train_sink, eval_sink, point_sink, update_sink):
            sink.close()


if __name__ == "__main__":
    main()
