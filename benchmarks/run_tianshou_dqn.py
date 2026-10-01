#!/usr/bin/env python3
"""Official Tianshou Double DQN control with frozen native example settings."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback

# Reuse the already verified logging/evaluation helpers without modifying their
# frozen source. That module also pins CPU numerical library thread settings.
import run_tianshou_discrete as common
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy

torch, np, gym = common.torch, common.np, common.gym
Batch, ReplayBuffer, Net = common.Batch, common.ReplayBuffer, common.Net
write_json, sha256, Jsonl = common.write_json, common.sha256, common.Jsonl


def epsilon_at_step(config, step):
    ratio = min(step / config["epsilon_decay_steps"], 1.0)
    return config["epsilon_start"] + ratio * (config["epsilon_end"] - config["epsilon_start"])


def build_algorithm(config, env):
    c = config["agents"][0]["config"]
    a = c["agent"]
    hidden = [a["hidden_size"]] * (a["hidden_layers"] + 1)
    target_freq = max(1, math.floor(c["target_update_interval"] / c["update_interval"] + .5))
    net = Net(state_shape=a["obs_size"], action_shape=a["action_size"], hidden_sizes=hidden).to("cpu")
    policy = DiscreteQLearningPolicy(model=net, action_space=env.action_space,
                                     observation_space=env.observation_space,
                                     eps_training=c["epsilon_start"], eps_inference=0.0)
    algorithm = DQN(policy=policy, optim=common.AdamOptimizerFactory(lr=c["learning_rate"]),
                    gamma=a["gamma"], n_step_return_horizon=c["replay_n_steps"],
                    target_update_freq=target_freq, is_double=True, huber_loss_delta=1.0)
    target_env = target_freq * c["update_interval"]
    resolved = {
        "net_arch": hidden, "activation": "ReLU", "is_double": True, "huber_loss_delta": 1.0,
        "max_grad_norm": None, "learning_rate": c["learning_rate"], "gamma": a["gamma"],
        "batch_size": c["batch_size"], "replay_capacity": c["replay_capacity"],
        "n_step_return_horizon": c["replay_n_steps"], "learning_starts": c["batch_size"],
        "learning_starts_rule": "strictly more than batch_size collected transitions",
        "update_interval": c["update_interval"], "target_update_freq_gradient_steps": target_freq,
        "native_target_interval_environment_steps": c["target_update_interval"],
        "nominal_target_interval_environment_steps": target_env,
        "target_interval_relative_difference": target_env / c["target_update_interval"] - 1,
        "target_interval_rounding": "nearest positive integer; half ties rounded up",
        "epsilon_start": c["epsilon_start"], "epsilon_end": c["epsilon_end"],
        "epsilon_decay_steps": c["epsilon_decay_steps"], "epsilon_inference": 0.0,
        "q_network_parameters": sum(p.numel() for p in net.parameters()),
    }
    return algorithm, resolved


class Evaluator(common.Evaluator):
    def __init__(self, args, episodes, points):
        self.args, self.episodes, self.points = args, episodes, points
        self.env = gym.make(args.env_id)
        self.seconds, self.point = 0.0, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--progress-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--success-threshold", type=float)
    args = parser.parse_args()
    if min(args.steps, args.eval_episodes, args.progress_eval_episodes, args.checkpoints) <= 0:
        parser.error("budgets and evaluation counts must be positive")
    config = json.loads(args.config.read_text())
    if (config["env_id"] not in ("CartPole-v1", "LunarLander-v3") or len(config["agents"]) != 1
            or config["agents"][0]["algorithm"] != "dqn"
            or config["agents"][0].get("rnd_config") is not None):
        parser.error("this control only supports single-agent CartPole/LunarLander DQN")
    args.env_id = config["env_id"]
    case = "cartpole_dqn" if args.env_id == "CartPole-v1" else "lunar_dqn"
    expected_mode = "cartpole" if case == "cartpole_dqn" else "raw"
    if config["reward_mode"] != expected_mode:
        parser.error("unexpected reward transform for the frozen example")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    if args.success_threshold is None:
        args.success_threshold = env.spec.reward_threshold
    algorithm, resolved = build_algorithm(config, env)
    c = config["agents"][0]["config"]
    buffer = ReplayBuffer(size=c["replay_capacity"])
    versions = {name: importlib.metadata.version(name) for name in
                ("tianshou", "torch", "gymnasium", "numpy", "numba", "Box2D")}
    metadata = {
        "status": "running", "provider": "tianshou_dqn", "supplementary": True,
        "case": case, "algorithm": "double_dqn", "seed": args.seed,
        "requested_steps": args.steps, "device": "cpu", "torch_num_threads": 1,
        "torch_num_interop_threads": 1, "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable, "versions": versions,
        "native_libtorch_version": "2.7.0", "command": sys.argv,
        "runner_sha256": sha256(__file__), "config_sha256": sha256(args.config),
        "evaluation_helper_sha256": sha256(common.__file__),
        "official_source_sha256": {str(Path(inspect.getfile(cls))): sha256(inspect.getfile(cls))
                                   for cls in (DQN, DiscreteQLearningPolicy, ReplayBuffer, Net)},
        "evaluation": {"initial_episodes": args.progress_eval_episodes,
                       "progress_points": args.checkpoints, "progress_episodes": args.progress_eval_episodes,
                       "validation_seed_start": 800000, "final_seed_start": 900000,
                       "final_episodes": args.eval_episodes, "deterministic": True},
        "comparison_caveats": [
            "Official Tianshou Double DQN, Huber(delta1), Gymnasium1.3, PyTorch2.7 match the native method/loss/environment/tensor-version choices; implementation equality is not claimed.",
            "The target period is rounded to the nearest gradient-step count, ties up: CartPole250/4 ->63 (252 environment steps, +0.8%); Lunar50/8 ->6 (48 environment steps, -4%).",
            "Official target synchronization occurs on gradient iterations, including the first; native synchronizes on absolute environment steps. Target computation and copy ordering also differ. No private method is overridden.",
            "Native DQN clips gradient norm to10; official Tianshou DQN has no public clipping constructor option and is used unchanged.",
            "Tianshou stores each transition immediately and uses shorter n-step returns at an unfinished buffer tail; native only exposes completed n-step sequences. Strictly exceeding batch_size transitions before learning aligns the first update and total update counts for these fixed configurations; sampled transition availability still differs.",
            "Native chooses the next action before updating from its previous reward; this reference collects one transition then updates. Replay sampling, RNG streams and initialization remain implementation differences.",
            "Epsilon is the native linear schedule at one-based environment step t. Official add_exploration_noise supplies epsilon-greedy training; evaluation is pure argmax.",
            "Progress evaluation runs after the current gradient update. It never selects a checkpoint or changes the training budget. Timing includes contention from other CPU jobs.",
        ],
    }
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "config.json", {"source_config": config, "resolved": resolved,
               "env_id": args.env_id, "seed": args.seed, "requested_steps": args.steps,
               "reward_transform": expected_mode, "max_episode_steps": env.spec.max_episode_steps})
    train_sink = Jsonl(args.output / "train_episodes.jsonl")
    eval_sink = Jsonl(args.output / "eval_episodes.jsonl")
    point_sink = Jsonl(args.output / "evaluations.jsonl")
    update_sink = Jsonl(args.output / "update_diagnostics.jsonl")
    evaluator = Evaluator(args, eval_sink, point_sink)
    started = time.monotonic()
    steps = episodes = updates = length = 0
    raw_return = train_return = 0.0
    returns = []
    try:
        evaluator.run(algorithm.policy, "initial", 0, 0)
        obs, _ = env.reset(seed=args.seed)
        thresholds = {math.ceil(args.steps * i / args.checkpoints) for i in range(1, args.checkpoints + 1)}
        with common.policy_within_training_step(algorithm.policy):
            for steps in range(1, args.steps + 1):
                epsilon = epsilon_at_step(c, steps)
                algorithm.policy.set_eps_training(epsilon)
                with torch.no_grad():
                    batch = Batch(obs=np.asarray([obs]), info={})
                    greedy = algorithm.policy(batch).act
                    action = int(algorithm.policy.add_exploration_noise(greedy, batch).item())
                next_obs, reward, terminated, truncated, _ = env.step(action)
                length += 1
                shaped = (common.learning_reward(reward, terminated, length, env.spec.max_episode_steps)
                          if expected_mode == "cartpole" else float(reward))
                if not math.isfinite(float(reward)) or not math.isfinite(shaped):
                    raise FloatingPointError("non-finite environment reward")
                raw_return += float(reward)
                train_return += shaped
                buffer.add(Batch(obs=obs, act=action, rew=shaped, obs_next=next_obs,
                                 terminated=terminated, truncated=truncated, info={}))
                if len(buffer) > c["batch_size"] and steps % c["update_interval"] == 0:
                    stats = algorithm.update(buffer=buffer, sample_size=c["batch_size"])
                    updates += 1
                    losses = dataclasses.asdict(stats)
                    for key, value in losses.items():
                        if isinstance(value, (int, float)) and not math.isfinite(value):
                            raise FloatingPointError(f"non-finite update metric: {key}")
                    if updates == 1 or steps in thresholds:
                        update_sink.write({"env_steps": steps, "updates": updates,
                                           "epsilon": epsilon, "metrics": losses})
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
                    "note": "Budget ended without a synthetic terminal or reset."} if length else None)
        total = time.monotonic() - started
        final = {"case": case, "algorithm": "double_dqn", "seed": args.seed,
                 "requested_steps": args.steps, "actual_steps": steps, "completed_episodes": episodes,
                 "updates": updates, "last100_training_mean_return": float(np.mean(returns[-100:])) if returns else None,
                 "partial_episode_excluded": partial, "final_evaluation": final_eval,
                 "final_epsilon": epsilon_at_step(c, steps),
                 "training_seconds_excluding_evaluation": total - evaluator.seconds,
                 "evaluation_seconds": evaluator.seconds, "total_seconds": total,
                 "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "max_rss_unit": "bytes", "checkpoint_sha256": sha256(args.output / "final_model.pt")}
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
