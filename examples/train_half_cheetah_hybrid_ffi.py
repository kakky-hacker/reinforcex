"""Train HalfCheetah-v5 PPO and SAC agents with one shared replay buffer."""

import argparse
import ctypes as C
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxPpoConfig,
    RxSacConfig,
    check,
    create_ppo,
    create_replay_buffer,
    create_sac,
    cuda_is_available,
    evaluate_gym_agent,
    load_reinforcex,
    train_gym_agent,
)


ENV_ID = "HalfCheetah-v5"
OBS_SIZE = 17
ACTION_SIZE = 6


def candidate_path(template: str, algorithm: str, worker_id: int) -> str:
    return template.replace("{algorithm}", algorithm).replace("{agent_id}", str(worker_id))


def configure_ppo(lib, args) -> RxPpoConfig:
    config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(config), OBS_SIZE, ACTION_SIZE), "rx_ppo_config_default")
    config.action_space = RX_ACTION_CONTINUOUS
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 256
    config.agent.gamma = 0.99
    config.learning_rate = args.ppo_learning_rate
    config.gae_lambda = 0.95
    config.update_interval = args.ppo_rollout
    config.epochs = 10
    config.minibatch_size = args.ppo_minibatch_size
    config.policy_clip_epsilon = 0.2
    config.value_clip_range = 0.2
    config.value_loss_coefficient = 0.5
    config.entropy_coefficient = args.ppo_entropy
    config.standardize_gae = 1
    config.min_action = -1.0
    config.max_action = 1.0
    config.min_variance = args.ppo_min_variance
    return config


def configure_sac(lib, args) -> RxSacConfig:
    config = RxSacConfig()
    check(lib.rx_sac_config_default(C.byref(config), OBS_SIZE, ACTION_SIZE), "rx_sac_config_default")
    config.action_space = RX_ACTION_CONTINUOUS
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 256
    config.agent.gamma = 0.99
    config.actor_learning_rate = args.sac_learning_rate
    config.critic_learning_rate = args.sac_learning_rate
    config.replay_capacity = args.replay_capacity
    config.replay_start_size = args.replay_start_size
    config.replay_n_steps = 1
    config.batch_size = args.sac_batch_size
    config.update_interval = args.sac_update_interval
    config.target_update_interval = 1
    config.tau = 0.005
    config.alpha = args.sac_alpha
    config.min_variance = 1e-3
    config.squash_action = 1
    return config


def evaluate_candidates(lib, args, algorithm, config, worker_count, path_template):
    results = [None] * worker_count

    def evaluate(worker_id):
        load_path = candidate_path(path_template, algorithm, worker_id)
        create = create_ppo if algorithm == "ppo" else create_sac
        agent = create(lib, config, None, load_path)
        try:
            returns = evaluate_gym_agent(
                agent=agent,
                env_id=ENV_ID,
                agent_id=f"{algorithm}-{worker_id}",
                seed=args.seed + 20_000_000 + worker_id * 1_000_000,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                render=False,
            )
            results[worker_id] = returns
        finally:
            agent.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(evaluate, worker_id) for worker_id in range(worker_count)]
        for future in futures:
            future.result()

    means = [float(np.mean(returns)) for returns in results]
    best_worker = int(np.argmax(means))
    return best_worker, means, results


def promote_checkpoint(lib, algorithm, config, source_path, destination_path):
    create = create_ppo if algorithm == "ppo" else create_sac
    agent = create(lib, config, destination_path, source_path)
    try:
        agent.save()
    finally:
        agent.close()


def evaluation_summary(returns):
    values = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": int(values.size),
        "mean_return": float(np.mean(values)),
        "std_return": float(np.std(values)),
        "min_return": float(np.min(values)),
        "max_return": float(np.max(values)),
    }


def evaluate_loaded(lib, args, algorithm, config, load_path, render):
    create = create_ppo if algorithm == "ppo" else create_sac
    agent = create(lib, config, None, load_path)
    try:
        return evaluate_gym_agent(
            agent=agent,
            env_id=ENV_ID,
            agent_id=algorithm,
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            render=render,
        )
    finally:
        agent.close()


def validate_args(args):
    for name in (
        "episodes",
        "max_steps",
        "log_interval",
        "ppo_workers",
        "sac_workers",
        "eval_episodes",
        "replay_capacity",
        "replay_start_size",
        "sac_batch_size",
        "sac_update_interval",
        "ppo_rollout",
        "ppo_minibatch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("ppo_episodes", "sac_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.replay_start_size > args.replay_capacity:
        raise ValueError("--replay-start-size cannot exceed --replay-capacity")
    if args.sac_batch_size > args.replay_capacity:
        raise ValueError("--sac-batch-size cannot exceed --replay-capacity")
    if args.ppo_minibatch_size > args.ppo_rollout:
        raise ValueError("--ppo-minibatch-size cannot exceed --ppo-rollout")
    if args.ppo_learning_rate <= 0 or args.sac_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.ppo_entropy < 0 or args.ppo_min_variance <= 0 or args.sac_alpha < 0:
        raise ValueError("entropy, variance, and alpha settings are invalid")


def train(args):
    validate_args(args)
    lib = load_reinforcex()
    device = "cuda" if cuda_is_available(lib) else "cpu"
    print(f"reinforcex_device={device}")
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA was required but is not available")

    ppo_config = configure_ppo(lib, args)
    sac_config = configure_sac(lib, args)

    if args.eval_only:
        algorithms = ("ppo", "sac") if args.eval_algorithm == "both" else (args.eval_algorithm,)
        for algorithm in algorithms:
            load_path = args.load_ppo_path if algorithm == "ppo" else args.load_sac_path
            if not load_path:
                raise ValueError(f"--load-{algorithm}-path is required")
            config = ppo_config if algorithm == "ppo" else sac_config
            evaluate_loaded(lib, args, algorithm, config, load_path, args.render)
        return

    replay = create_replay_buffer(lib, args.replay_capacity, 1)
    ppo_episodes = args.episodes if args.ppo_episodes is None else args.ppo_episodes
    sac_episodes = args.episodes if args.sac_episodes is None else args.sac_episodes
    ppo_agents = []
    sac_agents = []
    try:
        for worker_id in range(args.ppo_workers):
            ppo_agents.append(
                create_ppo(
                    lib,
                    ppo_config,
                    candidate_path(args.ppo_save_path, "ppo", worker_id),
                    (
                        candidate_path(args.load_ppo_path, "ppo", worker_id)
                        if args.load_ppo_path
                        else None
                    ),
                    replay=replay,
                )
            )
        for worker_id in range(args.sac_workers):
            sac_agents.append(
                create_sac(
                    lib,
                    sac_config,
                    candidate_path(args.sac_save_path, "sac", worker_id),
                    (
                        candidate_path(args.load_sac_path, "sac", worker_id)
                        if args.load_sac_path
                        else None
                    ),
                    replay,
                )
            )

        jobs = []
        for worker_id, agent in enumerate(ppo_agents):
            jobs.append(("ppo", worker_id, agent))
        for worker_id, agent in enumerate(sac_agents):
            jobs.append(("sac", worker_id, agent))

        def train_worker(job):
            algorithm, worker_id, agent = job
            seed_offset = 0 if algorithm == "ppo" else 10_000_000
            returns = train_gym_agent(
                agent=agent,
                env_id=ENV_ID,
                agent_id=f"{algorithm}-{worker_id}",
                seed=args.seed + seed_offset + worker_id * 1_000_000,
                episodes=ppo_episodes if algorithm == "ppo" else sac_episodes,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                save_best=True,
            )
            return algorithm, worker_id, returns

        training_returns = {
            "ppo": [None] * args.ppo_workers,
            "sac": [None] * args.sac_workers,
        }
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(train_worker, job) for job in jobs]
            for future in futures:
                algorithm, worker_id, returns = future.result()
                training_returns[algorithm][worker_id] = returns

        replay_size = len(replay)
        print(f"shared_replay_size={replay_size}")
    finally:
        for agent in reversed(sac_agents):
            agent.close()
        for agent in reversed(ppo_agents):
            agent.close()
        replay.close()

    ppo_best, ppo_means, ppo_eval_returns = evaluate_candidates(
        lib, args, "ppo", ppo_config, args.ppo_workers, args.ppo_save_path
    )
    sac_best, sac_means, sac_eval_returns = evaluate_candidates(
        lib, args, "sac", sac_config, args.sac_workers, args.sac_save_path
    )
    ppo_source = candidate_path(args.ppo_save_path, "ppo", ppo_best)
    sac_source = candidate_path(args.sac_save_path, "sac", sac_best)
    promote_checkpoint(lib, "ppo", ppo_config, ppo_source, args.best_ppo_path)
    promote_checkpoint(lib, "sac", sac_config, sac_source, args.best_sac_path)

    result = {
        "environment": ENV_ID,
        "device": device,
        "shared_replay_size": replay_size,
        "episodes_per_worker": {"ppo": ppo_episodes, "sac": sac_episodes},
        "ppo": {
            "workers": args.ppo_workers,
            "candidate_means": ppo_means,
            "best_worker": ppo_best,
            "best_mean_return": ppo_means[ppo_best],
            "best_model": args.best_ppo_path,
            "evaluation": evaluation_summary(ppo_eval_returns[ppo_best]),
            "training_returns": training_returns["ppo"],
        },
        "sac": {
            "workers": args.sac_workers,
            "candidate_means": sac_means,
            "best_worker": sac_best,
            "best_mean_return": sac_means[sac_best],
            "best_model": args.best_sac_path,
            "evaluation": evaluation_summary(sac_eval_returns[sac_best]),
            "training_returns": training_returns["sac"],
        },
    }
    result_path = Path(args.results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"selected ppo_worker={ppo_best} mean={ppo_means[ppo_best]:.2f} "
        f"sac_worker={sac_best} mean={sac_means[sac_best]:.2f}"
    )
    print(f"results_path={result_path}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--episodes", type=int, default=300)
    result.add_argument("--ppo-episodes", type=int)
    result.add_argument("--sac-episodes", type=int)
    result.add_argument("--max-steps", type=int, default=1000)
    result.add_argument("--log-interval", type=int, default=50)
    result.add_argument("--ppo-workers", type=int, default=2)
    result.add_argument("--sac-workers", type=int, default=2)
    result.add_argument("--eval-episodes", type=int, default=30)
    result.add_argument("--replay-capacity", type=int, default=1_000_000)
    result.add_argument("--replay-start-size", type=int, default=5_000)
    result.add_argument("--sac-batch-size", type=int, default=256)
    result.add_argument("--sac-update-interval", type=int, default=2)
    result.add_argument("--sac-learning-rate", type=float, default=3e-4)
    result.add_argument("--sac-alpha", type=float, default=0.2)
    result.add_argument("--ppo-rollout", type=int, default=2048)
    result.add_argument("--ppo-minibatch-size", type=int, default=128)
    result.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    result.add_argument("--ppo-entropy", type=float, default=0.01)
    result.add_argument("--ppo-min-variance", type=float, default=0.02)
    result.add_argument(
        "--ppo-save-path",
        default="artifacts/half_cheetah_ppo_candidate_{agent_id}.ot",
    )
    result.add_argument(
        "--sac-save-path",
        default="artifacts/half_cheetah_sac_candidate_{agent_id}.ot",
    )
    result.add_argument("--best-ppo-path", default="artifacts/half_cheetah_ppo_best.ot")
    result.add_argument("--best-sac-path", default="artifacts/half_cheetah_sac_best.ot")
    result.add_argument(
        "--results-path",
        default="artifacts/half-cheetah-ppo-sac-shared-results.json",
    )
    result.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--eval-only", action="store_true")
    result.add_argument("--eval-algorithm", choices=("ppo", "sac", "both"), default="both")
    result.add_argument("--load-ppo-path")
    result.add_argument("--load-sac-path")
    result.add_argument("--render", action="store_true")
    return result


if __name__ == "__main__":
    train(parser().parse_args())
