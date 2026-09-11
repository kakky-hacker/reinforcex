"""Compare Ant-v5 PPO+RND/SAC sharing, PPO/SAC sharing, and SAC alone."""

import argparse
import ctypes as C
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxPpoConfig,
    RxRndConfig,
    RxSacConfigV2,
    check,
    create_ppo,
    create_replay_buffer,
    create_rnd,
    create_sac,
    cuda_is_available,
    evaluate_gym_agent,
    load_reinforcex,
    manual_seed,
    train_gym_agent,
)


ENV_ID = "Ant-v5"
OBS_SIZE = 105
ACTION_SIZE = 8
RND_SHARED = "rnd_shared"
PPO_SHARED = "ppo_shared"
SAC_ONLY = "sac_only"
ALL_CONDITIONS = (RND_SHARED, PPO_SHARED, SAC_ONLY)


def scaled_reward(reward: float, _step: int, _done: bool, _max_steps: int) -> float:
    """Remove Ant's constant healthy bonus, then keep targets compact."""
    return (reward - 1.0) * 0.1


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
    config.epochs = args.ppo_epochs
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


def configure_sac(lib, args, update_interval: int) -> RxSacConfigV2:
    config = RxSacConfigV2()
    check(lib.rx_sac_config_default_v2(C.byref(config), OBS_SIZE, ACTION_SIZE), "rx_sac_config_default_v2")
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
    config.update_interval = update_interval
    config.target_update_interval = 1
    config.tau = 0.005
    config.alpha = args.sac_alpha
    config.min_variance = 1e-3
    config.squash_action = 1
    return config


def configure_rnd(lib, args) -> RxRndConfig:
    config = RxRndConfig()
    check(lib.rx_rnd_config_default(C.byref(config), OBS_SIZE), "rx_rnd_config_default")
    config.feature_size = args.rnd_feature_size
    config.hidden_layers = 1
    config.hidden_size = 256
    config.learning_rate = args.rnd_learning_rate
    config.update_interval = args.rnd_minibatch_size
    return config


def condition_paths(args, condition: str) -> dict[str, str]:
    stem = f"{args.run_name}_{condition}"
    return {
        "ppo_candidate": f"artifacts/{stem}_ppo_candidate_{{worker_id}}.ot",
        "rnd_candidate": f"artifacts/{stem}_rnd_candidate_{{worker_id}}",
        "sac_candidate": f"artifacts/{stem}_sac_candidate_{{worker_id}}.ot",
        "ppo_best": f"artifacts/{stem}_ppo_best.ot",
        "rnd_best": f"artifacts/{stem}_rnd_best",
        "sac_best": f"artifacts/{stem}_sac_best.ot",
    }


def worker_path(template: str, worker_id: int) -> str:
    return template.replace("{worker_id}", str(worker_id))


def evaluation_summary(returns) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": int(values.size),
        "mean_return": float(np.mean(values)),
        "std_return": float(np.std(values)),
        "min_return": float(np.min(values)),
        "max_return": float(np.max(values)),
    }


def evaluate_candidates(lib, args, algorithm, config, worker_count, path_template):
    all_returns = []
    for worker_id in range(worker_count):
        load_path = worker_path(path_template, worker_id)
        create = create_ppo if algorithm == "ppo" else create_sac
        agent = create(lib, config, None, load_path)
        try:
            returns = evaluate_gym_agent(
                agent=agent,
                env_id=ENV_ID,
                agent_id=f"{algorithm}-{worker_id}",
                seed=args.seed + 50_000_000,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
                render=False,
            )
            all_returns.append(returns)
        finally:
            agent.close()
    means = [float(np.mean(returns)) for returns in all_returns]
    best_worker = int(np.argmax(means))
    return best_worker, means, all_returns


def promote_agent(lib, algorithm, config, source_path, destination_path):
    create = create_ppo if algorithm == "ppo" else create_sac
    agent = create(lib, config, destination_path, source_path)
    try:
        agent.save()
    finally:
        agent.close()


def promote_rnd(lib, config, source_path, destination_path):
    rnd = create_rnd(lib, config, destination_path, source_path)
    try:
        rnd.save()
    finally:
        rnd.close()


def write_results(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"results_path={path}")


def run_condition(lib, args, condition: str):
    hybrid = condition in (RND_SHARED, PPO_SHARED)
    rnd_enabled = condition == RND_SHARED
    paths = condition_paths(args, condition)
    ppo_config = configure_ppo(lib, args)
    sac_update_interval = (
        args.hybrid_sac_update_interval if hybrid else args.sac_only_update_interval
    )
    sac_config = configure_sac(lib, args, sac_update_interval)
    rnd_config = configure_rnd(lib, args) if rnd_enabled else None
    replay = create_replay_buffer(lib, args.replay_capacity, 1)
    ppo_agents = []
    sac_agents = []
    rnd_modules = []
    training_returns = {"ppo": [], "sac": []}
    try:
        if hybrid:
            if rnd_enabled:
                manual_seed(lib, args.seed + 100)
                rnd_modules.append(
                    create_rnd(
                        lib,
                        rnd_config,
                        worker_path(paths["rnd_candidate"], 0),
                        None,
                    )
                )
            manual_seed(lib, args.seed + 1)
            ppo_agents.append(
                create_ppo(
                    lib,
                    ppo_config,
                    worker_path(paths["ppo_candidate"], 0),
                    None,
                    rnd=rnd_modules[0] if rnd_enabled else None,
                    curiosity_reward_coefficient=args.rnd_coefficient,
                    replay=replay,
                )
            )
        manual_seed(lib, args.seed + 2)
        sac_agents.append(
            create_sac(
                lib,
                sac_config,
                worker_path(paths["sac_candidate"], 0),
                None,
                replay,
            )
        )

        jobs = []
        if hybrid:
            jobs.append(("ppo", ppo_agents[0], args.hybrid_episodes, args.seed))
            sac_episodes = args.hybrid_episodes
        else:
            sac_episodes = args.sac_only_episodes
        jobs.append(("sac", sac_agents[0], sac_episodes, args.seed + 10_000_000))

        def train_worker(job):
            algorithm, agent, episodes, seed = job
            returns = train_gym_agent(
                agent=agent,
                env_id=ENV_ID,
                agent_id=f"{condition}-{algorithm}",
                seed=seed,
                episodes=episodes,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                reward_transform=scaled_reward,
                save_best=True,
            )
            return algorithm, returns

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(train_worker, job) for job in jobs]
            for future in futures:
                algorithm, returns = future.result()
                training_returns[algorithm].append(returns)

        final_statistics = {
            "ppo": [agent.statistics() for agent in ppo_agents],
            "sac": [agent.statistics() for agent in sac_agents],
        }
        replay_size = len(replay)
        print(f"condition={condition} shared_replay_size={replay_size}")
    finally:
        for agent in reversed(sac_agents):
            agent.close()
        for agent in reversed(ppo_agents):
            agent.close()
        for rnd in reversed(rnd_modules):
            rnd.close()
        replay.close()

    result = {
        "condition": condition,
        "rnd_enabled": rnd_enabled,
        "reward_transform": "0.1 * (raw_reward - 1.0 healthy_bonus)",
        "shared_replay_size": replay_size,
        "episodes": {
            "ppo": args.hybrid_episodes if hybrid else 0,
            "sac": args.hybrid_episodes if hybrid else args.sac_only_episodes,
        },
        "sac_update_interval": sac_update_interval,
        "final_statistics": final_statistics,
        "training_returns": training_returns,
    }

    if hybrid:
        ppo_best, ppo_means, ppo_eval_returns = evaluate_candidates(
            lib, args, "ppo", ppo_config, 1, paths["ppo_candidate"]
        )
        ppo_source = worker_path(paths["ppo_candidate"], ppo_best)
        promote_agent(lib, "ppo", ppo_config, ppo_source, paths["ppo_best"])
        result["ppo"] = {
            "candidate_means": ppo_means,
            "best_worker": ppo_best,
            "best_model": paths["ppo_best"],
            "evaluation": evaluation_summary(ppo_eval_returns[ppo_best]),
        }
        if rnd_enabled:
            rnd_source = worker_path(paths["rnd_candidate"], ppo_best)
            promote_rnd(lib, rnd_config, rnd_source, paths["rnd_best"])
            result["ppo"]["rnd_model"] = paths["rnd_best"]

    sac_best, sac_means, sac_eval_returns = evaluate_candidates(
        lib, args, "sac", sac_config, 1, paths["sac_candidate"]
    )
    sac_source = worker_path(paths["sac_candidate"], sac_best)
    promote_agent(lib, "sac", sac_config, sac_source, paths["sac_best"])
    result["sac"] = {
        "candidate_means": sac_means,
        "best_worker": sac_best,
        "best_model": paths["sac_best"],
        "evaluation": evaluation_summary(sac_eval_returns[sac_best]),
    }
    print(
        f"condition={condition} selected_sac_mean="
        f"{result['sac']['evaluation']['mean_return']:.2f}"
    )
    return result


def evaluate_only(lib, args):
    condition = args.eval_condition
    paths = condition_paths(args, condition)
    if args.eval_algorithm == "ppo":
        if condition == SAC_ONLY:
            raise ValueError("sac_only has no PPO checkpoint")
        config = configure_ppo(lib, args)
        load_path = args.load_path or paths["ppo_best"]
        agent = create_ppo(lib, config, None, load_path)
    else:
        update_interval = (
            args.sac_only_update_interval
            if condition == SAC_ONLY
            else args.hybrid_sac_update_interval
        )
        config = configure_sac(lib, args, update_interval)
        load_path = args.load_path or paths["sac_best"]
        agent = create_sac(lib, config, None, load_path)
    try:
        evaluate_gym_agent(
            agent=agent,
            env_id=ENV_ID,
            agent_id=f"{condition}-{args.eval_algorithm}",
            seed=args.seed,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            render=args.render,
        )
    finally:
        agent.close()


def validate_args(args):
    positive_names = (
        "hybrid_episodes",
        "sac_only_episodes",
        "max_steps",
        "log_interval",
        "eval_episodes",
        "replay_capacity",
        "replay_start_size",
        "sac_batch_size",
        "hybrid_sac_update_interval",
        "sac_only_update_interval",
        "ppo_rollout",
        "ppo_epochs",
        "ppo_minibatch_size",
        "rnd_feature_size",
        "rnd_minibatch_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.replay_start_size > args.replay_capacity:
        raise ValueError("--replay-start-size cannot exceed --replay-capacity")
    if args.sac_batch_size > args.replay_capacity:
        raise ValueError("--sac-batch-size cannot exceed --replay-capacity")
    if args.ppo_minibatch_size > args.ppo_rollout:
        raise ValueError("--ppo-minibatch-size cannot exceed --ppo-rollout")
    if args.ppo_learning_rate <= 0 or args.sac_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.rnd_learning_rate <= 0 or args.rnd_coefficient < 0:
        raise ValueError("RND settings are invalid")
    if args.ppo_entropy < 0 or args.ppo_min_variance <= 0 or args.sac_alpha < 0:
        raise ValueError("entropy, variance, and alpha settings are invalid")


def train(args):
    validate_args(args)
    lib = load_reinforcex()
    device = "cuda" if cuda_is_available(lib) else "cpu"
    print(f"reinforcex_device={device}")
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA was required but is not available")
    if args.eval_only:
        evaluate_only(lib, args)
        return

    results_path = Path(
        args.results_path or f"artifacts/{args.run_name}-ppo-rnd-sac-comparison.json"
    )
    payload = {
        "environment": ENV_ID,
        "device": device,
        "seed": args.seed,
        "protocol": {
            "evaluation_seed_start": args.seed + 50_000_000,
            "evaluation_episodes": args.eval_episodes,
            "hybrid_max_transitions": args.hybrid_episodes * 2 * args.max_steps,
            "sac_only_max_transitions": args.sac_only_episodes * args.max_steps,
            "hybrid_sac_update_interval": args.hybrid_sac_update_interval,
            "sac_only_update_interval": args.sac_only_update_interval,
            "budget_basis": (
                "Fixed episode and per-episode step limits; actual transitions and "
                "updates differ when Ant terminates early and are reported per condition."
            ),
            "note": (
                "RND changes PPO behavior only; shared replay stores the same "
                "healthy-bonus-removed extrinsic reward for every condition."
            ),
        },
        "conditions": {},
    }
    for condition in args.conditions:
        print(f"condition={condition} start")
        payload["conditions"][condition] = run_condition(lib, args, condition)
        write_results(results_path, payload)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, default=list(ALL_CONDITIONS))
    result.add_argument("--run-name", default="ant")
    result.add_argument("--seed", type=int, default=20260730)
    result.add_argument("--hybrid-episodes", type=int, default=400)
    result.add_argument("--sac-only-episodes", type=int, default=800)
    result.add_argument("--max-steps", type=int, default=1000)
    result.add_argument("--log-interval", type=int, default=50)
    result.add_argument("--eval-episodes", type=int, default=30)
    result.add_argument("--replay-capacity", type=int, default=1_000_000)
    result.add_argument("--replay-start-size", type=int, default=10_000)
    result.add_argument("--sac-batch-size", type=int, default=256)
    result.add_argument("--hybrid-sac-update-interval", type=int, default=2)
    result.add_argument("--sac-only-update-interval", type=int, default=4)
    result.add_argument("--sac-learning-rate", type=float, default=3e-4)
    result.add_argument("--sac-alpha", type=float, default=0.2)
    result.add_argument("--ppo-rollout", type=int, default=512)
    result.add_argument("--ppo-epochs", type=int, default=5)
    result.add_argument("--ppo-minibatch-size", type=int, default=64)
    result.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    result.add_argument("--ppo-entropy", type=float, default=0.0)
    result.add_argument("--ppo-min-variance", type=float, default=0.02)
    result.add_argument("--rnd-feature-size", type=int, default=128)
    result.add_argument("--rnd-minibatch-size", type=int, default=128)
    result.add_argument("--rnd-learning-rate", type=float, default=1e-4)
    result.add_argument("--rnd-coefficient", type=float, default=0.01)
    result.add_argument("--results-path")
    result.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--eval-only", action="store_true")
    result.add_argument("--eval-condition", choices=ALL_CONDITIONS, default=RND_SHARED)
    result.add_argument("--eval-algorithm", choices=("ppo", "sac"), default="sac")
    result.add_argument("--load-path")
    result.add_argument("--render", action="store_true")
    return result


if __name__ == "__main__":
    train(parser().parse_args())
