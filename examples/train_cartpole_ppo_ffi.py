"""Train CartPole PPO for a fixed CPU step budget, then evaluate raw returns.

The default separate-Tanh model and linear learning-rate schedule were checked
at 204,800 steps per agent. --episodes selects an unvalidated episode budget and
requires an explicit --schedule-steps. --load-path is a weights-only warm start
when training: optimizer, replay, and the schedule counter start fresh. Use
--eval-only to evaluate a saved model without learning; --legacy-model loads the
previous shared-ReLU architecture. Different architectures are not convertible.
"""

import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

from reinforcex_ffi import (
    RX_ACTION_DISCRETE,
    RxPpoConfig,
    RxPpoConfigV2,
    check,
    create_ppo,
    load_reinforcex,
    manual_seed,
    path_for_agent,
)
from reinforcex_schedules import LinearLearningRateAgent
from reinforcex_training import (
    close_agents, evaluate_agent, resolve_budget, run_workers, train_agent,
    training_parser, validate_output_paths,
)


def shaped_reward(reward: float, step: int, done: bool, max_steps: int) -> float:
    """Keep value targets compact and clearly mark premature failure."""
    if done and step < max_steps:
        return -1.0
    return 0.01 * reward


def configure_ppo(lib, *, legacy=False):
    if legacy:
        config = RxPpoConfig()
        check(lib.rx_ppo_config_default(C.byref(config), 4, 2), "rx_ppo_config_default")
    else:
        if not hasattr(lib, "rx_ppo_config_default_v2") or not hasattr(lib, "rx_ppo_create_v2"):
            raise RuntimeError("Rebuild ReinforceX with PPO V2 support or use --legacy-model for an old checkpoint")
        config = RxPpoConfigV2()
        check(lib.rx_ppo_config_default_v2(C.byref(config), 4, 2), "rx_ppo_config_default_v2")
        config.model = 1  # Separate actor/value networks.
        config.activation = 0  # Tanh; hidden_layers=1 means two hidden layers.
        config.initial_log_std = 0.0
        config.adam_epsilon = 1e-8
        config.target_kl = 0.0
    config.action_space = RX_ACTION_DISCRETE
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 64
    config.agent.gamma = 0.99
    config.learning_rate = 5e-4
    config.gae_lambda = 0.95
    config.update_interval = 256
    config.epochs = 6
    config.minibatch_size = 64
    config.policy_clip_epsilon = 0.2
    config.value_clip_range = 0.2
    config.value_loss_coefficient = 0.5
    config.entropy_coefficient = 0.0
    config.standardize_gae = 1
    return config


def train(args):
    budget = resolve_budget(args, scheduled=not args.legacy_model)
    # Reject an occupied output before expensive training; do not overwrite it.
    result_path = Path(args.results_path) if args.results_path else None
    if result_path is not None and result_path.exists():
        raise FileExistsError(f"results already exist: {result_path}")
    outputs = [result_path]
    if not args.eval_only:
        for worker_id in range(args.parallel):
            save_path = path_for_agent(args.save_path, worker_id)
            outputs.append(save_path)
            if save_path and not args.legacy_model:
                outputs.append(save_path + ".learning_rate.json")
    validate_output_paths(outputs)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(__file__).with_name("reinforcex_training.py"),
                                 Path(__file__).with_name("reinforcex_schedules.py"),
                                 Path(__file__).with_name("reinforcex_ffi.py"))}
    lib = load_reinforcex()
    library_path = Path(lib._name)
    library_hash = hashlib.sha256(library_path.read_bytes()).hexdigest() if library_path.is_file() else None
    if not args.eval_only and not args.legacy_model and not hasattr(lib, "rx_agent_set_learning_rate"):
        raise RuntimeError("Rebuild ReinforceX with rx_agent_set_learning_rate support")
    config = configure_ppo(lib, legacy=args.legacy_model)
    if args.load_path and not args.eval_only:
        print("Loading weights only; optimizer, replay, and LR schedule start fresh.", flush=True)

    agents = []
    try:
        for worker_id in range(args.parallel):
            manual_seed(lib, args.seed + worker_id * 1_000_000)
            save_path = path_for_agent(args.save_path, worker_id)
            agent = create_ppo(
                lib, config, save_path, path_for_agent(args.load_path, worker_id),
            )
            # Retain ownership immediately, including when wrapper setup fails.
            agents.append(agent)
            if not args.eval_only and not args.legacy_model:
                agents[-1] = LinearLearningRateAgent(
                    agent, config.learning_rate,
                    total_steps=budget.schedule_horizon_steps, final_fraction=.05,
                    save_path=save_path + ".learning_rate.json" if save_path else None,
                )
        trained = None
        if not args.eval_only:
            trained = run_workers(args.parallel, lambda worker_id, cancel: train_agent(
                agents[worker_id], "CartPole-v1", budget=budget,
                agent_id=worker_id, seed=args.seed + worker_id * 1_000_000,
                max_steps=args.max_steps, log_interval=args.log_interval,
                reward_transform=shaped_reward, save_enabled=bool(args.save_path),
                cancel_event=cancel,
            ))
        # Complete all training before evaluation, including in parallel mode.
        evaluated = run_workers(args.parallel, lambda worker_id, cancel: evaluate_agent(
            agents[worker_id], "CartPole-v1", agent_id=worker_id,
            seed=args.eval_seed, episodes=args.eval_episodes, max_steps=args.max_steps,
            cancel_event=cancel,
        ))
        result = {"environment": "CartPole-v1", "algorithm": "ppo",
                  "model": "legacy" if args.legacy_model else "separate_tanh",
                  "args": vars(args), "training": trained, "evaluation": evaluated,
                  "success_threshold": 475.0,
                  "every_worker_passed": all(r["mean_return"] >= 475 for r in evaluated),
                  "statistics": [agent.statistics() for agent in agents],
                  "checkpoint_scope": "weights; training load resets optimizer/replay/schedule",
                  "library": str(lib._name),
                  "library_sha256": library_hash,
                  "source_sha256_at_start": source_hashes}
        result["source_sha256_at_end"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in source_hashes
        }
        result["source_unchanged"] = result["source_sha256_at_end"] == source_hashes
        for worker_id, evaluation in enumerate(evaluated):
            print(f"agent={worker_id} final_raw_mean={evaluation['mean_return']:.2f} "
                  f"episodes={args.eval_episodes} threshold=475 "
                  f"passed={evaluation['mean_return'] >= 475}", flush=True)
        if result_path is not None:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
        return result
    finally:
        close_agents(agents, primary_error=sys.exc_info()[1])


def parser():
    result = training_parser(__doc__, steps_per_agent=204_800, max_steps=500, log_interval=25)
    result.add_argument("--legacy-model", action="store_true",
                        help="previous shared-ReLU model without LR schedule; not the validated default")
    result.add_argument("--results-path", help="save per-episode training and final evaluation JSON")
    return result


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
