"""Train Gymnasium Walker2d-v5 with continuous PPO through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxPpoConfig,
    check,
    create_ppo,
    cuda_is_available,
    evaluate_gym_agent,
    load_reinforcex,
    path_for_agent,
    run_parallel,
    train_gym_agent,
    training_parser,
    validate_training_args,
)


def forward_progress_reward(
    reward: float, _step: int, _done: bool, _max_steps: int
) -> float:
    """Remove the per-step healthy bonus and scale PPO's value targets."""
    return 0.1 * (reward - 1.0)


def scaled_reward(reward: float, _step: int, _done: bool, _max_steps: int) -> float:
    """Scale the environment reward while retaining its healthy bonus."""
    return 0.1 * reward


def train(args) -> None:
    validate_training_args(args)
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.update_interval <= 0 or args.epochs <= 0 or args.minibatch_size <= 0:
        raise ValueError("PPO rollout and minibatch settings must be positive")
    if args.minibatch_size > args.update_interval:
        raise ValueError("--minibatch-size cannot exceed --update-interval")
    if args.entropy_coefficient < 0 or args.min_variance <= 0:
        raise ValueError("entropy coefficient must be non-negative and variance positive")
    if args.eval_episodes < 0 or (args.eval_only and args.eval_episodes == 0):
        raise ValueError("--eval-episodes must be positive for evaluation")
    if args.eval_only and not args.load_path:
        raise ValueError("--eval-only requires --load-path")

    lib = load_reinforcex()
    device = "cuda" if cuda_is_available(lib) else "cpu"
    print(f"reinforcex_device={device}")

    config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(config), 17, 6), "rx_ppo_config_default")
    config.action_space = RX_ACTION_CONTINUOUS
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 256
    config.agent.gamma = 0.99
    config.learning_rate = args.learning_rate
    config.gae_lambda = 0.95
    config.update_interval = args.update_interval
    config.epochs = args.epochs
    config.minibatch_size = args.minibatch_size
    config.policy_clip_epsilon = 0.2
    config.value_clip_range = 0.2
    config.value_loss_coefficient = 0.5
    config.entropy_coefficient = args.entropy_coefficient
    config.standardize_gae = 1
    config.min_action = -1.0
    config.max_action = 1.0
    config.min_variance = args.min_variance
    reward_transform = (
        scaled_reward if args.reward_mode == "survival" else forward_progress_reward
    )

    if args.eval_only:
        agents = []
        try:
            for worker_id in range(args.parallel):
                agents.append(
                    create_ppo(
                        lib,
                        config,
                        None,
                        path_for_agent(args.load_path, worker_id),
                    )
                )
            run_parallel(
                args.parallel,
                lambda worker_id: evaluate_gym_agent(
                    agent=agents[worker_id],
                    env_id="Walker2d-v5",
                    agent_id=worker_id,
                    seed=args.seed + worker_id * 1_000_000,
                    episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    render=args.render,
                ),
            )
        finally:
            for agent in reversed(agents):
                agent.close()
        return

    agents = []
    try:
        for worker_id in range(args.parallel):
            agents.append(
                create_ppo(
                    lib,
                    config,
                    path_for_agent(args.save_path, worker_id),
                    path_for_agent(args.load_path, worker_id),
                )
            )
        run_parallel(
            args.parallel,
            lambda worker_id: train_gym_agent(
                agent=agents[worker_id],
                env_id="Walker2d-v5",
                agent_id=worker_id,
                seed=args.seed + worker_id * 1_000_000,
                episodes=args.episodes,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                reward_transform=reward_transform,
                save_best=bool(args.save_path),
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()

    if args.save_path and args.eval_episodes > 0:
        evaluation_agents = []
        try:
            for worker_id in range(args.parallel):
                evaluation_agents.append(
                    create_ppo(
                        lib,
                        config,
                        None,
                        path_for_agent(args.save_path, worker_id),
                    )
                )
            run_parallel(
                args.parallel,
                lambda worker_id: evaluate_gym_agent(
                    agent=evaluation_agents[worker_id],
                    env_id="Walker2d-v5",
                    agent_id=worker_id,
                    seed=args.seed + 10_000_000 + worker_id * 1_000_000,
                    episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    render=args.render,
                ),
            )
        finally:
            for agent in reversed(evaluation_agents):
                agent.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=3_000, max_steps=1_000, log_interval=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer during evaluation")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--update-interval", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--min-variance", type=float, default=0.01)
    parser.add_argument(
        "--reward-mode",
        choices=("survival", "forward"),
        default="survival",
        help="retain the healthy bonus or remove it to emphasize forward motion",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
