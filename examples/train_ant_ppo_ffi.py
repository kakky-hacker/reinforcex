"""Train Gymnasium Ant-v5 with continuous PPO through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxPpoConfig,
    check,
    create_ppo,
    evaluate_gym_agent,
    load_reinforcex,
    manual_seed,
    path_for_agent,
    run_parallel,
    train_gym_agent,
    training_parser,
    validate_training_args,
)


def scaled_reward(reward: float, _step: int, _done: bool, _max_steps: int) -> float:
    return reward * 0.1


def clipped_reward(reward: float, _step: int, _done: bool, _max_steps: int) -> float:
    return min(1.0, max(-1.0, reward))


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    manual_seed(lib, args.seed)
    config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(config), 105, 8), "rx_ppo_config_default")
    config.action_space = RX_ACTION_CONTINUOUS
    if args.preset == "baseline":
        config.agent.hidden_size = 256
        config.learning_rate = 1e-4
        config.gae_lambda = 0.99
        config.update_interval = 512
        config.epochs = 10
        config.minibatch_size = 32
        config.policy_clip_epsilon = 0.2
        config.value_clip_range = 0.2
        config.value_loss_coefficient = 0.003
        config.entropy_coefficient = 0.005
        config.min_action = -1.0
        config.max_action = 1.0
        config.min_variance = 0.1
        reward_transform = clipped_reward
    else:
        config.agent.hidden_layers = 1
        config.agent.hidden_size = 256
        config.learning_rate = 1e-4
        config.gae_lambda = 0.95
        config.update_interval = 512
        config.epochs = 5
        config.minibatch_size = 64
        config.policy_clip_epsilon = 0.2
        config.value_clip_range = 0.2
        config.value_loss_coefficient = 0.5
        config.entropy_coefficient = 0.0
        config.min_action = -1.0
        config.max_action = 1.0
        config.min_variance = 1e-3
        reward_transform = scaled_reward

    if args.eval_only and not args.load_path:
        raise ValueError("--eval-only requires --load-path")
    if args.eval_episodes < 0 or (args.eval_only and args.eval_episodes == 0):
        raise ValueError("--eval-episodes must be positive for evaluation")

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
                    env_id="Ant-v5",
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
                env_id="Ant-v5",
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
                    env_id="Ant-v5",
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
    parser = training_parser(__doc__, episodes=1_000, max_steps=1_000, log_interval=25)
    parser.add_argument("--preset", choices=("tuned", "baseline"), default="tuned")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true", help="show the MuJoCo viewer during evaluation")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
