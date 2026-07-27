"""Train Gymnasium Ant-v5 with continuous PPO through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxPpoConfig,
    check,
    create_ppo,
    load_reinforcex,
    path_for_agent,
    run_parallel,
    train_gym_agent,
    training_parser,
    validate_training_args,
)


def clipped_reward(reward: float, _step: int, _done: bool, _max_steps: int) -> float:
    return min(1.0, max(-1.0, reward))


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(config), 105, 8), "rx_ppo_config_default")
    config.action_space = RX_ACTION_CONTINUOUS
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
                reward_transform=clipped_reward,
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=10_000, max_steps=10_000, log_interval=50)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
