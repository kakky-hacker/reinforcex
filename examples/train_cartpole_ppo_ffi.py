"""Train Gymnasium CartPole-v1 with PPO through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_DISCRETE,
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


def shaped_reward(reward: float, step: int, done: bool, max_steps: int) -> float:
    """Keep value targets compact and clearly mark premature failure."""
    if done and step < max_steps:
        return -1.0
    return 0.01 * reward


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(config), 4, 2), "rx_ppo_config_default")
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
                env_id="CartPole-v1",
                agent_id=worker_id,
                seed=args.seed + worker_id * 1_000_000,
                episodes=args.episodes,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                reward_transform=shaped_reward,
                solved_return=475.0,
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=500, max_steps=500, log_interval=25)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
