"""Train Gymnasium CartPole-v1 with DQN through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RxDqnConfig,
    check,
    create_dqn,
    create_replay_buffer,
    load_reinforcex,
    path_for_agent,
    run_parallel,
    train_gym_agent,
    training_parser,
    validate_training_args,
)


def shaped_reward(reward: float, step: int, done: bool, max_steps: int) -> float:
    """Keep targets small and make premature failure immediately distinguishable."""
    if done and step < max_steps:
        return -1.0
    return 0.01 * reward


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxDqnConfig()
    check(lib.rx_dqn_config_default(C.byref(config), 4, 2), "rx_dqn_config_default")
    # CartPole needs little model capacity. A smaller network and replay buffer
    # reduce both update cost and the amount of stale experience retained.
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 64
    config.agent.gamma = 0.99
    config.learning_rate = 5e-4
    config.batch_size = 64
    config.replay_capacity = 50_000
    config.replay_n_steps = 3
    config.update_interval = 4
    config.target_update_interval = 250
    config.epsilon_start = 1.0
    config.epsilon_end = 0.05
    config.epsilon_decay_steps = 10_000

    replay = create_replay_buffer(lib, config.replay_capacity, config.replay_n_steps)
    agents = []
    try:
        for worker_id in range(args.parallel):
            agents.append(
                create_dqn(
                    lib,
                    config,
                    path_for_agent(args.save_path, worker_id),
                    path_for_agent(args.load_path, worker_id),
                    replay,
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
        replay.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=500, max_steps=500, log_interval=25)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
