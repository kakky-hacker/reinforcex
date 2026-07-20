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


def shaped_reward(_reward: float, step: int, done: bool, max_steps: int) -> float:
    reward = 5.0 if step % 20 == 0 else 0.0
    if done and max_steps - step + 1 > 10:
        reward = -30.0
    return reward


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxDqnConfig()
    check(lib.rx_dqn_config_default(C.byref(config), 4, 2), "rx_dqn_config_default")
    config.agent.hidden_size = 200
    config.agent.gamma = 0.97
    config.learning_rate = 3e-4
    config.batch_size = 128
    config.replay_capacity = 300_000
    config.replay_n_steps = 5
    config.update_interval = 16
    config.target_update_interval = 100
    config.epsilon_start = 0.5
    config.epsilon_end = 0.0
    config.epsilon_decay_steps = 20_000

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
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()
        replay.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=10_000, max_steps=500, log_interval=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
