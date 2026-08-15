"""Train Gymnasium LunarLander-v3 with DQN through the ReinforceX C FFI."""

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


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxDqnConfig()
    check(lib.rx_dqn_config_default(C.byref(config), 8, 4), "rx_dqn_config_default")
    config.agent.hidden_size = 300
    config.learning_rate = 3e-4
    config.batch_size = 64
    config.replay_capacity = 36_000
    config.replay_n_steps = 1
    config.update_interval = 8
    config.target_update_interval = 50
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
                env_id="LunarLander-v3",
                agent_id=worker_id,
                seed=args.seed + worker_id * 1_000_000,
                episodes=args.episodes,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                solved_return=200.0,
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()
        replay.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=1_000, max_steps=1_000, log_interval=25)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
