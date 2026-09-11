"""Train Gymnasium LunarLanderContinuous-v3 with SAC through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxSacConfigV2,
    check,
    create_replay_buffer,
    create_sac,
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
    config = RxSacConfigV2()
    check(lib.rx_sac_config_default_v2(C.byref(config), 8, 2), "rx_sac_config_default_v2")
    config.action_space = RX_ACTION_CONTINUOUS
    config.agent.hidden_size = 128
    config.actor_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.replay_capacity = 100_000
    config.replay_start_size = 2_000
    config.batch_size = 128
    config.replay_n_steps = 1
    config.update_interval = 8
    config.target_update_interval = 8
    config.tau = 0.01
    config.alpha = 0.05
    config.min_variance = 1e-3
    config.squash_action = 1

    replay = create_replay_buffer(lib, config.replay_capacity, config.replay_n_steps)
    agents = []
    try:
        for worker_id in range(args.parallel):
            agents.append(
                create_sac(
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
                env_id="LunarLanderContinuous-v3",
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
