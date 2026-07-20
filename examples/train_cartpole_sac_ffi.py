"""Train Gymnasium CartPole-v1 with discrete SAC through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_DISCRETE,
    RxSacConfig,
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


def shaped_reward(_reward: float, step: int, done: bool, max_steps: int) -> float:
    reward = 1.0 if step % 20 == 0 else 0.0
    if done and max_steps - step + 1 > 10:
        reward = -1.0
    return reward


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxSacConfig()
    check(lib.rx_sac_config_default(C.byref(config), 4, 2), "rx_sac_config_default")
    config.action_space = RX_ACTION_DISCRETE
    config.agent.hidden_size = 128
    config.actor_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.replay_capacity = 300_000
    config.replay_start_size = 1_000
    config.batch_size = 32
    config.replay_n_steps = 3
    config.update_interval = 4
    config.target_update_interval = 8
    config.tau = 0.01
    config.alpha = 0.3
    config.squash_action = 0

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
