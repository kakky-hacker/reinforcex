"""Train Gymnasium CartPole-v1 with discrete SAC through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_DISCRETE,
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


def shaped_reward(reward: float, step: int, done: bool, max_steps: int) -> float:
    """Keep targets small and make premature failure immediately distinguishable."""
    if done and step < max_steps:
        return -1.0
    return 0.01 * reward


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    config = RxSacConfigV2()
    check(lib.rx_sac_config_default_v2(C.byref(config), 4, 2), "rx_sac_config_default_v2")
    config.action_space = RX_ACTION_DISCRETE
    # CartPole needs little model capacity. A shorter warm-up and compact replay
    # improve sample efficiency while avoiding stale early-policy experience.
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 64
    config.agent.gamma = 0.99
    config.actor_learning_rate = 3e-4
    config.critic_learning_rate = 5e-4
    config.replay_capacity = 50_000
    config.replay_start_size = 512
    config.batch_size = 64
    config.replay_n_steps = 3
    config.update_interval = 1
    config.target_update_interval = 1
    config.tau = 0.005
    # Keep just enough adaptive entropy to avoid premature policy collapse. The
    # original 0.98 ratio makes the sampled policy nearly random indefinitely.
    config.alpha = 0.05
    config.discrete_target_entropy_ratio = 0.01
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
