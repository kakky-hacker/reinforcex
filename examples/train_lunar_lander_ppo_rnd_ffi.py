"""Train LunarLander-v3 with PPO and one RND predictor per FFI worker."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_DISCRETE,
    RxPpoConfig,
    RxRndConfig,
    check,
    create_ppo,
    create_rnd,
    load_reinforcex,
    path_for_agent,
    rnd_path,
    run_parallel,
    train_gym_agent,
    training_parser,
    validate_training_args,
)


def train(args) -> None:
    validate_training_args(args)
    lib = load_reinforcex()
    ppo_config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(ppo_config), 8, 4), "rx_ppo_config_default")
    ppo_config.action_space = RX_ACTION_DISCRETE
    ppo_config.agent.hidden_size = 256
    ppo_config.learning_rate = 2.5e-4
    ppo_config.gae_lambda = 0.95
    ppo_config.update_interval = 2_048
    ppo_config.epochs = 10
    ppo_config.minibatch_size = 64
    ppo_config.policy_clip_epsilon = 0.2
    ppo_config.value_clip_range = 0.2
    ppo_config.value_loss_coefficient = 0.5
    ppo_config.entropy_coefficient = 0.01

    rnd_config = RxRndConfig()
    check(lib.rx_rnd_config_default(C.byref(rnd_config), 8), "rx_rnd_config_default")
    rnd_config.feature_size = 128
    rnd_config.hidden_layers = 2
    rnd_config.hidden_size = 256
    rnd_config.learning_rate = 1e-4
    rnd_config.update_interval = 128

    agents = []
    curiosities = []
    try:
        for worker_id in range(args.parallel):
            curiosities.append(
                create_rnd(
                    lib,
                    rnd_config,
                    rnd_path(args.save_path, worker_id),
                    rnd_path(args.load_path, worker_id),
                )
            )
            agents.append(
                create_ppo(
                    lib,
                    ppo_config,
                    path_for_agent(args.save_path, worker_id),
                    path_for_agent(args.load_path, worker_id),
                    curiosities[worker_id],
                    args.curiosity_coefficient,
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
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()
        for curiosity in reversed(curiosities):
            curiosity.close()


def main() -> None:
    parser = training_parser(__doc__, episodes=5_000, max_steps=1_000, log_interval=20)
    parser.add_argument("--curiosity-coefficient", type=float, default=1.0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
