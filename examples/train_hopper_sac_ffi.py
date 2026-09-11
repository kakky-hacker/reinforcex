"""Train Gymnasium Hopper-v5 with continuous SAC through the ReinforceX C FFI."""

import ctypes as C

from reinforcex_ffi import (
    RX_ACTION_CONTINUOUS,
    RxSacConfigV2,
    check,
    create_replay_buffer,
    create_sac,
    evaluate_gym_agent,
    load_reinforcex,
    path_for_agent,
    run_parallel,
    train_gym_agent,
    training_parser,
    validate_training_args,
)


def forward_progress_reward(
    reward: float, _step: int, _done: bool, _max_steps: int
) -> float:
    """Remove Hopper's per-step healthy bonus so standing still is not optimal."""
    return reward - 1.0


def train(args) -> None:
    validate_training_args(args)
    if args.eval_episodes < 0 or (args.eval_only and args.eval_episodes == 0):
        raise ValueError("--eval-episodes must be positive for evaluation")
    if args.eval_only and not args.load_path:
        raise ValueError("--eval-only requires --load-path")

    lib = load_reinforcex()
    config = RxSacConfigV2()
    check(lib.rx_sac_config_default_v2(C.byref(config), 11, 3), "rx_sac_config_default_v2")
    config.action_space = RX_ACTION_CONTINUOUS
    config.agent.gamma = 0.99
    config.actor_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.replay_n_steps = 1
    config.min_variance = 1e-3
    config.squash_action = 1

    if args.preset == "baseline":
        config.agent.hidden_layers = 2
        config.agent.hidden_size = 256
        config.replay_capacity = 1_000_000
        config.replay_start_size = 10_000
        config.batch_size = 256
        config.update_interval = 1
        config.target_update_interval = 1
        config.tau = 0.005
        config.alpha = 0.2
        reward_transform = None
    else:
        config.agent.hidden_layers = 1
        config.agent.hidden_size = 128
        config.replay_capacity = 300_000
        config.replay_start_size = 512
        config.batch_size = 128
        config.update_interval = 1
        config.target_update_interval = 1
        config.tau = 0.005
        config.alpha = 0.05
        reward_transform = forward_progress_reward

    if args.eval_only:
        agents = []
        try:
            for worker_id in range(args.parallel):
                agents.append(
                    create_sac(
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
                    env_id="Hopper-v5",
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
        train_kwargs = {}
        if reward_transform is not None:
            train_kwargs["reward_transform"] = reward_transform
        run_parallel(
            args.parallel,
            lambda worker_id: train_gym_agent(
                agent=agents[worker_id],
                env_id="Hopper-v5",
                agent_id=worker_id,
                seed=args.seed + worker_id * 1_000_000,
                episodes=args.episodes,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                solved_return=3_800.0,
                save_best=bool(args.save_path),
                **train_kwargs,
            ),
        )
    finally:
        for agent in reversed(agents):
            agent.close()
        replay.close()

    if args.save_path and args.eval_episodes > 0:
        evaluation_agents = []
        try:
            for worker_id in range(args.parallel):
                evaluation_agents.append(
                    create_sac(
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
                    env_id="Hopper-v5",
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
