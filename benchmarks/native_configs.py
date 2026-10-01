"""Example-faithful ReinforceX configurations for fixed-budget CPU benchmarks.

This module creates config structs only; it never constructs models or changes
the Rust core. Construct any RND first, then reseed immediately before creating
each policy so ``lunar_ppo`` and ``lunar_rnd`` start from identical policies.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import sys

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import reinforcex_ffi as rx
import train_ant_ppo_rnd_sac_shared_ffi as ant_comparison
import train_half_cheetah_hybrid_ffi as halfcheetah


CASE_NAMES = (
    "cartpole_dqn", "cartpole_ppo", "cartpole_sac", "lunar_dqn",
    "lunar_ppo", "lunar_rnd", "lunar_sac", "ant_ppo", "hopper_sac",
    "walker_ppo", "halfcheetah_hybrid", "halfcheetah_sac", "ant_shared",
    "ant_rnd_shared", "ant_sac",
)


def _default(lib, algorithm, observations, actions):
    if algorithm == "dqn":
        result, operation = rx.RxDqnConfig(), "rx_dqn_config_default"
    elif algorithm == "sac":
        result, operation = rx.RxSacConfigV2(), "rx_sac_config_default_v2"
    elif algorithm == "ppo":
        result, operation = rx.RxPpoConfig(), "rx_ppo_config_default"
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    rx.check(getattr(lib, operation)(C.byref(result), observations, actions), operation)
    return result


def _entry(algorithm, settings, rnd=None, coefficient=0.0):
    # Distinct structs prevent per-worker overrides from mutating another worker.
    return {"algorithm": algorithm, "config": type(settings).from_buffer_copy(bytes(settings)),
            "rnd_config": None if rnd is None else type(rnd).from_buffer_copy(bytes(rnd)),
            "coefficient": float(coefficient)}


def _result(env_id, reward_mode, agents, source_scripts):
    return {"env_id": env_id, "reward_mode": reward_mode, "agents": agents,
            "shared_replay": any(a["algorithm"] in ("dqn", "sac") for a in agents),
            "source_scripts": ["examples/" + name for name in source_scripts]}


def config(lib, case):
    """Return the default example settings for one named benchmark case.

    ``lunar_ppo`` is the exact LunarLander PPO+RND policy with RND detached;
    ``halfcheetah_sac`` is the hybrid's SAC configuration as a single learner.
    Episode-based example limits/early stopping are intentionally left to the
    caller, which must enforce a total environment-step budget across agents.
    """
    if case not in CASE_NAMES:
        raise ValueError(f"unknown case {case!r}; expected one of {CASE_NAMES}")

    if case.startswith("cartpole_"):
        algorithm = case.removeprefix("cartpole_")
        settings = _default(lib, algorithm, 4, 2)
        settings.agent.hidden_layers, settings.agent.hidden_size = 1, 64
        settings.agent.gamma = 0.99
        if algorithm == "dqn":
            settings.learning_rate = 5e-4
            settings.batch_size, settings.replay_capacity = 64, 50000
            settings.replay_n_steps, settings.update_interval = 3, 4
            settings.target_update_interval = 250
            settings.epsilon_start, settings.epsilon_end = 1.0, 0.05
            settings.epsilon_decay_steps = 10000
        elif algorithm == "ppo":
            settings.action_space = rx.RX_ACTION_DISCRETE
            settings.learning_rate, settings.gae_lambda = 5e-4, 0.95
            settings.update_interval, settings.epochs, settings.minibatch_size = 256, 6, 64
            settings.policy_clip_epsilon, settings.value_clip_range = 0.2, 0.2
            settings.value_loss_coefficient, settings.entropy_coefficient = 0.5, 0.0
            settings.standardize_gae = 1
        else:
            settings.action_space = rx.RX_ACTION_DISCRETE
            settings.actor_learning_rate, settings.critic_learning_rate = 3e-4, 5e-4
            settings.replay_capacity, settings.replay_start_size = 50000, 512
            settings.batch_size, settings.replay_n_steps = 64, 3
            settings.update_interval, settings.target_update_interval = 1, 1
            settings.tau, settings.alpha = 0.005, 0.05
            settings.discrete_target_entropy_ratio, settings.squash_action = 0.01, 0
        return _result("CartPole-v1", "cartpole", [_entry(algorithm, settings)],
                       [f"train_cartpole_{algorithm}_ffi.py"])

    if case == "lunar_dqn":
        settings = _default(lib, "dqn", 8, 4)
        settings.agent.hidden_size = 300
        settings.learning_rate, settings.batch_size = 3e-4, 64
        settings.replay_capacity, settings.replay_n_steps = 36000, 1
        settings.update_interval, settings.target_update_interval = 8, 50
        settings.epsilon_start, settings.epsilon_end = 1.0, 0.05
        settings.epsilon_decay_steps = 10000
        return _result("LunarLander-v3", "raw", [_entry("dqn", settings)],
                       ["train_lunar_lander_dqn_ffi.py"])

    if case in ("lunar_ppo", "lunar_rnd"):
        settings = _default(lib, "ppo", 8, 4)
        settings.action_space, settings.agent.hidden_size = rx.RX_ACTION_DISCRETE, 128
        settings.learning_rate, settings.gae_lambda = 3e-4, 0.95
        settings.update_interval, settings.epochs, settings.minibatch_size = 1024, 6, 128
        settings.policy_clip_epsilon, settings.value_clip_range = 0.2, 0.2
        settings.value_loss_coefficient, settings.entropy_coefficient = 0.5, 0.005
        rnd = None
        if case == "lunar_rnd":
            rnd = rx.RxRndConfig()
            rx.check(lib.rx_rnd_config_default(C.byref(rnd), 8), "rx_rnd_config_default")
            rnd.feature_size, rnd.hidden_layers, rnd.hidden_size = 64, 1, 128
            rnd.learning_rate, rnd.update_interval = 1e-4, 128
        return _result("LunarLander-v3", "raw",
                       [_entry("ppo", settings, rnd, 0.01 if rnd is not None else 0.0)],
                       ["train_lunar_lander_ppo_rnd_ffi.py"])

    if case == "lunar_sac":
        settings = _default(lib, "sac", 8, 2)
        settings.action_space, settings.agent.hidden_size = rx.RX_ACTION_CONTINUOUS, 128
        settings.actor_learning_rate, settings.critic_learning_rate = 3e-4, 3e-4
        settings.replay_capacity, settings.replay_start_size = 100000, 2000
        settings.batch_size, settings.replay_n_steps = 128, 1
        settings.update_interval, settings.target_update_interval = 8, 8
        settings.tau, settings.alpha, settings.min_variance = 0.01, 0.05, 1e-3
        settings.squash_action = 1
        return _result("LunarLanderContinuous-v3", "raw", [_entry("sac", settings)],
                       ["train_lunar_lander_sac_ffi.py"])

    if case in ("ant_ppo", "walker_ppo"):
        observations, actions = (105, 8) if case == "ant_ppo" else (17, 6)
        settings = _default(lib, "ppo", observations, actions)
        settings.action_space = rx.RX_ACTION_CONTINUOUS
        settings.agent.hidden_layers, settings.agent.hidden_size = 1, 256
        settings.agent.gamma = 0.99
        settings.learning_rate, settings.gae_lambda = 1e-4, 0.95
        settings.update_interval, settings.minibatch_size = 512, 64
        settings.epochs = 5 if case == "ant_ppo" else 10
        settings.policy_clip_epsilon, settings.value_clip_range = 0.2, 0.2
        settings.value_loss_coefficient = 0.5
        settings.entropy_coefficient = 0.0 if case == "ant_ppo" else 0.01
        settings.standardize_gae = 1
        settings.min_action, settings.max_action = -1.0, 1.0
        settings.min_variance = 1e-3 if case == "ant_ppo" else 0.01
        return _result("Ant-v5" if case == "ant_ppo" else "Walker2d-v5", "scale0.1",
                       [_entry("ppo", settings)],
                       ["train_ant_ppo_ffi.py" if case == "ant_ppo" else "train_walker2d_ppo_ffi.py"])

    if case == "hopper_sac":
        settings = _default(lib, "sac", 11, 3)
        settings.action_space = rx.RX_ACTION_CONTINUOUS
        settings.agent.hidden_layers, settings.agent.hidden_size = 1, 128
        settings.agent.gamma = 0.99
        settings.actor_learning_rate, settings.critic_learning_rate = 3e-4, 3e-4
        settings.replay_capacity, settings.replay_start_size = 300000, 512
        settings.batch_size, settings.replay_n_steps = 128, 1
        settings.update_interval, settings.target_update_interval = 1, 1
        settings.tau, settings.alpha, settings.min_variance = 0.005, 0.05, 1e-3
        settings.squash_action = 1
        return _result("Hopper-v5", "hopper", [_entry("sac", settings)],
                       ["train_hopper_sac_ffi.py"])

    if case in ("halfcheetah_hybrid", "halfcheetah_sac"):
        args = halfcheetah.parser().parse_args([])
        sac = halfcheetah.configure_sac(lib, args)
        agents = [_entry("sac", sac)]
        if case == "halfcheetah_hybrid":
            ppo = halfcheetah.configure_ppo(lib, args)
            agents = ([_entry("ppo", ppo) for _ in range(args.ppo_workers)] +
                      [_entry("sac", sac) for _ in range(args.sac_workers)])
        return _result("HalfCheetah-v5", "raw", agents, ["train_half_cheetah_hybrid_ffi.py"])

    args = ant_comparison.parser().parse_args([])
    hybrid = case != "ant_sac"
    sac = ant_comparison.configure_sac(
        lib, args, args.hybrid_sac_update_interval if hybrid else args.sac_only_update_interval)
    agents = []
    if hybrid:
        ppo = ant_comparison.configure_ppo(lib, args)
        rnd = ant_comparison.configure_rnd(lib, args) if case == "ant_rnd_shared" else None
        agents.append(_entry("ppo", ppo, rnd, args.rnd_coefficient if rnd is not None else 0.0))
    agents.append(_entry("sac", sac))
    return _result("Ant-v5", "ant_shared", agents, ["train_ant_ppo_rnd_sac_shared_ffi.py"])


def as_dict(value):
    """Serialize ctypes settings, including anonymous SAC-v2 base fields."""
    if isinstance(value, C.Structure):
        result = {}
        anonymous = getattr(type(value), "_anonymous_", ())
        for name, *_ in value._fields_:
            field = as_dict(getattr(value, name))
            if name in anonymous:
                result.update(field)
            else:
                result[name] = field
        return result
    if isinstance(value, dict):
        return {key: as_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_dict(item) for item in value]
    return value
