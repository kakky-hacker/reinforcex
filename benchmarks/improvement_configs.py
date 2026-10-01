"""Load explicit study settings, independent of changing example defaults."""
from __future__ import annotations

import copy
import ctypes as C
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import reinforcex_ffi as rx

BASELINE = ROOT / "reports/core_improvements_20261001/baseline/configurations.json"
REWARD_MODES = {"cartpole", "scale0.1", "hopper", "ant_shared", "raw"}
VALIDATION_SEED = 1_100_000
CONFIRMATION_SEED = 1_200_000
EVALUATION_SEED_BLOCK = 100_000


def as_dict(value):
    if isinstance(value, C.Structure):
        result = {}
        for name, *_ in value._fields_:
            field = as_dict(getattr(value, name))
            if name in getattr(type(value), "_anonymous_", ()):
                result.update(field)
            else:
                result[name] = field
        return result
    if isinstance(value, dict):
        return {key: as_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_dict(item) for item in value]
    return value


def field_types(struct):
    result = {name: typ for name, typ, *_ in struct._fields_}
    for name in getattr(type(struct), "_anonymous_", ()):
        result.update(field_types(getattr(struct, name)))
    return result


def fill(struct, values):
    if not isinstance(values, dict):
        raise ValueError(f"{type(struct).__name__} settings must be an object")
    fields = field_types(struct)
    for key, value in values.items():
        if key not in fields:
            raise ValueError(f"unknown {type(struct).__name__} field: {key}")
        if isinstance(getattr(struct, key), C.Structure):
            fill(getattr(struct, key), value)
        else:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            converted = fields[key](value).value
            if converted != value:
                raise ValueError(f"{key} cannot be represented by {fields[key].__name__}: {value}")
            setattr(struct, key, value)
    return struct


def merge(destination, changes):
    if not isinstance(destination, dict) or not isinstance(changes, dict):
        raise ValueError("nested overrides must be objects")
    for key, value in changes.items():
        if key not in destination:
            raise ValueError(f"unknown setting: {key}")
        if isinstance(value, dict):
            merge(destination[key], value)
        elif isinstance(destination[key], dict):
            raise ValueError(f"{key} must remain an object")
        else:
            destination[key] = value


def configuration(case, override_path=None):
    diagnostic_ppo = case == "halfcheetah_ppo_diagnostic"
    source_case = "halfcheetah_hybrid" if diagnostic_ppo else case
    result = copy.deepcopy(json.loads(BASELINE.read_text())[source_case])
    if diagnostic_ppo:
        # Same PPO settings and per-learner budget; isolates learner diagnostics.
        # Success here is not evidence of success in the four-worker hybrid.
        result["agents"] = [next(a for a in result["agents"] if a["algorithm"] == "ppo")]
        result["shared_replay"] = False
        result["diagnostic_scope"] = "single PPO extracted from halfcheetah_hybrid; no SAC/replay sharing"
    overrides = {} if override_path is None else json.loads(Path(override_path).read_text())
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object")
    unknown = set(overrides) - {"reward_mode", "algorithms", "ppo_v2", "normalization", "learning_rate_schedule"}
    if unknown:
        raise ValueError(f"unknown overrides: {unknown}")
    if "reward_mode" in overrides:
        result["reward_mode"] = overrides["reward_mode"]
    if result["reward_mode"] not in REWARD_MODES:
        raise ValueError(f"unknown reward mode: {result['reward_mode']}")
    present = {a["algorithm"] for a in result["agents"]}
    if not isinstance(overrides.get("algorithms", {}), dict):
        raise ValueError("algorithms overrides must be an object")
    if set(overrides.get("algorithms", {})) - present:
        raise ValueError("override refers to algorithm absent from case")
    if "ppo_v2" in overrides and "ppo" not in present:
        raise ValueError("ppo_v2 override refers to algorithm absent from case")
    normalization = overrides.get("normalization", {})
    if not isinstance(normalization, dict) or set(normalization) - (present & {"ppo"}):
        raise ValueError("normalization is currently supported for present PPO learners only")
    schedules = overrides.get("learning_rate_schedule", {})
    if not isinstance(schedules, dict) or set(schedules) - (present & {"ppo", "dqn"}):
        raise ValueError("learning rate schedules support present DQN/PPO learners only")
    for spec in result["agents"]:
        algo = spec["algorithm"]
        values = spec["config"]
        merge(values, overrides.get("algorithms", {}).get(algo, {}))
        typ = {"ppo": rx.RxPpoConfig, "dqn": rx.RxDqnConfig, "sac": rx.RxSacConfigV2}[algo]
        struct = typ()
        if algo == "ppo" and "ppo_v2" in overrides:
            struct = rx.RxPpoConfigV2()
            fill(struct, {"model": 1, "activation": 0, "initial_log_std": 0.,
                          "adam_epsilon": 1e-5, "target_kl": 0.})
            extension_fields = set(field_types(struct)) - set(field_types(rx.RxPpoConfig()))
            extension_fields -= set(getattr(type(struct), "_anonymous_", ()))
            if not isinstance(overrides["ppo_v2"], dict) or set(overrides["ppo_v2"]) - extension_fields:
                raise ValueError("ppo_v2 accepts extension fields only; use algorithms.ppo for base settings")
            fill(struct, overrides["ppo_v2"])
        spec["config"] = fill(struct, values)
        if spec["rnd_config"] is not None:
            spec["rnd_config"] = fill(rx.RxRndConfig(), spec["rnd_config"])
        if algo in schedules:
            options = schedules[algo]
            if not isinstance(options, dict) or set(options) != {"kind", "final_fraction"} or options["kind"] != "linear":
                raise ValueError("learning rate schedule needs kind=linear and final_fraction")
            fraction = options["final_fraction"]
            if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError("final_fraction must be finite and in (0, 1]")
            spec["learning_rate_schedule"] = dict(options)
        if algo in normalization:
            options = {"normalize_observations": True, "normalize_rewards": True,
                       "clip_observations": 10.0, "clip_rewards": 10.0}
            changes = normalization[algo]
            if not isinstance(changes, dict) or set(changes) - (set(options) | {"preserve_replay_inputs"}):
                raise ValueError("unknown normalization option")
            options.update(changes)
            for flag in ("normalize_observations", "normalize_rewards", "preserve_replay_inputs"):
                if type(options.get(flag, False)) is not bool:
                    raise ValueError(f"{flag} must be boolean")
            if result["shared_replay"] and not options.get("preserve_replay_inputs", False):
                raise ValueError("normalization before FFI would mix transformed and raw transitions in shared replay")
            for field in ("clip_observations", "clip_rewards"):
                value = options[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field} must be finite and positive")
            if spec["rnd_config"] is not None:
                raise ValueError("normalization with RND is not part of this study protocol")
            spec["normalization"] = options
    return result


def effective_configuration(case_name, override_path=None, workers=None, share_rnd=False):
    case = configuration(case_name, override_path)
    specs = case["agents"]
    if workers is not None:
        if type(workers) is not int or workers <= 0 or len(specs) != 1:
            raise ValueError("workers must be positive and can only replicate a single-agent case")
        specs = [copy.deepcopy(specs[0]) for _ in range(workers)]
    if not specs:
        raise ValueError("a run must contain at least one agent")
    if share_rnd and not all(spec.get("rnd_config") is not None for spec in specs):
        raise ValueError("shared RND requires an RND configuration for every worker")
    replay_settings = {(s["config"].replay_capacity, s["config"].replay_n_steps)
                       for s in specs if s["algorithm"] != "ppo"}
    if case["shared_replay"] and len(replay_settings) != 1:
        raise ValueError("shared replay requires matching replay capacity and n-step settings")
    return case, specs


def study_request(case_name, seed, steps, case, specs, *, stage="development", checkpoints=10,
                  eval_episodes=100, validation_episodes=10, share_rnd=False, serial_workers=False,
                  final_seed=None):
    """Serializable identity of a requested measurement, shared by runner and resume checks."""
    for name, value in (("seed", seed), ("steps", steps), ("checkpoints", checkpoints),
                        ("eval_episodes", eval_episodes), ("validation_episodes", validation_episodes)):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"invalid {name}: {value}")
    if stage not in {"development", "confirmation"}:
        raise ValueError(f"unknown study stage: {stage}")
    if final_seed is not None and (
            stage != "confirmation" or type(final_seed) is not int
            or final_seed < CONFIRMATION_SEED or final_seed % EVALUATION_SEED_BLOCK):
        raise ValueError("final_seed must start a reserved confirmation block (1200000, 1300000, ...)")
    if final_seed is None:
        final_seed = CONFIRMATION_SEED if stage == "confirmation" else VALIDATION_SEED
    if min(eval_episodes, validation_episodes) < 2:
        raise ValueError("evaluation and validation need at least two episodes")
    if max(eval_episodes, validation_episodes) > EVALUATION_SEED_BLOCK:
        raise ValueError("evaluation episode count would leave its reserved seed block")
    if steps % (len(specs) * checkpoints):
        raise ValueError("total steps must divide equally over workers and checkpoints")
    counts = {}
    for spec in specs:
        ordinal = counts.get(spec["algorithm"], 0)
        counts[spec["algorithm"]] = ordinal + 1
        environment_seed = seed + ordinal * 10_000
        if (VALIDATION_SEED <= environment_seed < CONFIRMATION_SEED + EVALUATION_SEED_BLOCK
                or final_seed <= environment_seed < final_seed + EVALUATION_SEED_BLOCK):
            raise ValueError("training environment seed overlaps a reserved evaluation seed block")
    return {"case": case_name, "seed": seed, "steps": steps, "worker_count": len(specs),
            "configuration": as_dict(case), "effective_agents": as_dict(specs),
            "stage": stage, "checkpoints": checkpoints, "eval_episodes": eval_episodes,
            "validation_episodes": validation_episodes, "share_rnd": bool(share_rnd),
            "serial_workers": bool(serial_workers), "validation_seed": VALIDATION_SEED,
            "final_seed": final_seed}
