"""Independent ctypes tests for the opt-in PPO V2 ABI (standard library only).

Run with the new library and its matching LibTorch loader environment::

    python3 ffi/tests/test_ppo_v2.py --library /path/to/libreinforcex.dylib

Each native probe runs in a separate process: an ABI error or native abort is a
test failure, not a crash of the complete suite. Synthetic transitions exercise
training mechanics; they do not establish an environment learning benchmark.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


# These fixtures deliberately do not import the evolving Python wrapper/header.
class LegacyAgentConfig(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ("obs_size", "action_size", "hidden_layers", "hidden_size")] + [
        ("gamma", C.c_double),
    ]


class LegacyPpoConfig(C.Structure):
    _fields_ = [
        ("agent", LegacyAgentConfig), ("action_space", C.c_uint32),
        ("learning_rate", C.c_double), ("gae_lambda", C.c_double),
        ("update_interval", C.c_uint64), ("epochs", C.c_uint64),
        ("minibatch_size", C.c_uint64), ("policy_clip_epsilon", C.c_double),
        ("value_clip_range", C.c_double), ("value_loss_coefficient", C.c_double),
        ("entropy_coefficient", C.c_double), ("standardize_gae", C.c_uint32),
        ("min_action", C.c_double), ("max_action", C.c_double),
        ("min_variance", C.c_double),
    ]


class PpoConfigV2(C.Structure):
    _fields_ = [("base", LegacyPpoConfig), ("model", C.c_uint32),
                ("activation", C.c_uint32), ("initial_log_std", C.c_double),
                ("adam_epsilon", C.c_double), ("target_kl", C.c_double)]


class ReplayConfig(C.Structure):
    _fields_ = [("capacity", C.c_uint64), ("n_steps", C.c_uint64)]


class RndConfig(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ("obs_size", "feature_size", "hidden_layers", "hidden_size")] + [
        ("learning_rate", C.c_double), ("update_interval", C.c_uint64),
    ]


class Statistic(C.Structure):
    _fields_ = [("name", C.c_char * 64), ("value", C.c_double)]


def guarded(config_type):
    class Guard(C.Structure):
        _fields_ = [("before", C.c_ubyte * 16), ("config", config_type),
                    ("after", C.c_ubyte * 16)]
    result = Guard()
    C.memset(C.byref(result), 0xA5, C.sizeof(result))
    return result


DLL_DIRECTORY_HANDLES = []
OPTIONS = None


def load_library(path):
    if os.name == "nt":
        C.windll.kernel32.SetErrorMode(0x8003)
        directories = [Path(path).resolve().parent]
        if os.environ.get("LIBTORCH"):
            directories.append(Path(os.environ["LIBTORCH"]) / "lib")
        directories.extend(Path(item) for item in
                           os.environ.get("REINFORCEX_DLL_DIRS", "").split(os.pathsep) if item)
        for directory in dict.fromkeys(item.resolve() for item in directories):
            if directory.is_dir():
                DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
    lib = C.CDLL(str(Path(path).resolve()))
    u, f, p, s = C.c_uint64, C.c_float, C.POINTER, C.c_char_p
    signatures = {
        "rx_manual_seed": ([C.c_int64], C.c_int32),
        "rx_ppo_config_default": ([p(LegacyPpoConfig), u, u], C.c_int32),
        "rx_ppo_config_default_v2": ([p(PpoConfigV2), u, u], C.c_int32),
        "rx_ppo_create_v2": ([p(PpoConfigV2), u, u, C.c_double, s, s, p(u)], C.c_int32),
        "rx_rnd_config_default": ([p(RndConfig), u], C.c_int32),
        "rx_rnd_create": ([p(RndConfig), p(u)], C.c_int32),
        "rx_replay_buffer_create": ([p(ReplayConfig), p(u)], C.c_int32),
        "rx_replay_buffer_len": ([u, p(u)], C.c_int32),
        "rx_agent_act": ([u, p(f), u, p(f), u], C.c_int64),
        "rx_agent_act_and_train": ([u, p(f), u, f, p(f), u], C.c_int64),
        "rx_agent_stop_episode_with_terminal": ([u, p(f), u, f, C.c_uint32], C.c_int32),
        "rx_agent_statistics_len": ([u, p(u)], C.c_int32),
        "rx_agent_statistics": ([u, p(Statistic), u], C.c_int64),
    }
    for name in ("rx_agent_destroy", "rx_agent_save", "rx_agent_load",
                 "rx_rnd_destroy", "rx_replay_buffer_destroy"):
        signatures[name] = ([u], C.c_int32)
    for rnd in (False, True):
        for replay in (False, True):
            for paths in (False, True):
                parts = (["rnd"] if rnd else []) + (["replay"] if replay else [])
                parts += ["paths"] if paths else []
                name = "rx_ppo_create" + ("_with_" + "_and_".join(parts) if parts else "")
                args = [p(LegacyPpoConfig)] + ([u] if rnd else []) + ([u] if replay else [])
                args += [C.c_double] if rnd else []
                args += [s, s] if paths else []
                signatures[name] = (args + [p(u)], C.c_int32)
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib


def config_v2(check, lib, action_space=0, activation=0, model=1):
    config = PpoConfigV2()
    check.assertEqual(lib.rx_ppo_config_default_v2(C.byref(config), 4, 2), 0)
    config.base.action_space = action_space
    config.base.agent.hidden_layers = 0
    config.base.agent.hidden_size = 16
    config.base.update_interval = 16
    config.base.epochs = 2
    config.base.minibatch_size = 4
    config.base.value_clip_range = 0.0
    config.model, config.activation = model, activation
    return config


def create(check, lib, config, *, rnd=0, replay=0, save=None, load=None):
    handle = C.c_uint64(987654321)
    status = lib.rx_ppo_create_v2(C.byref(config), rnd, replay, 0.25,
                                 os.fsencode(save) if save else None,
                                 os.fsencode(load) if load else None, C.byref(handle))
    check.assertEqual(status, 0)
    check.assertNotEqual(handle.value, 0)
    return handle.value


def statistics(check, lib, handle):
    size = C.c_uint64()
    check.assertEqual(lib.rx_agent_statistics_len(handle, C.byref(size)), 0)
    values = (Statistic * size.value)()
    check.assertEqual(lib.rx_agent_statistics(handle, values, size), size.value)
    result = {entry.name.decode(): entry.value for entry in values}
    check.assertTrue(all(math.isfinite(value) for value in result.values()), result)
    return result


def observation(index):
    return (C.c_float * 4)(math.sin(index * 0.3), math.cos(index * 0.2),
                           ((index % 7) - 3) * 0.1, index % 2 * 0.25)


def action(check, lib, handle, config, index, reward=None):
    output_size = 1 if config.base.action_space == 0 else 2
    output = (C.c_float * output_size)()
    obs = observation(index)
    if reward is None:
        written = lib.rx_agent_act(handle, obs, 4, output, output_size)
    else:
        written = lib.rx_agent_act_and_train(handle, obs, 4, reward, output, output_size)
    check.assertEqual(written, output_size)
    check.assertTrue(all(math.isfinite(value) for value in output))
    if config.base.action_space == 0:
        check.assertIn(output[0], (0.0, 1.0))
    else:
        check.assertTrue(all(config.base.min_action <= value <= config.base.max_action
                             for value in output), list(output))
    return list(output)


def train(check, lib, handle, config, episodes=4):
    # Exactly eight transitions per episode, with both terminal and time-limit endings.
    for episode in range(episodes):
        for step in range(8):
            action(check, lib, handle, config, episode * 9 + step,
                   0.0 if step == 0 else (1.0 if step % 2 else -0.25))
        check.assertEqual(lib.rx_agent_stop_episode_with_terminal(
            handle, observation(episode * 9 + 8), 4, 0.5, episode % 2), 0)
    stats = statistics(check, lib, handle)
    check.assertGreater(stats["updates"], 0)
    check.assertGreater(stats["optimizer_steps"], 0)
    return stats


def resources(check, lib, obs_size=4):
    rnd_config = RndConfig()
    check.assertEqual(lib.rx_rnd_config_default(C.byref(rnd_config), obs_size), 0)
    rnd_config.feature_size, rnd_config.hidden_layers, rnd_config.hidden_size = 8, 0, 16
    rnd_config.update_interval = 4
    rnd = C.c_uint64()
    check.assertEqual(lib.rx_rnd_create(C.byref(rnd_config), C.byref(rnd)), 0)
    replay = C.c_uint64()
    check.assertEqual(lib.rx_replay_buffer_create(C.byref(ReplayConfig(128, 3)), C.byref(replay)), 0)
    return rnd.value, replay.value


def check_guards(check, value):
    check.assertEqual(bytes(value.before), b"\xA5" * 16)
    check.assertEqual(bytes(value.after), b"\xA5" * 16)


def probe_defaults(check, lib):
    old, new = guarded(LegacyPpoConfig), guarded(PpoConfigV2)
    check.assertEqual(lib.rx_ppo_config_default(C.byref(old.config), 4, 2), 0)
    check.assertEqual(lib.rx_ppo_config_default_v2(C.byref(new.config), 4, 2), 0)
    check_guards(check, old)
    check_guards(check, new)
    for name, _ in LegacyPpoConfig._fields_:
        if name != "agent":
            check.assertEqual(getattr(old.config, name), getattr(new.config.base, name))
    for name, _ in LegacyAgentConfig._fields_:
        check.assertEqual(getattr(old.config.agent, name), getattr(new.config.base.agent, name))
    check.assertEqual((new.config.model, new.config.activation), (1, 0))
    check.assertEqual(new.config.initial_log_std, 0)
    check.assertEqual(new.config.adam_epsilon, 1e-5)
    check.assertEqual(new.config.target_kl, 0)
    check.assertEqual(lib.rx_ppo_config_default_v2(None, 4, 2), -1)
    # A failed default must leave the caller's output intact.
    before = bytes(new)
    for obs_size, action_size in ((0, 2), (4, 0)):
        check.assertEqual(lib.rx_ppo_config_default_v2(C.byref(new.config), obs_size, action_size), -2)
        check.assertEqual(bytes(new), before)
    return {"legacy_size": C.sizeof(LegacyPpoConfig), "v2_size": C.sizeof(PpoConfigV2)}


def probe_invalid(check, lib):
    cases = [("model", 2), ("model", 2**32 - 1), ("activation", 2),
             ("initial_log_std", -1000), ("initial_log_std", 1000),
             ("adam_epsilon", 0), ("adam_epsilon", -1), ("target_kl", -1),
             ("base.action_space", 2), ("base.standardize_gae", 2),
             ("base.update_interval", 0), ("base.minibatch_size", 17),
             ("base.agent.obs_size", 0)]
    for field in ("initial_log_std", "adam_epsilon", "target_kl", "base.learning_rate",
                  "base.gae_lambda", "base.value_clip_range"):
        cases.extend((field, value) for value in (math.nan, math.inf, -math.inf))
    for field, value in cases:
        config = config_v2(check, lib)
        owner = config
        names = field.split(".")
        for name in names[:-1]:
            owner = getattr(owner, name)
        setattr(owner, names[-1], value)
        handle = C.c_uint64(123)
        check.assertEqual(lib.rx_ppo_create_v2(C.byref(config), 0, 0, 0, None, None,
                                              C.byref(handle)), -2, (field, value))
        check.assertEqual(handle.value, 0, (field, value))
    for field, value in (("initial_log_std", -10), ("min_variance", 1e-300),
                         ("min_variance", 1e300)):
        config = config_v2(check, lib, action_space=1)
        setattr(config if field == "initial_log_std" else config.base, field, value)
        handle = C.c_uint64(123)
        check.assertEqual(lib.rx_ppo_create_v2(C.byref(config), 0, 0, 0, None, None,
                                              C.byref(handle)), -2, (field, value))
        check.assertEqual(handle.value, 0)
    config = config_v2(check, lib)
    for rnd, replay, coefficient, save, expected in (
        (2**64 - 1, 0, 0, None, -3), (0, 2**64 - 1, 0, None, -3),
        (0, 0, math.nan, None, -2), (0, 0, math.inf, None, -2),
        (0, 0, 0, b"\xff", -2),
    ):
        handle = C.c_uint64(123)
        check.assertEqual(lib.rx_ppo_create_v2(C.byref(config), rnd, replay, coefficient,
                                              save, None, C.byref(handle)), expected)
        check.assertEqual(handle.value, 0)
    handle = C.c_uint64(123)
    check.assertEqual(lib.rx_ppo_create_v2(None, 0, 0, 0, None, None, C.byref(handle)), -1)
    check.assertEqual(handle.value, 0)
    check.assertEqual(lib.rx_ppo_create_v2(C.byref(config), 0, 0, 0, None, None, None), -1)
    rnd, replay = resources(check, lib, obs_size=3)
    try:
        handle.value = 123
        check.assertEqual(lib.rx_ppo_create_v2(C.byref(config), rnd, replay, 1, None, None,
                                              C.byref(handle)), -2)
        check.assertEqual(handle.value, 0)
    finally:
        check.assertEqual(lib.rx_rnd_destroy(rnd), 0)
        check.assertEqual(lib.rx_replay_buffer_destroy(replay), 0)
    # Registry still usable after errors; zero optional handles really mean absent.
    agent = create(check, lib, config)
    check.assertEqual(lib.rx_agent_destroy(agent), 0)
    return {"invalid_config_cases": len(cases) + 3}


def probe_legacy(check, lib):
    config = config_v2(check, lib, model=0)
    rnd, replay = resources(check, lib)
    exercised = []
    try:
        for use_rnd in (False, True):
            for use_replay in (False, True):
                for paths in (False, True):
                    parts = (["rnd"] if use_rnd else []) + (["replay"] if use_replay else [])
                    parts += ["paths"] if paths else []
                    name = "rx_ppo_create" + ("_with_" + "_and_".join(parts) if parts else "")
                    args = [C.byref(config.base)]
                    args += [rnd] if use_rnd else []
                    args += [replay] if use_replay else []
                    args += [0.25] if use_rnd else []
                    args += [None, None] if paths else []
                    handle = C.c_uint64()
                    check.assertEqual(getattr(lib, name)(*args, C.byref(handle)), 0, name)
                    try:
                        action(check, lib, handle.value, config, 1)
                    finally:
                        check.assertEqual(lib.rx_agent_destroy(handle), 0)
                    exercised.append(name)
        for action_space in (0, 1):
            config.base.action_space = action_space
            agent = create(check, lib, config)
            try:
                train(check, lib, agent, config)
            finally:
                check.assertEqual(lib.rx_agent_destroy(agent), 0)
    finally:
        check.assertEqual(lib.rx_rnd_destroy(rnd), 0)
        check.assertEqual(lib.rx_replay_buffer_destroy(replay), 0)
    return {"legacy_symbols": exercised, "v2_legacy_action_spaces": 2}


def probe_roundtrip(check, lib, action_space, activation):
    config = config_v2(check, lib, action_space=action_space, activation=activation)
    with tempfile.TemporaryDirectory(prefix="reinforcex-ppo-v2-") as directory:
        path = Path(directory) / "model.ot"
        agent = create(check, lib, config, save=path)
        try:
            stats = train(check, lib, agent, config)
            check.assertEqual(stats["updates"], 2)
            check.assertEqual(stats["optimizer_steps"], 16)
            expected = [action(check, lib, agent, config, index) for index in range(12)]
            check.assertEqual(statistics(check, lib, agent), stats, "evaluation changed training state")
            check.assertEqual(lib.rx_agent_save(agent), 0)
            check.assertGreater(path.stat().st_size, 0)
        finally:
            check.assertEqual(lib.rx_agent_destroy(agent), 0)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        restored = create(check, lib, config, load=path)
        try:
            check.assertEqual(statistics(check, lib, restored)["updates"], 0)
            check.assertEqual([action(check, lib, restored, config, index) for index in range(12)], expected)
            train(check, lib, restored, config)
            check.assertEqual(lib.rx_agent_load(restored), 0)
            check.assertEqual([action(check, lib, restored, config, index) for index in range(12)], expected)
            check.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        finally:
            check.assertEqual(lib.rx_agent_destroy(restored), 0)
    return {"action_space": action_space, "activation": activation, "statistics": stats,
            "matching_observations": 12, "checkpoint_sha256": digest}


def probe_shared(check, lib):
    config = config_v2(check, lib)
    rnd, replay = resources(check, lib)
    agents = []
    try:
        agents = [create(check, lib, config, rnd=rnd, replay=replay) for _ in range(2)]
        stats = [train(check, lib, agent, config) for agent in agents]
        for item in stats:
            check.assertGreater(item["intrinsic_reward_mean"], 0)
            check.assertEqual(item["curiosity_coefficient"], 0.25)
            check.assertEqual(item["updates"], 2)
        size = C.c_uint64()
        check.assertEqual(lib.rx_replay_buffer_len(replay, C.byref(size)), 0)
        check.assertEqual(size.value, 64)
        # The already-bound buffer rejects a mismatched observation contract.
        mismatch = config_v2(check, lib)
        mismatch.base.agent.obs_size = 3
        handle = C.c_uint64(123)
        check.assertEqual(lib.rx_ppo_create_v2(C.byref(mismatch), 0, replay, 0, None, None,
                                              C.byref(handle)), -2)
        check.assertEqual(handle.value, 0)
        # Registry ownership may be dropped while the agents retain their Arcs.
        check.assertEqual(lib.rx_rnd_destroy(rnd), 0)
        rnd = 0
        check.assertEqual(lib.rx_replay_buffer_destroy(replay), 0)
        replay = 0
        for agent in agents:
            train(check, lib, agent, config, episodes=2)
    finally:
        for agent in agents:
            check.assertEqual(lib.rx_agent_destroy(agent), 0)
        if rnd:
            check.assertEqual(lib.rx_rnd_destroy(rnd), 0)
        if replay:
            check.assertEqual(lib.rx_replay_buffer_destroy(replay), 0)
    return {"agents": 2, "shared_transitions": 64, "statistics": stats}


def probe_mismatch(check, lib):
    with tempfile.TemporaryDirectory(prefix="reinforcex-ppo-mismatch-") as directory:
        configs = {"legacy": config_v2(check, lib, model=0),
                   "tanh": config_v2(check, lib),
                   "relu": config_v2(check, lib, activation=1),
                   "continuous": config_v2(check, lib, action_space=1)}
        paths, hashes = {}, {}
        for name, config in configs.items():
            paths[name] = Path(directory) / (name + ".ot")
            agent = create(check, lib, config, save=paths[name])
            try:
                check.assertEqual(lib.rx_agent_save(agent), 0)
            finally:
                check.assertEqual(lib.rx_agent_destroy(agent), 0)
            hashes[name] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        pairs = [(destination, source) for destination, source in
                 (("legacy", "tanh"), ("tanh", "legacy"), ("tanh", "relu"),
                  ("relu", "tanh"), ("tanh", "continuous"), ("continuous", "tanh"))]
        for destination, source in pairs:
            handle = C.c_uint64(123)
            status = lib.rx_ppo_create_v2(C.byref(configs[destination]), 0, 0, 0, None,
                                         os.fsencode(paths[source]), C.byref(handle))
            check.assertLess(status, 0, (destination, source, "silently accepted checkpoint"))
            check.assertEqual(handle.value, 0)
        config = config_v2(check, lib, action_space=1)
        config.base.max_action = 2
        handle = C.c_uint64(123)
        check.assertLess(lib.rx_ppo_create_v2(C.byref(config), 0, 0, 0, None,
                                             os.fsencode(paths["continuous"]), C.byref(handle)), 0)
        check.assertEqual(handle.value, 0)
        # Explicit reload must report the same incompatibility. The panic may
        # poison this agent's mutex; only destruction is required after failure.
        swapped = Path(directory) / "swapped.ot"
        shutil.copyfile(paths["tanh"], swapped)
        agent = create(check, lib, configs["tanh"], load=swapped)
        try:
            shutil.copyfile(paths["relu"], swapped)
            check.assertLess(lib.rx_agent_load(agent), 0)
        finally:
            check.assertEqual(lib.rx_agent_destroy(agent), 0)
        for name in paths:
            check.assertEqual(hashlib.sha256(paths[name].read_bytes()).hexdigest(), hashes[name])
        agent = create(check, lib, configs["tanh"], load=paths["tanh"])
        check.assertEqual(lib.rx_agent_destroy(agent), 0)
    return {"constructor_mismatches_rejected": len(pairs) + 1, "explicit_reload_rejected": True}


def child_probe(name, library):
    check = unittest.TestCase()
    lib = load_library(library)
    check.assertEqual(lib.rx_manual_seed(731), 0)
    if name.startswith("roundtrip_"):
        _, action_space, activation = name.split("_")
        return probe_roundtrip(check, lib, int(action_space), int(activation))
    return {"defaults": probe_defaults, "invalid": probe_invalid, "legacy": probe_legacy,
            "shared": probe_shared, "mismatch": probe_mismatch}[name](check, lib)


class PpoV2Regressions(unittest.TestCase):
    def probe(self, name):
        library = OPTIONS.library if OPTIONS else os.environ.get("REINFORCEX_LIB")
        if not library:
            self.skipTest("set REINFORCEX_LIB or pass --library to run native probes")
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--library", str(Path(library).resolve()),
             "--probe", name], env=environment, capture_output=True, text=True, timeout=60,
            creationflags=0x08000000 if os.name == "nt" else 0)
        self.assertEqual(result.returncode, 0,
                         f"{name} child exited {result.returncode}:\n{result.stdout}\n{result.stderr}")
        lines = [line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, result.stdout)
        return json.loads(lines[0])

    def test_independent_legacy_layout_and_v2_offsets(self):
        self.assertEqual(C.sizeof(LegacyAgentConfig), 40)
        self.assertEqual(C.sizeof(LegacyPpoConfig), 152)
        self.assertEqual(C.sizeof(PpoConfigV2), 184)
        self.assertEqual({name: getattr(LegacyPpoConfig, name).offset for name in
                          ("action_space", "learning_rate", "standardize_gae", "min_variance")},
                         {"action_space": 40, "learning_rate": 48, "standardize_gae": 120,
                          "min_variance": 144})
        self.assertEqual({name: getattr(PpoConfigV2, name).offset for name, _ in PpoConfigV2._fields_},
                         {"base": 0, "model": 152, "activation": 156, "initial_log_std": 160,
                          "adam_epsilon": 168, "target_kl": 176})

    def test_defaults_preserve_legacy_and_buffer_guards(self):
        self.assertEqual(self.probe("defaults")["v2_size"], 184)

    def test_invalid_config_and_handles_clear_output(self):
        self.assertGreaterEqual(self.probe("invalid")["invalid_config_cases"], 30)

    def test_all_legacy_constructors_and_v2_legacy_mode(self):
        self.assertEqual(len(self.probe("legacy")["legacy_symbols"]), 8)

    def test_separate_discrete_tanh_training_and_checkpoint(self):
        self.probe("roundtrip_0_0")

    def test_separate_discrete_relu_training_and_checkpoint(self):
        self.probe("roundtrip_0_1")

    def test_separate_continuous_tanh_training_and_checkpoint(self):
        self.probe("roundtrip_1_0")

    def test_separate_continuous_relu_training_and_checkpoint(self):
        self.probe("roundtrip_1_1")

    def test_rnd_and_shared_replay_simultaneously(self):
        self.assertEqual(self.probe("shared")["shared_transitions"], 64)

    def test_checkpoint_model_activation_and_bounds_mismatch_rejected(self):
        self.assertEqual(self.probe("mismatch")["constructor_mismatches_rejected"], 7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=os.environ.get("REINFORCEX_LIB"),
                        help="new ReinforceX FFI library (or set REINFORCEX_LIB)")
    parser.add_argument("--probe", help=argparse.SUPPRESS)
    OPTIONS = parser.parse_args()
    if OPTIONS.probe:
        if not OPTIONS.library:
            parser.error("--probe requires --library")
        print("RESULT " + json.dumps(child_probe(OPTIONS.probe, OPTIONS.library)), flush=True)
    else:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
