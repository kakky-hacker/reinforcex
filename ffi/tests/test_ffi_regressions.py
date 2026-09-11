"""DLL-level ABI and CUDA-loader regressions, using only the Python standard library.

Example (load the project's LibTorch environment first)::

    py ffi/tests/test_ffi_regressions.py --library target/debug/reinforcex.dll --expect-cuda-loader

Every native call runs in a child process so an ABI or loader regression produces
a failed test instead of terminating the test runner.
"""

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


# Deliberately independent of examples/reinforcex_ffi.py and the current header:
# changing the public ABI must not silently change the old-client fixture.
class LegacyAgentConfig(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ("obs_size", "action_size", "hidden_layers", "hidden_size")] + [
        ("gamma", C.c_double),
    ]


class LegacySacConfig(C.Structure):
    _fields_ = [
        ("agent", LegacyAgentConfig),
        ("action_space", C.c_uint32),
        ("actor_learning_rate", C.c_double),
        ("critic_learning_rate", C.c_double),
        ("replay_capacity", C.c_uint64),
        ("replay_start_size", C.c_uint64),
        ("batch_size", C.c_uint64),
        ("replay_n_steps", C.c_uint64),
        ("update_interval", C.c_uint64),
        ("target_update_interval", C.c_uint64),
        ("tau", C.c_double),
        ("alpha", C.c_double),
        ("min_variance", C.c_double),
        ("squash_action", C.c_uint32),
    ]


class SacConfigV2(C.Structure):
    _fields_ = [("base", LegacySacConfig), ("discrete_target_entropy_ratio", C.c_double)]


class GuardedLegacyConfig(C.Structure):
    _fields_ = [
        ("before", C.c_ubyte * 16),
        ("config", LegacySacConfig),
        ("after", C.c_ubyte * 16),
    ]


DLL_DIRECTORY_HANDLES = []
OPTIONS = None


def load_library(path):
    if os.name == "nt":
        # Suppress Windows error dialogs if a regression crashes the child.
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
    signatures = {
        "rx_cuda_is_available": ([], C.c_uint32),
        "rx_sac_config_default": ([C.POINTER(LegacySacConfig), C.c_uint64, C.c_uint64], C.c_int32),
        "rx_sac_config_default_v2": ([C.POINTER(SacConfigV2), C.c_uint64, C.c_uint64], C.c_int32),
        "rx_sac_create": ([C.POINTER(LegacySacConfig), C.POINTER(C.c_uint64)], C.c_int32),
        "rx_sac_create_v2": ([C.POINTER(SacConfigV2), C.POINTER(C.c_uint64)], C.c_int32),
        "rx_agent_act": ([C.c_uint64, C.POINTER(C.c_float), C.c_uint64,
                          C.POINTER(C.c_float), C.c_uint64], C.c_int64),
        "rx_agent_destroy": ([C.c_uint64], C.c_int32),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    return lib


def exercise_agent(check, lib, config, create, action_space):
    base = config.base if isinstance(config, SacConfigV2) else config
    base.action_space = action_space
    base.agent.hidden_layers = 1
    base.agent.hidden_size = 16
    base.replay_capacity = 32
    base.replay_start_size = 4
    base.batch_size = 4
    agent = C.c_uint64()
    check.assertEqual(create(C.byref(config), C.byref(agent)), 0)
    check.assertNotEqual(agent.value, 0)
    try:
        observation = (C.c_float * 4)(0.1, 0.2, -0.1, 0.0)
        action = (C.c_float * 2)()
        expected = 1 if action_space == 0 else 2
        check.assertEqual(lib.rx_agent_act(agent, observation, 4, action, 2), expected)
        check.assertTrue(all(math.isfinite(action[i]) for i in range(expected)))
        if action_space == 0:
            check.assertIn(action[0], (0.0, 1.0))
        else:
            check.assertTrue(all(-1.0 <= value <= 1.0 for value in action))
    finally:
        check.assertEqual(lib.rx_agent_destroy(agent), 0)


def child_probe(name, library):
    check = unittest.TestCase()
    lib = load_library(library)
    if name == "legacy":
        check.assertEqual(C.sizeof(LegacySacConfig), 144)
        check.assertEqual(LegacySacConfig.min_variance.offset, 128)
        check.assertEqual(LegacySacConfig.squash_action.offset, 136)
        guarded = GuardedLegacyConfig()
        C.memset(C.byref(guarded), 0xA5, C.sizeof(guarded))
        check.assertEqual(lib.rx_sac_config_default(C.byref(guarded.config), 4, 2), 0)
        check.assertEqual(bytes(guarded.before), b"\xA5" * 16)
        check.assertEqual(bytes(guarded.after), b"\xA5" * 16)
        check.assertEqual(guarded.config.agent.obs_size, 4)
        check.assertEqual(guarded.config.agent.action_size, 2)
        check.assertAlmostEqual(guarded.config.min_variance, 0.001)
        check.assertEqual(guarded.config.squash_action, 1)
        exercise_agent(check, lib, guarded.config, lib.rx_sac_create, 1)
        check.assertEqual(bytes(guarded.after), b"\xA5" * 16)
        return {"size": C.sizeof(LegacySacConfig)}
    if name == "v2":
        check.assertEqual(C.sizeof(SacConfigV2), 152)
        check.assertEqual(SacConfigV2.discrete_target_entropy_ratio.offset, 144)
        config = SacConfigV2()
        check.assertEqual(lib.rx_sac_config_default_v2(C.byref(config), 4, 2), 0)
        check.assertAlmostEqual(config.discrete_target_entropy_ratio, 0.98)
        for ratio in (-0.1, 1.1, math.nan, math.inf):
            config.discrete_target_entropy_ratio = ratio
            agent = C.c_uint64(123)
            check.assertEqual(lib.rx_sac_create_v2(C.byref(config), C.byref(agent)), -2)
            check.assertEqual(agent.value, 0)
        # Acceptance of a non-default ratio and rejection of invalid ratios
        # checks that the new DLL actually reads the V2 extension field.
        config.discrete_target_entropy_ratio = 0.02
        exercise_agent(check, lib, config, lib.rx_sac_create_v2, 0)
        return {"ratio": config.discrete_target_entropy_ratio}
    first = lib.rx_cuda_is_available()
    check.assertIn(first, (0, 1))
    if name == "retry":
        check.assertEqual(first, 0)
        os.environ["TORCH_CUDA_DLL"] = os.environ["RX_TEST_VALID_CUDA_DLL"]
        retried = lib.rx_cuda_is_available()
        check.assertIn(retried, (0, 1))
        os.environ.pop("TORCH_CUDA_DLL", None)
        check.assertEqual(lib.rx_cuda_is_available(), retried)
        return {"first": first, "valid": retried}
    if name == "cache":
        os.environ.pop("TORCH_CUDA_DLL", None)
        check.assertEqual(lib.rx_cuda_is_available(), first)
    return {"available": first}


class DllRegressions(unittest.TestCase):
    def probe(self, name, cuda_path="inherit"):
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        if cuda_path is None:
            environment.pop("TORCH_CUDA_DLL", None)
        elif cuda_path != "inherit":
            environment["TORCH_CUDA_DLL"] = cuda_path
        valid_path = os.environ.get("TORCH_CUDA_DLL")
        if not valid_path and os.environ.get("LIBTORCH"):
            valid_path = str(Path(os.environ["LIBTORCH"]) / "lib" / "torch_cuda.dll")
        if valid_path:
            environment["RX_TEST_VALID_CUDA_DLL"] = valid_path
        command = [sys.executable, str(Path(__file__).resolve()), "--library",
                   str(Path(OPTIONS.library).resolve()), "--probe", name]
        result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                timeout=30, creationflags=0x08000000 if os.name == "nt" else 0)
        self.assertEqual(result.returncode, 0,
                         f"{name} child exited {result.returncode}:\n{result.stdout}\n{result.stderr}")
        lines = [line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, result.stdout)
        return json.loads(lines[0])

    def require_cuda_loader(self):
        if not OPTIONS.expect_cuda_loader:
            self.skipTest("requires --expect-cuda-loader (Windows CUDA-feature build)")
        self.assertEqual(os.name, "nt", "CUDA DLL loader tests require Windows")

    def valid_cuda_path(self):
        path = os.environ.get("TORCH_CUDA_DLL")
        if not path and os.environ.get("LIBTORCH"):
            path = str(Path(os.environ["LIBTORCH"]) / "lib" / "torch_cuda.dll")
        self.assertTrue(path and Path(path).is_file(),
                        "set TORCH_CUDA_DLL or LIBTORCH to an existing CUDA DLL")
        return path

    def test_legacy_sac_abi_and_lifecycle(self):
        self.assertEqual(self.probe("legacy")["size"], 144)

    def test_sac_v2_ratio_and_lifecycle(self):
        self.assertEqual(self.probe("v2")["ratio"], 0.02)

    def test_cuda_query_returns_boolean(self):
        self.assertIn(self.probe("cuda")["available"], (0, 1))

    def test_cuda_loader_rejects_bad_configuration_without_abort(self):
        self.require_cuda_loader()
        missing = str(Path(OPTIONS.library).resolve().parent / "does-not-exist-cuda.dll")
        self.assertFalse(Path(missing).exists())
        for path in (None, "", missing):
            with self.subTest(path=path):
                self.assertEqual(self.probe("cuda", path)["available"], 0)

    def test_cuda_loader_retries_after_failure(self):
        self.require_cuda_loader()
        valid = self.valid_cuda_path()
        expected = self.probe("cuda", valid)["available"]
        result = self.probe("retry", None)
        self.assertEqual(result, {"first": 0, "valid": expected})

    def test_cuda_loader_retains_success_after_environment_removed(self):
        self.require_cuda_loader()
        self.assertIn(self.probe("cache", self.valid_cuda_path())["available"], (0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="path to the compiled ReinforceX FFI library")
    parser.add_argument("--expect-cuda-loader", action="store_true",
                        help="also test Windows CUDA DLL loading and failure recovery")
    parser.add_argument("--probe", choices=("legacy", "v2", "cuda", "retry", "cache"),
                        help=argparse.SUPPRESS)
    OPTIONS = parser.parse_args()
    if OPTIONS.probe:
        print("RESULT " + json.dumps(child_probe(OPTIONS.probe, OPTIONS.library)), flush=True)
    else:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
