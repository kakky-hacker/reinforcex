"""Train Gymnasium's CartPole-v1 with ReinforceX PPO through its C FFI."""

from __future__ import annotations

import argparse
import ctypes as C
import ctypes.util
import importlib.util
import os
import re
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np


RX_OK = 0
RX_ACTION_DISCRETE = 0
_DLL_DIRECTORY_HANDLES = []


class RxAgentConfig(C.Structure):
    _fields_ = [
        ("obs_size", C.c_uint64),
        ("action_size", C.c_uint64),
        ("hidden_layers", C.c_uint64),
        ("hidden_size", C.c_uint64),
        ("gamma", C.c_double),
    ]


class RxPpoConfig(C.Structure):
    _fields_ = [
        ("agent", RxAgentConfig),
        ("action_space", C.c_uint32),
        ("learning_rate", C.c_double),
        ("gae_lambda", C.c_double),
        ("update_interval", C.c_uint64),
        ("epochs", C.c_uint64),
        ("minibatch_size", C.c_uint64),
        ("policy_clip_epsilon", C.c_double),
        ("value_clip_range", C.c_double),
        ("value_loss_coefficient", C.c_double),
        ("entropy_coefficient", C.c_double),
        ("standardize_gae", C.c_uint32),
        ("min_action", C.c_double),
        ("max_action", C.c_double),
        ("min_variance", C.c_double),
    ]


def add_windows_dll_directories(repo_root: Path) -> None:
    """Make LibTorch dependencies visible to Python 3.8+'s DLL loader."""
    if os.name != "nt":
        return

    directories = [repo_root / "target" / "release"]
    libtorch = os.environ.get("LIBTORCH")
    if libtorch:
        directories.append(Path(libtorch) / "lib")

    # The repository's .env is a PowerShell file, so read just its LIBTORCH path.
    env_file = repo_root / ".env"
    if env_file.is_file():
        match = re.search(
            r'^\s*\$env:LIBTORCH\s*=\s*["\']([^"\']+)["\']',
            env_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            directories.append(Path(match.group(1)) / "lib")

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        directories.append(Path(torch_spec.origin).parent / "lib")

    directories.extend(
        Path(item)
        for item in os.environ.get("REINFORCEX_DLL_DIRS", "").split(os.pathsep)
        if item
    )
    for directory in dict.fromkeys(path.resolve() for path in directories):
        if directory.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))


def load_reinforcex() -> C.CDLL:
    """Load an installed library, with a repository build as a convenience fallback."""
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    add_windows_dll_directories(repo_root)
    names = {
        "nt": "reinforcex.dll",
        "posix": "libreinforcex.so",
    }
    candidates = [
        os.environ.get("REINFORCEX_LIB"),
        ctypes.util.find_library("reinforcex"),
        str(repo_root / "target" / "release" / names[os.name]),
    ]
    if os.name == "posix":
        candidates.append(
            str(
                repo_root / "target" / "release" / "libreinforcex.dylib"
            )
        )

    errors = []
    for candidate in filter(None, candidates):
        try:
            return C.CDLL(candidate)
        except OSError as error:
            errors.append(f"  {candidate}: {error}")
    details = "\n".join(errors) or "  no library candidate was found"
    raise RuntimeError(
        "Could not load ReinforceX. Set REINFORCEX_LIB to the dynamic library path.\n"
        + details
    )


def configure_ffi(lib: C.CDLL) -> None:
    float_pointer = C.POINTER(C.c_float)
    lib.rx_ppo_config_default.argtypes = [C.POINTER(RxPpoConfig), C.c_uint64, C.c_uint64]
    lib.rx_ppo_config_default.restype = C.c_int32
    lib.rx_ppo_create.argtypes = [C.POINTER(RxPpoConfig), C.POINTER(C.c_uint64)]
    lib.rx_ppo_create.restype = C.c_int32
    lib.rx_agent_act_and_train.argtypes = [
        C.c_uint64,
        float_pointer,
        C.c_uint64,
        C.c_float,
        float_pointer,
        C.c_uint64,
    ]
    lib.rx_agent_act_and_train.restype = C.c_int64
    lib.rx_agent_stop_episode.argtypes = [
        C.c_uint64,
        float_pointer,
        C.c_uint64,
        C.c_float,
    ]
    lib.rx_agent_stop_episode.restype = C.c_int32
    lib.rx_agent_destroy.argtypes = [C.c_uint64]
    lib.rx_agent_destroy.restype = C.c_int32


def check(status: int, operation: str) -> None:
    if status < RX_OK:
        raise RuntimeError(f"{operation} failed with ReinforceX status {status}")


def observation_pointer(observation: np.ndarray) -> tuple[np.ndarray, C.POINTER(C.c_float)]:
    array = np.ascontiguousarray(observation, dtype=np.float32)
    return array, array.ctypes.data_as(C.POINTER(C.c_float))


def train(seed: int) -> None:
    env = gym.make("CartPole-v1")
    lib = load_reinforcex()
    configure_ffi(lib)

    config = RxPpoConfig()
    check(lib.rx_ppo_config_default(C.byref(config), 4, 2), "rx_ppo_config_default")
    config.action_space = RX_ACTION_DISCRETE
    config.agent.hidden_size = 64
    config.learning_rate = 2.5e-4
    config.update_interval = 500
    config.epochs = 10
    config.minibatch_size = 32

    agent_id = C.c_uint64()
    check(lib.rx_ppo_create(C.byref(config), C.byref(agent_id)), "rx_ppo_create")
    recent_returns: deque[float] = deque(maxlen=100)

    episodes = 10000

    try:
        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode - 1)
            episode_return = 0.0
            reward_for_previous_action = 0.0

            while True:
                obs_array, obs_ptr = observation_pointer(observation)
                action_buffer = (C.c_float * 1)()
                written = lib.rx_agent_act_and_train(
                    agent_id.value,
                    obs_ptr,
                    obs_array.size,
                    reward_for_previous_action,
                    action_buffer,
                    1,
                )
                check(written, "rx_agent_act_and_train")
                if written != 1:
                    raise RuntimeError(f"expected one action value, got {written}")

                action = int(action_buffer[0])
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward

                if terminated or truncated:
                    final_obs, final_obs_ptr = observation_pointer(observation)
                    check(
                        lib.rx_agent_stop_episode(
                            agent_id.value,
                            final_obs_ptr,
                            final_obs.size,
                            reward,
                        ),
                        "rx_agent_stop_episode",
                    )
                    break
                reward_for_previous_action = reward

            recent_returns.append(episode_return)
            if episode == 1 or episode % 20 == 0:
                print(
                    f"episode={episode:4d} return={episode_return:6.1f} "
                    f"mean({len(recent_returns)})={np.mean(recent_returns):6.1f}"
                )
    finally:
        env.close()
        check(lib.rx_agent_destroy(agent_id.value), "rx_agent_destroy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.seed)


if __name__ == "__main__":
    main()
