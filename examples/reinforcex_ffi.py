"""Small ``ctypes`` helpers shared by the ReinforceX Python examples."""

from __future__ import annotations

import argparse
import ctypes as C
from ctypes import util as ctypes_util
import importlib.util
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np


RX_OK = 0
RX_ACTION_DISCRETE = 0
RX_ACTION_CONTINUOUS = 1
RX_STAT_NAME_LEN = 64
_DLL_DIRECTORY_HANDLES = []


class RxAgentConfig(C.Structure):
    _fields_ = [
        ("obs_size", C.c_uint64),
        ("action_size", C.c_uint64),
        ("hidden_layers", C.c_uint64),
        ("hidden_size", C.c_uint64),
        ("gamma", C.c_double),
    ]


class RxDqnConfig(C.Structure):
    _fields_ = [
        ("agent", RxAgentConfig),
        ("learning_rate", C.c_double),
        ("batch_size", C.c_uint64),
        ("replay_capacity", C.c_uint64),
        ("replay_n_steps", C.c_uint64),
        ("update_interval", C.c_uint64),
        ("target_update_interval", C.c_uint64),
        ("epsilon_start", C.c_double),
        ("epsilon_end", C.c_double),
        ("epsilon_decay_steps", C.c_uint64),
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


class RxSacConfig(C.Structure):
    _fields_ = [
        ("agent", RxAgentConfig),
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


class RxReplayBufferConfig(C.Structure):
    _fields_ = [
        ("capacity", C.c_uint64),
        ("n_steps", C.c_uint64),
    ]


class RxRndConfig(C.Structure):
    _fields_ = [
        ("obs_size", C.c_uint64),
        ("feature_size", C.c_uint64),
        ("hidden_layers", C.c_uint64),
        ("hidden_size", C.c_uint64),
        ("learning_rate", C.c_double),
        ("update_interval", C.c_uint64),
    ]


class RxStatistic(C.Structure):
    _fields_ = [
        ("name", C.c_char * RX_STAT_NAME_LEN),
        ("value", C.c_double),
    ]


def add_windows_dll_directories(repo_root: Path) -> None:
    """Make LibTorch dependencies visible to Python 3.8+'s DLL loader."""
    if os.name != "nt":
        return

    directories = [repo_root / "target" / "release"]
    libtorch = os.environ.get("LIBTORCH")
    if libtorch:
        directories.append(Path(libtorch) / "lib")

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


def configure_ffi(lib: C.CDLL) -> None:
    float_pointer = C.POINTER(C.c_float)
    char_pointer = C.c_char_p
    uint64_pointer = C.POINTER(C.c_uint64)

    lib.rx_dqn_config_default.argtypes = [C.POINTER(RxDqnConfig), C.c_uint64, C.c_uint64]
    lib.rx_ppo_config_default.argtypes = [C.POINTER(RxPpoConfig), C.c_uint64, C.c_uint64]
    lib.rx_sac_config_default.argtypes = [C.POINTER(RxSacConfig), C.c_uint64, C.c_uint64]
    lib.rx_replay_buffer_config_default.argtypes = [
        C.POINTER(RxReplayBufferConfig),
        C.c_uint64,
        C.c_uint64,
    ]
    lib.rx_rnd_config_default.argtypes = [C.POINTER(RxRndConfig), C.c_uint64]
    for name in (
        "rx_dqn_config_default",
        "rx_ppo_config_default",
        "rx_sac_config_default",
        "rx_replay_buffer_config_default",
        "rx_rnd_config_default",
    ):
        getattr(lib, name).restype = C.c_int32

    lib.rx_dqn_create_with_paths.argtypes = [
        C.POINTER(RxDqnConfig),
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    lib.rx_dqn_create_with_replay_and_paths.argtypes = [
        C.POINTER(RxDqnConfig),
        C.c_uint64,
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    lib.rx_ppo_create_with_paths.argtypes = [
        C.POINTER(RxPpoConfig),
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    lib.rx_ppo_create_with_rnd_and_paths.argtypes = [
        C.POINTER(RxPpoConfig),
        C.c_uint64,
        C.c_double,
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    lib.rx_sac_create_with_paths.argtypes = [
        C.POINTER(RxSacConfig),
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    lib.rx_sac_create_with_replay_and_paths.argtypes = [
        C.POINTER(RxSacConfig),
        C.c_uint64,
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    lib.rx_replay_buffer_create.argtypes = [
        C.POINTER(RxReplayBufferConfig),
        uint64_pointer,
    ]
    lib.rx_rnd_create_with_paths.argtypes = [
        C.POINTER(RxRndConfig),
        char_pointer,
        char_pointer,
        uint64_pointer,
    ]
    for name in (
        "rx_dqn_create_with_paths",
        "rx_dqn_create_with_replay_and_paths",
        "rx_ppo_create_with_paths",
        "rx_ppo_create_with_rnd_and_paths",
        "rx_sac_create_with_paths",
        "rx_sac_create_with_replay_and_paths",
        "rx_replay_buffer_create",
        "rx_rnd_create_with_paths",
    ):
        getattr(lib, name).restype = C.c_int32

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
    lib.rx_agent_statistics_len.argtypes = [C.c_uint64, uint64_pointer]
    lib.rx_agent_statistics_len.restype = C.c_int32
    lib.rx_agent_statistics.argtypes = [
        C.c_uint64,
        C.POINTER(RxStatistic),
        C.c_uint64,
    ]
    lib.rx_agent_statistics.restype = C.c_int64
    lib.rx_agent_save.argtypes = [C.c_uint64]
    lib.rx_agent_save.restype = C.c_int32
    lib.rx_agent_destroy.argtypes = [C.c_uint64]
    lib.rx_agent_destroy.restype = C.c_int32

    lib.rx_replay_buffer_destroy.argtypes = [C.c_uint64]
    lib.rx_replay_buffer_destroy.restype = C.c_int32
    lib.rx_rnd_save.argtypes = [C.c_uint64]
    lib.rx_rnd_save.restype = C.c_int32
    lib.rx_rnd_destroy.argtypes = [C.c_uint64]
    lib.rx_rnd_destroy.restype = C.c_int32


def load_reinforcex() -> C.CDLL:
    """Load an installed library, with a repository build as a fallback."""
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    add_windows_dll_directories(repo_root)
    names = {"nt": "reinforcex.dll", "posix": "libreinforcex.so"}
    candidates = [
        os.environ.get("REINFORCEX_LIB"),
        ctypes_util.find_library("reinforcex"),
        str(repo_root / "target" / "release" / names[os.name]),
    ]
    if os.name == "posix":
        candidates.append(str(repo_root / "target" / "release" / "libreinforcex.dylib"))

    errors = []
    for candidate in filter(None, candidates):
        try:
            lib = C.CDLL(candidate)
            configure_ffi(lib)
            return lib
        except OSError as error:
            errors.append(f"  {candidate}: {error}")
    details = "\n".join(errors) or "  no library candidate was found"
    raise RuntimeError(
        "Could not load ReinforceX. Set REINFORCEX_LIB to the dynamic library path.\n"
        + details
    )


def check(status: int, operation: str) -> None:
    if status < RX_OK:
        raise RuntimeError(f"{operation} failed with ReinforceX status {status}")


def observation_array(observation: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(observation, dtype=np.float32).reshape(-1)


def _path_bytes(path: str | None) -> bytes | None:
    return None if path is None else os.fspath(path).encode("utf-8")


def path_for_agent(path: str | None, agent_id: int) -> str | None:
    return None if path is None else path.replace("{agent_id}", str(agent_id))


def rnd_path(path: str | None, agent_id: int | str) -> str | None:
    resolved = None if path is None else path.replace("{agent_id}", str(agent_id))
    return None if resolved is None else f"{resolved}.rnd"


class Agent:
    def __init__(self, lib: C.CDLL, handle: int, output_size: int, discrete: bool):
        self.lib = lib
        self.handle = handle
        self.output_size = output_size
        self.discrete = discrete

    def act_and_train(self, observation: np.ndarray, reward: float):
        obs = observation_array(observation)
        output = (C.c_float * self.output_size)()
        written = self.lib.rx_agent_act_and_train(
            self.handle,
            obs.ctypes.data_as(C.POINTER(C.c_float)),
            obs.size,
            reward,
            output,
            self.output_size,
        )
        check(written, "rx_agent_act_and_train")
        if written != self.output_size:
            raise RuntimeError(f"expected {self.output_size} action values, got {written}")
        values = np.ctypeslib.as_array(output).copy()
        return int(values[0]) if self.discrete else values

    def stop_episode(self, observation: np.ndarray, reward: float) -> None:
        obs = observation_array(observation)
        check(
            self.lib.rx_agent_stop_episode(
                self.handle,
                obs.ctypes.data_as(C.POINTER(C.c_float)),
                obs.size,
                reward,
            ),
            "rx_agent_stop_episode",
        )

    def statistics(self) -> dict[str, float]:
        length = C.c_uint64()
        check(
            self.lib.rx_agent_statistics_len(self.handle, C.byref(length)),
            "rx_agent_statistics_len",
        )
        if length.value == 0:
            return {}
        output = (RxStatistic * length.value)()
        written = self.lib.rx_agent_statistics(self.handle, output, length.value)
        check(written, "rx_agent_statistics")
        return {
            bytes(item.name).split(b"\0", 1)[0].decode("utf-8"): item.value
            for item in output[:written]
        }

    def save(self) -> None:
        check(self.lib.rx_agent_save(self.handle), "rx_agent_save")

    def close(self) -> None:
        if self.handle:
            check(self.lib.rx_agent_destroy(self.handle), "rx_agent_destroy")
            self.handle = 0


class ReplayBuffer:
    def __init__(self, lib: C.CDLL, handle: int):
        self.lib = lib
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            check(
                self.lib.rx_replay_buffer_destroy(self.handle),
                "rx_replay_buffer_destroy",
            )
            self.handle = 0


class Rnd:
    def __init__(self, lib: C.CDLL, handle: int):
        self.lib = lib
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            check(self.lib.rx_rnd_destroy(self.handle), "rx_rnd_destroy")
            self.handle = 0


def create_replay_buffer(lib: C.CDLL, capacity: int, n_steps: int) -> ReplayBuffer:
    config = RxReplayBufferConfig()
    check(
        lib.rx_replay_buffer_config_default(C.byref(config), capacity, n_steps),
        "rx_replay_buffer_config_default",
    )
    handle = C.c_uint64()
    check(lib.rx_replay_buffer_create(C.byref(config), C.byref(handle)), "rx_replay_buffer_create")
    return ReplayBuffer(lib, handle.value)


def create_dqn(
    lib: C.CDLL,
    config: RxDqnConfig,
    save_path: str | None,
    load_path: str | None,
    replay: ReplayBuffer | None = None,
) -> Agent:
    handle = C.c_uint64()
    if replay is None:
        status = lib.rx_dqn_create_with_paths(
            C.byref(config), _path_bytes(save_path), _path_bytes(load_path), C.byref(handle)
        )
        operation = "rx_dqn_create_with_paths"
    else:
        status = lib.rx_dqn_create_with_replay_and_paths(
            C.byref(config),
            replay.handle,
            _path_bytes(save_path),
            _path_bytes(load_path),
            C.byref(handle),
        )
        operation = "rx_dqn_create_with_replay_and_paths"
    check(status, operation)
    return Agent(lib, handle.value, 1, True)


def create_ppo(
    lib: C.CDLL,
    config: RxPpoConfig,
    save_path: str | None,
    load_path: str | None,
    rnd: Rnd | None = None,
    curiosity_reward_coefficient: float = 1.0,
) -> Agent:
    handle = C.c_uint64()
    if rnd is None:
        status = lib.rx_ppo_create_with_paths(
            C.byref(config), _path_bytes(save_path), _path_bytes(load_path), C.byref(handle)
        )
        operation = "rx_ppo_create_with_paths"
    else:
        status = lib.rx_ppo_create_with_rnd_and_paths(
            C.byref(config),
            rnd.handle,
            curiosity_reward_coefficient,
            _path_bytes(save_path),
            _path_bytes(load_path),
            C.byref(handle),
        )
        operation = "rx_ppo_create_with_rnd_and_paths"
    check(status, operation)
    discrete = config.action_space == RX_ACTION_DISCRETE
    return Agent(lib, handle.value, 1 if discrete else config.agent.action_size, discrete)


def create_sac(
    lib: C.CDLL,
    config: RxSacConfig,
    save_path: str | None,
    load_path: str | None,
    replay: ReplayBuffer | None = None,
) -> Agent:
    handle = C.c_uint64()
    if replay is None:
        status = lib.rx_sac_create_with_paths(
            C.byref(config), _path_bytes(save_path), _path_bytes(load_path), C.byref(handle)
        )
        operation = "rx_sac_create_with_paths"
    else:
        status = lib.rx_sac_create_with_replay_and_paths(
            C.byref(config),
            replay.handle,
            _path_bytes(save_path),
            _path_bytes(load_path),
            C.byref(handle),
        )
        operation = "rx_sac_create_with_replay_and_paths"
    check(status, operation)
    discrete = config.action_space == RX_ACTION_DISCRETE
    return Agent(lib, handle.value, 1 if discrete else config.agent.action_size, discrete)


def create_rnd(
    lib: C.CDLL,
    config: RxRndConfig,
    save_path: str | None,
    load_path: str | None,
) -> Rnd:
    handle = C.c_uint64()
    check(
        lib.rx_rnd_create_with_paths(
            C.byref(config), _path_bytes(save_path), _path_bytes(load_path), C.byref(handle)
        ),
        "rx_rnd_create_with_paths",
    )
    return Rnd(lib, handle.value)


RewardTransform = Callable[[float, int, bool, int], float]


def identity_reward(reward: float, _step: int, _done: bool, _max_steps: int) -> float:
    return reward


def train_gym_agent(
    *,
    agent: Agent,
    env_id: str,
    agent_id: int,
    seed: int,
    episodes: int,
    max_steps: int,
    log_interval: int,
    reward_transform: RewardTransform = identity_reward,
) -> None:
    """Run one Gymnasium environment and feed transitions through an FFI agent."""
    try:
        import gymnasium as gym
    except ImportError as error:
        raise RuntimeError(
            "Gymnasium is required for the examples. Install the appropriate "
            "gymnasium extra before training."
        ) from error

    env = gym.make(env_id)
    recent_returns: deque[float] = deque(maxlen=100)

    try:
        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode - 1)
            previous_reward = 0.0
            episode_return = 0.0
            training_return = 0.0

            for step in range(1, max_steps + 1):
                action = agent.act_and_train(observation, previous_reward)

                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                previous_reward = reward_transform(float(reward), step, done, max_steps)
                episode_return += float(reward)
                training_return += previous_reward
                observation = next_observation

                if done:
                    agent.stop_episode(observation, previous_reward)
                    break
            else:
                agent.stop_episode(observation, previous_reward)

            recent_returns.append(episode_return)
            if episode == 1 or episode % log_interval == 0:
                stats = agent.statistics()
                stats_text = f" stats={stats}" if stats else ""
                print(
                    f"agent={agent_id} episode={episode:5d} "
                    f"return={episode_return:9.2f} train_return={training_return:9.2f} "
                    f"mean({len(recent_returns)})={np.mean(recent_returns):9.2f}{stats_text}"
                )
            if episode % log_interval == 0:
                agent.save()

        agent.save()
    finally:
        env.close()


def run_parallel(worker_count: int, worker: Callable[[int], None]) -> None:
    if worker_count == 1:
        worker(0)
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(worker_count)]
        for future in futures:
            future.result()


def training_parser(
    description: str,
    *,
    episodes: int,
    max_steps: int,
    log_interval: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=episodes)
    parser.add_argument("--max-steps", type=int, default=max_steps)
    parser.add_argument("--log-interval", type=int, default=log_interval)
    parser.add_argument("--parallel", type=int, default=1, help="number of independent Gym workers")
    parser.add_argument("--save-path", help="checkpoint path; {agent_id} is replaced per worker")
    parser.add_argument("--load-path", help="checkpoint path; {agent_id} is replaced per worker")
    return parser


def validate_training_args(args: argparse.Namespace) -> None:
    for name in ("episodes", "max_steps", "log_interval", "parallel"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
