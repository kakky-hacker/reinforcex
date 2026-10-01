"""Compare old and fixed cdylibs without requiring any newly added ABI symbol."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from reinforcex_ffi import RxDqnConfig, RxPpoConfig, RxReplayBufferConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    lib = C.CDLL(str(Path(args.library).resolve()))
    pointer = C.POINTER(C.c_float)
    out_id = C.POINTER(C.c_uint64)
    signatures = {
        "rx_dqn_config_default": ([C.POINTER(RxDqnConfig), C.c_uint64, C.c_uint64], C.c_int32),
        "rx_ppo_config_default": ([C.POINTER(RxPpoConfig), C.c_uint64, C.c_uint64], C.c_int32),
        "rx_dqn_create": ([C.POINTER(RxDqnConfig), out_id], C.c_int32),
        "rx_ppo_create": ([C.POINTER(RxPpoConfig), out_id], C.c_int32),
        "rx_replay_buffer_create": ([C.POINTER(RxReplayBufferConfig), out_id], C.c_int32),
        "rx_dqn_create_with_replay": ([C.POINTER(RxDqnConfig), C.c_uint64, out_id], C.c_int32),
        "rx_agent_act_and_train": ([C.c_uint64, pointer, C.c_uint64, C.c_float, pointer, C.c_uint64], C.c_int64),
        "rx_agent_destroy": ([C.c_uint64], C.c_int32),
        "rx_replay_buffer_destroy": ([C.c_uint64], C.c_int32),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    observation, action = (C.c_float * 2)(0.1, -0.1), (C.c_float * 1)()
    results = {}
    for algorithm, config in (("ppo", RxPpoConfig()), ("dqn", RxDqnConfig())):
        assert getattr(lib, f"rx_{algorithm}_config_default")(C.byref(config), 2, 2) == 0
        config.agent.hidden_layers, config.agent.hidden_size = 1, 8
        config.update_interval = 1
        if algorithm == "ppo":
            config.action_space, config.minibatch_size, config.epochs = 0, 1, 1
        else:
            config.batch_size, config.replay_capacity = 1, 16
        handle = C.c_uint64()
        assert getattr(lib, f"rx_{algorithm}_create")(C.byref(config), C.byref(handle)) == 0
        statuses = []
        try:
            for _ in range(4):
                status = lib.rx_agent_act_and_train(handle, observation, 2, 1.0, action, 1)
                statuses.append(status)
                if status < 0:
                    break
        finally:
            lib.rx_agent_destroy(handle)
        results[algorithm + "_singleton_statuses"] = statuses
    replay_config = RxReplayBufferConfig(100, 1)
    replay = C.c_uint64()
    assert lib.rx_replay_buffer_create(C.byref(replay_config), C.byref(replay)) == 0
    handles = []
    try:
        for obs_size in (2, 3):
            config = RxDqnConfig()
            lib.rx_dqn_config_default(C.byref(config), obs_size, 2)
            config.agent.hidden_layers, config.agent.hidden_size = 1, 8
            config.batch_size = 4
            handle = C.c_uint64()
            status = lib.rx_dqn_create_with_replay(C.byref(config), replay, C.byref(handle))
            results[f"shared_replay_obs_{obs_size}_status"] = status
            if status == 0:
                handles.append(handle)
    finally:
        for handle in handles:
            lib.rx_agent_destroy(handle)
        lib.rx_replay_buffer_destroy(replay)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
