"""Opt-in actual-library smoke: normalized PPO, raw shared replay, and SAC.

Each phase uses a fresh CPU process. This is a short integration test, not a
learning-performance benchmark. Existing output directories are never reused.
"""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
SYMBOLS = ("rx_agent_act_and_train_with_replay_input",
           "rx_agent_stop_episode_with_replay_input")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plain(value):
    return {name: plain(getattr(value, name)) if isinstance(getattr(value, name), C.Structure)
            else getattr(value, name) for name, _ in value._fields_}


def configs(rx, lib, legacy=False):
    ppo = rx.RxPpoConfig() if legacy else rx.RxPpoConfigV2()
    default = lib.rx_ppo_config_default if legacy else lib.rx_ppo_config_default_v2
    rx.check(default(C.byref(ppo), 3, 1), "ppo default")
    ppo.action_space = rx.RX_ACTION_CONTINUOUS
    ppo.agent.hidden_size, ppo.agent.hidden_layers = 16, 1
    ppo.agent.gamma = .99
    ppo.update_interval, ppo.minibatch_size, ppo.epochs = 16, 8, 2
    ppo.min_action, ppo.max_action = -1., 1.
    if not legacy:
        ppo.model, ppo.activation, ppo.initial_log_std = 1, 0, -.5
        ppo.target_kl = 0.
    sac = rx.RxSacConfig()
    rx.check(lib.rx_sac_config_default(C.byref(sac), 3, 1), "sac default")
    sac.action_space = rx.RX_ACTION_CONTINUOUS
    sac.agent.hidden_size, sac.agent.hidden_layers = 16, 1
    sac.agent.gamma = .99
    sac.replay_capacity, sac.replay_n_steps = 512, 1
    sac.replay_start_size, sac.batch_size = 32, 8
    sac.update_interval, sac.target_update_interval = 1, 1
    sac.alpha, sac.squash_action = .05, 1
    return ppo, sac


def checkpoint_hashes(output):
    paths = [output / "ppo.ot", output / "ppo.ot.ppo_model.json",
             output / "ppo.normalization.json"]
    # A descriptor is stored in the PPO tensor archive, not a separate file.
    paths = [p for p in paths if p.exists()]
    paths += [output / f"sac_{part}.ot" for part in
              ("actor", "critic1", "critic2", "temperature")]
    assert len(paths) >= 6 and all(p.is_file() for p in paths)
    return {p.name: digest(p) for p in paths}


def evaluate(agent, gym, rx):
    stats = agent.statistics()
    state = agent.state_dict() if hasattr(agent, "state_dict") else None
    runtime = (agent._pending_action, agent._discounted_return) if state else None
    records = []
    env = gym.make("Pendulum-v1", max_episode_steps=32)
    try:
        for seed in (730001, 730002, 730003):
            obs, _ = env.reset(seed=seed)
            total, actions = 0., []
            while True:
                action = agent.act(obs)
                actions.append(action.tolist())
                obs, reward, terminated, truncated, _ = env.step(rx.gym_action(agent, action, env.action_space))
                total += float(reward)
                if terminated or truncated:
                    break
            records.append({"seed": seed, "return": total, "length": len(actions), "actions": actions})
    finally:
        env.close()
    assert stats == agent.statistics(), "evaluation changed native statistics"
    if state:
        assert state == agent.state_dict(), "evaluation changed running moments"
        assert runtime == (agent._pending_action, agent._discounted_return), "evaluation changed episode state"
    return records


def child(args):
    import numpy as np
    import gymnasium as gym
    sys.path.insert(0, str(ROOT / "examples"))
    import reinforcex_ffi as rx
    from reinforcex_normalization import NormalizedAgent

    output = Path(args.output)
    request = json.loads((output / "request.json").read_text())
    library = Path(request["old_library"] if args.phase == "old" else request["library"])
    expected = request["old_library_sha256"] if args.phase == "old" else request["library_sha256"]
    assert digest(library) == expected
    assert all(os.environ.get(k) == "1" for k in THREADS)
    lib = C.CDLL(str(library))
    rx.configure_ffi(lib)
    rx.manual_seed(lib, 7231)
    ppo_config, sac_config = configs(rx, lib, legacy=args.phase == "old")
    result = {"status": "passed", "phase": args.phase, "pid": os.getpid(),
              "library": str(library), "library_sha256": expected,
              "device": "cpu", "threads": {k: os.environ[k] for k in THREADS},
              "gymnasium": gym.__version__, "numpy": np.__version__}
    if args.phase == "old":
        assert not any(hasattr(lib, name) for name in SYMBOLS)
        ppo = rx.create_ppo(lib, ppo_config, None, None)
        try:
            before = ppo.statistics()
            errors = []
            operations = [lambda: NormalizedAgent(ppo, 3, .99, preserve_replay_inputs=True),
                          lambda: ppo.act_and_train_with_replay_input(np.ones(3), .1, np.ones(3), 1.),
                          lambda: ppo.stop_episode_with_replay_input(np.ones(3), .1, np.ones(3), 1.)]
            for operation in operations:
                try:
                    operation()
                except RuntimeError as exc:
                    assert "does not support" in str(exc)
                    errors.append(str(exc))
                else:
                    raise AssertionError("old API silently accepted separate replay inputs")
            assert before == ppo.statistics()
            assert np.isfinite(ppo.act(np.zeros(3, dtype=np.float32))).all()
            assert before == ppo.statistics()
            result.update(rejections=errors, statistics_unchanged=True, legacy_inference_works=True)
        finally:
            ppo.close()
        return result

    assert all(hasattr(lib, name) for name in SYMBOLS)
    replay = rx.create_replay_buffer(lib, 512, 1)
    native_ppo = sac = None
    try:
        load = args.phase == "reload"
        native_ppo = rx.create_ppo(lib, ppo_config,
                                   None if load else str(output / "ppo.ot"),
                                   str(output / "ppo.ot") if load else None, replay=replay)
        sac = rx.create_sac(lib, sac_config, None if load else str(output / "sac.ot"),
                            str(output / "sac.ot") if load else None, replay=replay)
        streams = []

        class Trace:
            """Observe inputs immediately before forwarding to the real FFI."""
            def __init__(self, agent):
                self.agent = agent
                self.expected = None

            def __getattr__(self, name):
                return getattr(self.agent, name)

            def forward(self, operation, obs, reward, raw_obs, raw_reward, **kw):
                expected_obs, expected_reward = self.expected
                np.testing.assert_array_equal(raw_obs, expected_obs)
                assert raw_reward == expected_reward
                streams.append({"operation": operation, "learner_observation": obs.tolist(),
                                "learner_reward": reward, "replay_observation": raw_obs.tolist(),
                                "replay_reward": raw_reward, **kw})
                return getattr(self.agent, operation)(obs, reward, raw_obs, raw_reward, **kw)

            def act_and_train_with_replay_input(self, *args):
                return self.forward("act_and_train_with_replay_input", *args)

            def stop_episode_with_replay_input(self, *args, **kw):
                return self.forward("stop_episode_with_replay_input", *args, **kw)

        trace = Trace(native_ppo)
        ppo = NormalizedAgent(trace, 3, .99, preserve_replay_inputs=True,
                              save_path=None if load else output / "ppo.normalization.json",
                              load_path=output / "ppo.normalization.json" if load else None)
        result["configs"] = {"ppo": plain(ppo_config), "sac": plain(sac_config)}
        if not load:
            training = []
            for name, agent, seed in (("ppo", ppo, 7241), ("sac", sac, 7242)):
                env = gym.make("Pendulum-v1", max_episode_steps=32)
                try:
                    obs, _ = env.reset(seed=seed)
                    reward = 0.
                    for step in range(128):
                        if name == "ppo":
                            trace.expected = (np.asarray(obs, dtype=np.float32).copy(), float(reward))
                        before_len = len(replay)
                        action = agent.act_and_train(obs, float(reward))
                        if name == "sac" and step == 0:
                            assert before_len == len(replay) == 128
                            assert sac.statistics()["n_updates"] == 1
                            result["sac_first_update_from_ppo_only_replay"] = True
                        obs, reward, terminated, truncated, _ = env.step(rx.gym_action(agent, action, env.action_space))
                        training.append({"algorithm": name, "step": step + 1, "raw_reward": float(reward),
                                         "terminated": bool(terminated), "truncated": bool(truncated)})
                        if terminated or truncated:
                            if name == "ppo":
                                trace.expected = (np.asarray(obs, dtype=np.float32).copy(), float(reward))
                            agent.stop_episode(obs, float(reward), terminated=bool(terminated))
                            obs, _ = env.reset()
                            reward = 0.
                finally:
                    env.close()
            assert len(replay) == 256
            assert ppo.statistics()["updates"] > 0 and ppo.statistics()["optimizer_steps"] > 0
            assert sac.statistics()["n_updates"] == 128
            assert math.isclose(ppo.state_dict()["observation"]["count"], 132.0001)
            assert math.isclose(ppo.state_dict()["discounted_return"]["count"], 128.0001)
            assert len(streams) == 132
            assert any(row["learner_observation"] != row["replay_observation"] for row in streams)
            assert any(row["learner_reward"] != row["replay_reward"] for row in streams)
            (output / "raw_replay_stream.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in streams))
            (output / "training.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in training))
            ppo.save()
            sac.save()
            result.update(training_steps_per_agent=128, completed_episodes_per_agent=4,
                          shared_replay_length=len(replay), raw_forward_calls=len(streams),
                          raw_stream_matches_environment=True)
        before = checkpoint_hashes(output)
        result["statistics"] = {"ppo": ppo.statistics(), "sac": sac.statistics()}
        assert all(math.isfinite(x) for stats in result["statistics"].values() for x in stats.values())
        result["normalization_state"] = ppo.state_dict()
        result["evaluation"] = {"ppo": evaluate(ppo, gym, rx), "sac": evaluate(sac, gym, rx)}
        result["evaluation_statistics_and_normalization_unchanged"] = True
        result["checkpoint_hashes"] = checkpoint_hashes(output)
        assert before == result["checkpoint_hashes"]
        if load:
            trained = json.loads((output / "train.json").read_text())
            assert result["configs"] == trained["configs"]
            assert result["evaluation"] == trained["evaluation"], "saved policy action/return/length changed on reload"
            assert result["normalization_state"] == trained["normalization_state"]
            assert result["checkpoint_hashes"] == trained["checkpoint_hashes"]
            assert result["statistics"]["sac"]["temperature"] == trained["statistics"]["sac"]["temperature"]
            assert ppo.statistics()["updates"] == ppo.statistics()["optimizer_steps"] == 0
            assert sac.statistics()["n_updates"] == 0
            assert len(replay) == 0
            result.update(restored_evaluation_exact=True, max_return_difference=0.,
                          restored_temperature_exact=True, restored_updates_zero=True)
        return result
    finally:
        if native_ppo is not None:
            native_ppo.close()
        if sac is not None:
            sac.close()
        replay.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--old-library", type=Path)
    parser.add_argument("--old-manifest", type=Path)
    parser.add_argument("--libtorch-dir", type=Path)
    parser.add_argument("--phase", choices=("train", "reload", "old"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase:
        try:
            result = child(args)
        except Exception:
            result = {"status": "failed", "phase": args.phase, "error": traceback.format_exc()}
        write(args.output / f"{args.phase}.json", result)
        print(json.dumps({k: result[k] for k in ("status", "phase", "error") if k in result}), flush=True)
        return 0 if result["status"] == "passed" else 1
    if not all((args.library, args.manifest, args.old_library, args.old_manifest, args.libtorch_dir)):
        parser.error("both libraries/manifests and --libtorch-dir are required")
    assert not args.output.exists(), "refusing to overwrite a prior smoke"
    request = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "script": str(Path(__file__).resolve()), "script_sha256": digest(__file__),
               "scope": "integration smoke; no benchmark achievement claim",
               "python": sys.executable, "libtorch_dir": str(args.libtorch_dir.resolve())}
    for prefix in ("", "old_"):
        library = getattr(args, prefix + "library").resolve()
        manifest = getattr(args, prefix + "manifest").resolve()
        expected = json.loads(manifest.read_text())["library_sha256"]
        assert digest(library) == expected, "library does not match frozen manifest"
        request.update({prefix + "library": str(library), prefix + "library_sha256": expected,
                        prefix + "manifest": str(manifest), prefix + "manifest_sha256": digest(manifest)})
    sources = [Path(__file__), ROOT / "examples/reinforcex_ffi.py", ROOT / "examples/reinforcex_normalization.py"]
    request["source_sha256"] = {str(p.resolve()): digest(p) for p in sources}
    args.output.mkdir(parents=True)
    write(args.output / "request.json", request)
    env = os.environ.copy()
    env.update({k: "1" for k in THREADS})
    env.update(DYLD_LIBRARY_PATH=request["libtorch_dir"], LD_LIBRARY_PATH=request["libtorch_dir"])
    env.pop("REINFORCEX_LIB", None)
    results = []
    for phase in ("train", "reload", "old"):
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--phase", phase]
        with (args.output / f"{phase}.log").open("w") as log:
            done = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        result_file = args.output / f"{phase}.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {"status": "failed"}
        results.append({"phase": phase, "returncode": done.returncode, "status": result["status"]})
        print(json.dumps(results[-1]), flush=True)
        if done.returncode != 0 or result["status"] != "passed":
            break
    unchanged = request["source_sha256"] == {str(p.resolve()): digest(p) for p in sources}
    passed = len(results) == 3 and all(r["status"] == "passed" and r["returncode"] == 0 for r in results) and unchanged
    write(args.output / "summary.json", {"status": "passed" if passed else "failed", "phases": results,
                                         "source_unchanged": unchanged, "maximum_concurrent_children": 1,
                                         "cpu_threads": 1, "request": request})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
