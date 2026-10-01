"""CPU-only Python/ctypes integration and learning validation.

Run with REINFORCEX_LIB and the platform's libtorch loader path configured::

    OMP_NUM_THREADS=1 python3 ffi/tests/run_cpu_validation.py --suite full \
        --output reports/cpu_validation/results.json

Each case is isolated in a subprocess with a deadline. No Python torch import
is needed: all inference, replay and optimization execute through the Rust FFI.
The seed controls libtorch and environments, not Rust thread_rng or scheduling.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import numpy as np
import gymnasium as gym
import reinforcex_ffi as rx


class ContextBandit:
    """One-step sign classification; optimal return is 1, random is 0.5."""
    observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x = float(self.rng.choice([-1, 1]))
        return np.array([self.x, 1], dtype=np.float32), {}

    def step(self, action):
        return np.array([self.x, 1], dtype=np.float32), float(action == int(self.x > 0)), True, False, {}

    def close(self):
        pass


class QuadraticBandit(ContextBandit):
    """One-step continuous regression; optimal return is 1."""
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x = float(self.rng.uniform(-0.8, 0.8))
        return np.array([self.x, 1], dtype=np.float32), {}

    def step(self, action):
        error = float(np.asarray(action).reshape(-1)[0]) - self.x
        return np.array([self.x, 1], dtype=np.float32), 1.0 - error * error, True, False, {}


class SparseChain(ContextBandit):
    """Six right moves reach sparse reward; resets also test time limits."""
    def reset(self, seed=None):
        self.position, self.steps = 0, 0
        return np.array([0, 1], dtype=np.float32), {}

    def step(self, action):
        self.position = min(6, max(0, self.position + (1 if action == 1 else -1)))
        self.steps += 1
        terminal = self.position == 6
        return np.array([self.position / 6, 1], dtype=np.float32), float(terminal), terminal, self.steps >= 24 and not terminal, {}


def environment(name):
    return {"ContextBandit": ContextBandit, "QuadraticBandit": QuadraticBandit,
            "SparseChain": SparseChain}[name]() if name in ("ContextBandit", "QuadraticBandit", "SparseChain") else gym.make(name)


def configuration(lib, algorithm, obs_size, action_size, discrete, *, n_steps=3, interval=None):
    if algorithm == "dqn":
        config = rx.RxDqnConfig()
        rx.check(lib.rx_dqn_config_default(C.byref(config), obs_size, action_size), "dqn defaults")
        config.learning_rate = 1e-3
        config.batch_size = 32
        config.replay_capacity = 20000
        config.replay_n_steps = n_steps
        config.update_interval = interval or 4
        config.target_update_interval = 100
        config.epsilon_decay_steps = 4000
        config.epsilon_end = 0.05
    elif algorithm == "sac":
        config = rx.RxSacConfigV2()
        rx.check(lib.rx_sac_config_default_v2(C.byref(config), obs_size, action_size), "sac defaults")
        config.action_space = rx.RX_ACTION_DISCRETE if discrete else rx.RX_ACTION_CONTINUOUS
        config.actor_learning_rate = 1e-3
        config.critic_learning_rate = 1e-3
        config.replay_capacity = 20000
        config.replay_start_size = 64
        config.batch_size = 32
        config.replay_n_steps = n_steps
        config.update_interval = interval or 4
        config.alpha = 0.05
        config.discrete_target_entropy_ratio = 0.25
    else:
        config = rx.RxPpoConfig()
        rx.check(lib.rx_ppo_config_default(C.byref(config), obs_size, action_size), "ppo defaults")
        config.action_space = rx.RX_ACTION_DISCRETE if discrete else rx.RX_ACTION_CONTINUOUS
        config.learning_rate = 1e-3
        config.update_interval = interval or 128
        config.epochs = 4
        config.minibatch_size = min(32, config.update_interval)
        config.entropy_coefficient = 0.001
    config.agent.hidden_layers = 1
    config.agent.hidden_size = 32
    config.agent.gamma = 0.99
    return config


def make_rnd(lib, obs_size, path):
    config = rx.RxRndConfig()
    rx.check(lib.rx_rnd_config_default(C.byref(config), obs_size), "rnd defaults")
    config.feature_size = 16
    config.hidden_layers = 1
    config.hidden_size = 32
    config.learning_rate = 1e-3
    config.update_interval = 32
    return rx.create_rnd(lib, config, None if path is None else str(path),
                         str(path) if path is not None and Path(path).exists() else None)


def make_agent(lib, algorithm, config, path=None, rnd=None, replay=None, load=False):
    save = None if path is None else str(path)
    source = save if load else None
    if algorithm == "dqn":
        return rx.create_dqn(lib, config, save, source, replay)
    if algorithm == "sac":
        return rx.create_sac(lib, config, save, source, replay)
    return rx.create_ppo(lib, config, save, source, rnd, 0.05, replay)


def finite_statistics(agent):
    stats = agent.statistics()
    assert all(np.isfinite(v) for v in stats.values()), stats
    return stats


def mapped_action(agent, env, obs, reward=None):
    action = agent.act(obs) if reward is None else agent.act_and_train(obs, reward)
    assert np.all(np.isfinite(action)), action
    if isinstance(env.action_space, gym.spaces.Discrete):
        assert env.action_space.contains(action), action
        return action
    assert np.all(np.asarray(action) >= -1.00001) and np.all(np.asarray(action) <= 1.00001), action
    # All benchmark policies use normalized [-1, 1] model actions.
    return (env.action_space.low + (np.asarray(action) + 1) * 0.5 *
            (env.action_space.high - env.action_space.low)).astype(env.action_space.dtype).reshape(env.action_space.shape)


def evaluate(agent, name, seed, episodes=20):
    env = environment(name)
    returns = []
    try:
        before = agent.statistics()
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            for _ in range(500):
                obs, reward, terminated, truncated, _ = env.step(mapped_action(agent, env, obs))
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
        assert before == agent.statistics(), "evaluation changed training statistics"
    finally:
        env.close()
    return {"mean": float(np.mean(returns)), "std": float(np.std(returns)), "returns": returns}


def train(agent, name, seed, steps):
    env = environment(name)
    returns = []
    trace = []
    count = episodes = terminals = truncations = 0
    started = time.monotonic()
    try:
        while count < steps:
            obs, _ = env.reset(seed=seed + episodes)
            previous_reward, total = 0.0, 0.0
            for episode_step in range(500):
                obs, reward, terminated, truncated, _ = env.step(mapped_action(agent, env, obs, previous_reward))
                count += 1
                total += float(reward)
                previous_reward = float(reward)
                if name == "CartPole-v1":
                    previous_reward = -1.0 if terminated else 0.01 * previous_reward
                elif name == "Pendulum-v1":
                    previous_reward *= 0.1
                if terminated or truncated or episode_step == 499 or count == steps:
                    agent.stop_episode(obs, previous_reward, terminated=bool(terminated))
                    terminals += int(terminated)
                    truncations += int(not terminated)
                    break
            returns.append(total)
            episodes += 1
            if episodes == 1 or episodes % 25 == 0:
                trace.append({"step": count, "episode": episodes,
                              "mean_last_25": float(np.mean(returns[-25:])),
                              "statistics": finite_statistics(agent)})
        stats = finite_statistics(agent)
    finally:
        env.close()
    elapsed = time.monotonic() - started
    return {"steps": count, "episodes": episodes, "terminal_episodes": terminals,
            "truncated_episodes": truncations, "seconds": elapsed, "steps_per_second": count / elapsed,
            "mean_first_25": float(np.mean(returns[:25])), "mean_last_25": float(np.mean(returns[-25:])),
            "statistics": stats, "trace": trace}


def payload_hash(path):
    """Ignore ZIP timestamps and serialization metadata, compare tensor payloads."""
    with zipfile.ZipFile(path) as archive:
        chunks = sorted(archive.read(n) for n in archive.namelist() if "/data/" in n)
    assert chunks, f"no tensor payloads in {path}"
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def rnd_hashes(path):
    return {p.name: payload_hash(p) for p in Path(path).glob("*.ot")}


def learning_case(lib, algorithm, name, seed, steps):
    env = environment(name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_size = int(np.prod(env.observation_space.shape))
    action_size = int(env.action_space.n) if discrete else int(np.prod(env.action_space.shape))
    env.close()
    config = configuration(lib, algorithm, obs_size, action_size, discrete)
    with tempfile.TemporaryDirectory(prefix="reinforcex-learning-") as temp:
        path = Path(temp) / "agent.ot"
        rnd_path = Path(temp) / "rnd"
        if algorithm == "rnd":
            rx.manual_seed(lib, seed + 1000000)
        rnd = make_rnd(lib, obs_size, rnd_path) if algorithm == "rnd" else None
        # Match initial PPO policies when comparing curiosity on/off.
        rx.manual_seed(lib, seed)
        agent = make_agent(lib, algorithm, config, path, rnd)
        try:
            before = evaluate(agent, name, seed + 100000)
            hashes_before = None
            if rnd:
                rnd.save()
                hashes_before = rnd_hashes(rnd_path)
            training = train(agent, name, seed, steps)
            after = evaluate(agent, name, seed + 100000)
            agent.save()
            saved_rnd_hashes = rnd_hashes(rnd_path) if rnd else None
            loaded_rnd = make_rnd(lib, obs_size, rnd_path) if rnd else None
            loaded = make_agent(lib, algorithm, config, path, loaded_rnd, load=True)
            try:
                roundtrip = evaluate(loaded, name, seed + 100000)
                assert np.allclose(after["returns"], roundtrip["returns"], rtol=0, atol=1e-5), "checkpoint changed policy"
                if loaded_rnd:
                    loaded_rnd.save()
                    assert saved_rnd_hashes == rnd_hashes(rnd_path), "new RND handle did not restore saved tensors"
            finally:
                loaded.close()
                if loaded_rnd:
                    loaded_rnd.close()
            rnd_result = None
            if rnd:
                rnd.save()
                hashes_after = rnd_hashes(rnd_path)
                assert hashes_before["rnd_target.ot"] == hashes_after["rnd_target.ot"], "RND target changed"
                assert hashes_before["rnd_predictor.ot"] != hashes_after["rnd_predictor.ot"], "RND predictor did not update"
                rnd_result = {"before": hashes_before, "after": hashes_after, "target_unchanged": True,
                              "predictor_changed": True, "new_handle_checkpoint_roundtrip": True}
            gate = None
            if name == "ContextBandit":
                gate = after["mean"] >= 0.9
            elif name == "QuadraticBandit":
                gate = after["mean"] >= 0.85
            return {"algorithm": algorithm, "environment": name, "seed": seed,
                    "before": before, "after": after, "training": training,
                    "checkpoint_roundtrip": True, "rnd": rnd_result, "learning_gate_passed": gate}
        finally:
            agent.close()
            if rnd:
                rnd.close()


def parallel_case(lib, kind, workers, seed, steps):
    algorithms = (["rnd", "sac"] * ((workers + 1) // 2))[:workers] if kind == "hybrid" else ["rnd" if kind == "rnd_independent" else kind] * workers
    replay = rx.create_replay_buffer(lib, 20000, 3)
    agents = []
    with tempfile.TemporaryDirectory(prefix="reinforcex-parallel-") as temp:
        rnd = make_rnd(lib, 2, Path(temp) / "rnd") if "rnd" in algorithms and kind != "rnd_independent" else None
        private_rnds = []
        try:
            for index, algorithm in enumerate(algorithms):
                worker_rnd = rnd
                if kind == "rnd_independent":
                    worker_rnd = make_rnd(lib, 2, Path(temp) / f"rnd_{index}")
                    private_rnds.append(worker_rnd)
                config = configuration(lib, algorithm, 2, 2, True)
                agents.append(make_agent(lib, algorithm, config, rnd=worker_rnd if algorithm == "rnd" else None, replay=replay))
            # Detaching the public resource handles must not invalidate agents'
            # Arc ownership. Concurrent worker updates still use both resources.
            replay.close()
            for private_rnd in private_rnds:
                private_rnd.close()
            if rnd:
                rnd.close()
            for private_rnd in private_rnds:
                private_rnd.close()
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(train, agent, "SparseChain" if kind == "hybrid" else "ContextBandit", seed + i * 10000, steps)
                           for i, agent in enumerate(agents)]
                results = [future.result() for future in futures]
            evaluations = [evaluate(agent, "SparseChain" if kind == "hybrid" else "ContextBandit", seed + 100000) for agent in agents]
            return {"kind": kind, "workers": workers, "steps_per_worker": steps,
                    "seconds": time.monotonic() - started, "handle_detachment_survived": True,
                    "training": results, "evaluation": evaluations}
        finally:
            for agent in reversed(agents):
                agent.close()
            replay.close()
            for private_rnd in private_rnds:
                private_rnd.close()
            if rnd:
                rnd.close()


def boundary_case(lib):
    obs = np.array([0.0, 1.0], np.float32)
    pointer = obs.ctypes.data_as(C.POINTER(C.c_float))
    output = (C.c_float * 1)(777)
    checks = []
    for algorithm in ("dqn", "ppo", "rnd", "sac"):
        config = configuration(lib, algorithm, 2, 2, True, interval=1)
        if algorithm in ("dqn", "sac"):
            config.batch_size = 1
            if algorithm == "sac":
                config.replay_start_size = 1
        rnd = make_rnd(lib, 2, None) if algorithm == "rnd" else None
        agent = make_agent(lib, algorithm, config, rnd=rnd)
        try:
            output[0] = 777
            statistics_before = agent.statistics()
            for reward in (float("nan"), float("inf")):
                assert lib.rx_agent_act_and_train(agent.handle, pointer, 2, reward, output, 1) == -2
            assert lib.rx_agent_act_and_train(agent.handle, pointer, 2, 0, output, 0) == -4
            assert output[0] == 777
            assert lib.rx_agent_stop_episode_with_terminal(agent.handle, pointer, 2, 0, 2) == -2
            assert lib.rx_agent_act(agent.handle, pointer, 1, output, 1) == -2
            bad = np.array([np.nan, 1], np.float32)
            assert lib.rx_agent_act(agent.handle, bad.ctypes.data_as(C.POINTER(C.c_float)), 2, output, 1) == -2
            assert agent.statistics() == statistics_before, "invalid call mutated statistics"
            for i in range(20):
                agent.act_and_train(obs, 0.0)
                agent.stop_episode(obs, 1.0, terminated=(i % 2 == 0))
            statistics_after = finite_statistics(agent)
            assert statistics_after.get("updates", statistics_after.get("n_updates", 0)) > 0, statistics_after
            checks.append({"algorithm": algorithm, "interval_one_survived": True,
                           "invalid_calls_rejected_before_mutation": True, "statistics": statistics_after})
        finally:
            handle = agent.handle
            agent.close()
            assert lib.rx_agent_destroy(handle) == -3
            assert lib.rx_agent_act(handle, pointer, 2, output, 1) == -3
            if rnd:
                rnd.close()
    replay = rx.create_replay_buffer(lib, 1000, 3)
    config = configuration(lib, "dqn", 2, 2, True)
    first = make_agent(lib, "dqn", config, replay=replay)
    mismatches = []
    try:
        for field, value in (("obs_size", 3), ("action_size", 3), ("gamma", 0.9)):
            other = configuration(lib, "dqn", 2, 2, True)
            setattr(other.agent, field, value)
            try:
                wrong = make_agent(lib, "dqn", other, replay=replay)
            except RuntimeError as error:
                assert "status -2" in str(error)
                mismatches.append(field)
            else:
                wrong.close()
                raise AssertionError(f"accepted incompatible shared replay: {field}")
        other = configuration(lib, "sac", 2, 2, False)
        try:
            wrong = make_agent(lib, "sac", other, replay=replay)
        except RuntimeError as error:
            assert "status -2" in str(error)
            mismatches.append("action_space")
        else:
            wrong.close()
            raise AssertionError("accepted continuous/discrete replay mix")
    finally:
        first.close()
        replay.close()
    return {"boundary_checks": checks, "replay_mismatches_rejected": mismatches}


def lifecycle_case(lib):
    """Repeated abandoned trajectories must not retain agent/RND handles."""
    replay = rx.create_replay_buffer(lib, 256, 3)
    obs = np.array([0.25, 1.0], np.float32)
    snapshots = []
    started = time.monotonic()
    try:
        for cycle in range(100):
            rnd = make_rnd(lib, 2, None)
            agents = []
            try:
                for algorithm in ("dqn", "ppo", "sac", "rnd"):
                    config = configuration(lib, algorithm, 2, 2, True)
                    agents.append(make_agent(lib, algorithm, config, rnd=rnd if algorithm == "rnd" else None, replay=replay))
                rnd.close()
                for agent in agents:
                    for _ in range(12):
                        agent.act_and_train(obs, 0.1)
                    finite_statistics(agent)
                if cycle in (0, 9, 49, 99):
                    snapshots.append({"cycle": cycle + 1, "replay_len": len(replay),
                                      "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None})
            finally:
                for agent in agents:
                    agent.close()
                rnd.close()
        assert len(replay) == 256
        return {"cycles": 100, "agents_created_and_destroyed": 400, "rnd_handles_detached": 100,
                "unfinished_episodes_dropped": 400, "seconds": time.monotonic() - started,
                "memory_snapshots": snapshots, "note": "Peak RSS is diagnostic only, not a leak-free proof; exact tail release is tested in Rust."}
    finally:
        replay.close()


def cases(suite, seeds):
    result = [{"name": "ffi_boundaries", "type": "boundary", "seed": 42},
              {"name": "lifecycle_churn", "type": "lifecycle", "seed": 42}]
    for seed in seeds:
        for algorithm in ("dqn", "ppo", "sac", "rnd"):
            result.append({"name": f"bandit_{algorithm}_{seed}", "type": "learning", "algorithm": algorithm,
                           "environment": "ContextBandit", "seed": seed, "steps": 2000})
        for algorithm in ("ppo", "sac"):
            result.append({"name": f"quadratic_{algorithm}_{seed}", "type": "learning", "algorithm": algorithm,
                           "environment": "QuadraticBandit", "seed": seed, "steps": 4000})
        if suite == "full":
            for name, algorithms, steps in (("CartPole-v1", ("dqn", "ppo", "sac", "rnd"), 16000),
                                             ("Pendulum-v1", ("ppo", "sac"), 10000),
                                             ("SparseChain", ("ppo", "rnd"), 4000)):
                for algorithm in algorithms:
                    result.append({"name": f"{name}_{algorithm}_{seed}", "type": "learning", "algorithm": algorithm,
                                   "environment": name, "seed": seed, "steps": steps})
    for workers in ((2, 4, 8) if suite == "full" else (2,)):
        for kind in ("dqn", "sac", "ppo", "rnd", "rnd_independent", "hybrid"):
            result.append({"name": f"parallel_{kind}_{workers}", "type": "parallel", "kind": kind,
                           "workers": workers, "seed": seeds[0], "steps": 1000})
    if suite == "full":
        for name in ("Acrobot-v1", "MountainCar-v0"):
            result.append({"name": f"{name}_rnd_42", "type": "learning", "algorithm": "rnd",
                           "environment": name, "seed": 42, "steps": 2000})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cpu_validation/results.json")
    parser.add_argument("--case", help="run only names containing this string")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.child:
        spec = json.loads(args.child)
        lib = rx.load_reinforcex()
        assert not rx.cuda_is_available(lib), "this validation is intended for a CPU build"
        rx.manual_seed(lib, spec["seed"])
        if spec["type"] == "learning":
            result = learning_case(lib, spec["algorithm"], spec["environment"], spec["seed"], spec["steps"])
        elif spec["type"] == "parallel":
            result = parallel_case(lib, spec["kind"], spec["workers"], spec["seed"], spec["steps"])
        elif spec["type"] == "lifecycle":
            result = lifecycle_case(lib)
        else:
            result = boundary_case(lib)
        print("RESULT " + json.dumps(result, allow_nan=False))
        return
    report = {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
              "gymnasium": gym.__version__, "library": os.environ.get("REINFORCEX_LIB"),
              "threads": os.environ.get("OMP_NUM_THREADS"), "suite": args.suite,
              "seed_limitations": "libtorch and environment seeds only; Rust thread_rng and thread scheduling are not seeded",
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = [c for c in cases(args.suite, args.seeds) if not args.case or args.case in c["name"]]
    for index, spec in enumerate(selected, 1):
        started = time.monotonic()
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)],
                                     capture_output=True, text=True, timeout=args.timeout)
            lines = [line[7:] for line in process.stdout.splitlines() if line.startswith("RESULT ")]
            if process.returncode != 0 or len(lines) != 1:
                raise RuntimeError(f"exit={process.returncode}\n{process.stdout[-4000:]}\n{process.stderr[-6000:]}")
            entry = {"case": spec, "status": "passed", "result": json.loads(lines[0]),
                     "stderr": process.stderr[-2000:]}
        except Exception:
            entry = {"case": spec, "status": "failed", "error": traceback.format_exc()}
        entry["wall_seconds"] = time.monotonic() - started
        report["results"].append(entry)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[{index}/{len(selected)}] {spec['name']}: {entry['status']} ({entry['wall_seconds']:.1f}s)", flush=True)
        if entry["status"] == "failed":
            print(entry["error"][-2000:], flush=True)
    failures = [r for r in report["results"] if r["status"] != "passed"]
    gates = [r for r in report["results"] if r.get("result", {}).get("learning_gate_passed") is False]
    print(f"Saved {args.output}: {len(report['results'])} cases, {len(failures)} execution failures, {len(gates)} learning gate failures")
    raise SystemExit(bool(failures or gates))


if __name__ == "__main__":
    main()
