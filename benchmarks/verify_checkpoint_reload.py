#!/usr/bin/env python3
"""Verify final native/SB3 checkpoint inference in serial, isolated processes.

The controller uses only the standard library. Each worker uses the interpreter
and environment from its training manifest, reloads one model, and evaluates
fixed held-out seeds without training or saving a model. Successful results can
be reused only while all checkpoint/reference/config fingerprints still match.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(paths):
    result = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        stat = path.stat()
        if stat.st_size <= 0:
            raise ValueError(f"empty checkpoint/input: {path}")
        result[str(path)] = {"sha256": sha256(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def checkpoint_paths(backend, run_dir, worker, spec, metadata):
    if backend == "sb3":
        return [run_dir / "final_model.zip"]
    if spec["algorithm"] == "sac":
        paths = [run_dir / f"worker{worker}_{part}.ot" for part in ("actor", "critic1", "critic2", "temperature")]
    else:
        paths = [run_dir / f"worker{worker}.ot"]
    if spec.get("rnd_config") is not None:
        rnd_worker = 0 if metadata.get("share_rnd") else worker
        paths += [run_dir / f"rnd_worker{rnd_worker}" / f"rnd_{part}.ot" for part in ("predictor", "target")]
    return paths


def reference_episodes(backend, run_dir, final, worker, count):
    if backend == "native":
        entries = [record for record in final["test"] if record["worker"] == worker]
        if len(entries) != 1:
            raise ValueError(f"expected exactly one native final test for worker {worker}")
        test = entries[0]
        if (test["seed_start"] != 900000 or test["episodes"] != 100 or test.get("reward") != "raw"
                or test.get("deterministic") is not True or test.get("split") != "test"):
            raise ValueError("native final test protocol mismatch")
        if len(test["returns"]) != 100 or len(test["lengths"]) != 100:
            raise ValueError("native final per-episode results are incomplete")
        return [{"seed": 900000 + i, "return": test["returns"][i], "length": test["lengths"][i]} for i in range(count)]
    test = final["final_evaluation"]
    if (test["seed_start"] != 900000 or test["episodes"] != 100
            or test.get("deterministic") is not True or test.get("phase") != "final"):
        raise ValueError("SB3 final test protocol mismatch")
    records = [json.loads(line) for line in (run_dir / "eval_episodes.jsonl").read_text().splitlines() if line.strip()]
    records = [row for row in records if row["phase"] == "final"]
    if len(records) != 100 or {row["seed"] for row in records} != set(range(900000, 900100)):
        raise ValueError("SB3 final per-episode results are incomplete")
    by_seed = {row["seed"]: row for row in records}
    return [{key: by_seed[900000 + i][key] for key in ("seed", "return", "length")} for i in range(count)]


def compare_episodes(observed, expected, absolute_tolerance, relative_tolerance):
    if len(observed) != len(expected):
        raise ValueError("reloaded evaluation episode count mismatch")
    comparisons = []
    for actual, reference in zip(observed, expected):
        finite = math.isfinite(actual["return"]) and math.isfinite(reference["return"])
        difference = abs(actual["return"] - reference["return"]) if finite else None
        limit = max(absolute_tolerance, relative_tolerance * max(abs(actual["return"]), abs(reference["return"]))) if finite else None
        comparisons.append({"seed": reference["seed"], "expected_return": reference["return"],
                            "observed_return": actual["return"], "absolute_difference": difference,
                            "allowed_absolute_difference": limit,
                            "expected_length": reference["length"], "observed_length": actual["length"],
                            "seed_matches": actual["seed"] == reference["seed"],
                            "length_matches": actual["length"] == reference["length"],
                            "return_matches": finite and difference <= limit})
    return comparisons


def assign_structure(structure, values):
    for name, value in values.items():
        if not hasattr(structure, name):
            raise ValueError(f"unknown saved native config field {name}")
        if isinstance(value, dict):
            assign_structure(getattr(structure, name), value)
        else:
            setattr(structure, name, value)
    return structure


def native_model(request):
    sys.path.insert(0, str(ROOT / "examples"))
    import reinforcex_ffi as rx
    metadata, spec = request["metadata"], request["spec"]
    os.environ["REINFORCEX_LIB"] = metadata["library"]
    if sha256(metadata["library"]) != metadata["library_sha256"]:
        raise ValueError("native library hash differs from the training run")
    lib = rx.load_reinforcex()
    if rx.cuda_is_available(lib):
        raise ValueError("CPU reload verifier requires the original CPU library")
    algorithm, worker = spec["algorithm"], request["worker"]
    seed_record = metadata.get("worker_seeds", [{}] * (worker + 1))[worker]
    rnd = None
    if spec.get("rnd_config") is not None:
        rnd_worker = 0 if metadata.get("share_rnd") else worker
        rnd_path = Path(request["run_dir"]) / f"rnd_worker{rnd_worker}"
        rx.manual_seed(lib, seed_record.get("rnd_seed", request["seed"] + 1000000))
        rnd = rx.create_rnd(lib, assign_structure(rx.RxRndConfig(), spec["rnd_config"]), None, str(rnd_path))
    rx.manual_seed(lib, seed_record.get("policy_seed", request["seed"]))
    config_type = {"dqn": rx.RxDqnConfig, "ppo": rx.RxPpoConfig, "sac": rx.RxSacConfigV2}[algorithm]
    config = assign_structure(config_type(), spec["config"])
    checkpoint = str(Path(request["run_dir"]) / f"worker{worker}.ot")
    if algorithm == "ppo":
        agent = rx.create_ppo(lib, config, None, checkpoint, rnd, spec.get("coefficient", 0))
    elif algorithm == "dqn":
        agent = rx.create_dqn(lib, config, None, checkpoint)
    else:
        agent = rx.create_sac(lib, config, None, checkpoint)
    return rx, agent, rnd


def tensor_state_hash(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_worker(request):
    # One active child only, one numerical-library thread, lower scheduling priority.
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    nice = None
    nice_error = None
    if hasattr(os, "nice"):
        try:
            nice = os.nice(10)
        except OSError as error:
            nice_error = repr(error)
    import gymnasium as gym
    import numpy as np
    started = time.monotonic()
    expected_files = request["checkpoint_fingerprints"]
    before_files = fingerprint_files(expected_files)
    if before_files != expected_files:
        raise ValueError("checkpoint changed between controller inspection and worker start")
    packages = {}
    for name in request["metadata"].get("packages", {}):
        packages[name] = importlib.metadata.version(name)
    if packages != request["metadata"].get("packages", {}):
        raise ValueError(f"runtime packages differ from training: {packages}")
    calls = {"training": 0, "save": 0}
    def reject_training(*_args, **_kwargs):
        calls["training"] += 1
        raise AssertionError("training is forbidden during inference reload verification")
    def reject_save(*_args, **_kwargs):
        calls["save"] += 1
        raise AssertionError("saving a model is forbidden during inference reload verification")
    backend = request["backend"]
    agent = rnd = model = env = None
    model_hash_before = model_hash_after = None
    before = after = None
    try:
        if backend == "native":
            rx, agent, rnd = native_model(request)
            agent.act_and_train = agent.stop_episode = reject_training
            agent.save = reject_save
            if rnd is not None:
                rnd.save = reject_save
            before = agent.statistics()
            update_key = "n_updates" if "n_updates" in before else "updates"
            if update_key not in before or before[update_key] != 0:
                raise AssertionError(f"freshly loaded native learner update counter is not zero: {before}")
            predict = lambda observation: rx.gym_action(agent, agent.act(observation), env.action_space)
            device = "cpu"
        else:
            import torch
            from stable_baselines3 import DQN, PPO, SAC
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            algorithm = request["algorithm"]
            model = {"dqn": DQN, "ppo": PPO, "sac": SAC}[algorithm].load(
                str(Path(request["run_dir"]) / "final_model.zip"), device="cpu")
            model.learn = model.train = reject_training
            model.save = reject_save
            model.policy.set_training_mode(False)
            before = {"n_updates": int(model._n_updates), "num_timesteps": int(model.num_timesteps)}
            update_key = "n_updates"
            model_hash_before = tensor_state_hash(model.policy.state_dict())
            predict = lambda observation: model.predict(observation, deterministic=True)[0]
            device = str(model.device)
            if device != "cpu":
                raise AssertionError(f"unexpected device: {device}")
        env = gym.make(request["env_id"])
        returns = []
        for reference in request["reference_episodes"]:
            observation, _ = env.reset(seed=reference["seed"])
            total = 0.0
            for length in range(1, env.spec.max_episode_steps + 1):
                action = predict(observation)
                if not np.isfinite(np.asarray(action)).all():
                    raise FloatingPointError("nonfinite inference action")
                observation, reward, terminated, truncated, _ = env.step(action)
                if not np.isfinite(observation).all() or not math.isfinite(float(reward)):
                    raise FloatingPointError("nonfinite inference observation/reward")
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append({"seed": reference["seed"], "return": total, "length": length})
        if backend == "native":
            after = agent.statistics()
        else:
            after = {"n_updates": int(model._n_updates), "num_timesteps": int(model.num_timesteps)}
            model_hash_after = tensor_state_hash(model.policy.state_dict())
    finally:
        if env is not None:
            env.close()
        if agent is not None:
            agent.close()
        if rnd is not None:
            rnd.close()
    after_files = fingerprint_files(expected_files)
    comparisons = compare_episodes(returns, request["reference_episodes"],
                                   request["absolute_tolerance"], request["relative_tolerance"])
    immutable = before_files == after_files
    no_updates = before == after and after[update_key] - before[update_key] == 0 and calls == {"training": 0, "save": 0}
    weights_unchanged = model_hash_before == model_hash_after
    passed = immutable and no_updates and weights_unchanged and all(
        row["seed_matches"] and row["length_matches"] and row["return_matches"] for row in comparisons)
    return {"status": "passed" if passed else "failed", "verified_at_utc": utc_now(),
            "seconds": time.monotonic() - started, "pid": os.getpid(), "nice": nice, "nice_error": nice_error,
            "python": sys.executable, "python_version": sys.version, "packages": packages, "device": device,
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "library_sha256": request["metadata"].get("library_sha256"),
            "checkpoint_before": before_files, "checkpoint_after": after_files, "checkpoint_unchanged": immutable,
            "statistics_before": before, "statistics_after": after, "statistics_unchanged": before == after,
            "new_updates_during_verification": after[update_key] - before[update_key],
            "training_and_save_calls": calls,
            "in_memory_policy_hash_before": model_hash_before, "in_memory_policy_hash_after": model_hash_after,
            "in_memory_policy_hash_scope": "SB3 state_dict" if backend == "sb3" else "not exposed through FFI; act-only entry and unchanged zero update counters checked",
            "rnd_weights_loaded": backend == "native" and request["spec"].get("rnd_config") is not None,
            "inference_environment_steps": sum(row["length"] for row in returns),
            "episodes": comparisons,
            "max_absolute_return_difference": max(row["absolute_difference"] for row in comparisons),
            "all_lengths_match": all(row["length_matches"] for row in comparisons),
            "absolute_tolerance": request["absolute_tolerance"], "relative_tolerance": request["relative_tolerance"],
            "seed_start": 900000, "evaluated_episode_count": len(comparisons)}


def worker_mode(request_path, output_path):
    request = read_json(request_path)
    try:
        result = run_worker(request)
    except Exception as error:
        result = {"status": "error", "error": repr(error), "traceback": traceback.format_exc(),
                  "verified_at_utc": utc_now()}
        try:
            result["checkpoint_after"] = fingerprint_files(request["checkpoint_fingerprints"])
            result["checkpoint_unchanged"] = result["checkpoint_after"] == request["checkpoint_fingerprints"]
        except Exception as hash_error:
            result["checkpoint_hash_error"] = repr(hash_error)
    write_json(output_path, result)
    return 0 if result["status"] == "passed" else 1


def manifest_jobs(base):
    jobs = []
    for backend in ("native", "sb3"):
        manifest = base / f"{backend}_manifest.json"
        for job in read_json(manifest)["jobs"]:
            path = Path(job["output"])
            jobs.append({**job, "backend": backend, "run_dir": path if path.is_absolute() else ROOT / path})
    return jobs


def specifications(job, metadata, base):
    if job["backend"] == "native" and metadata.get("effective_agents"):
        return metadata["effective_agents"]
    agents = read_json(base / "configs" / f"{job['case']}.json")["agents"]
    if "--workers" in job["command"]:
        agents = agents * int(job["command"][job["command"].index("--workers") + 1])
    return agents


def cache_fingerprint(request, interpreter, process_environment, script_hash):
    value = {"request": request, "python": interpreter, "environment": process_environment, "script_sha256": script_hash}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime_fingerprints(interpreter, metadata, backend):
    """Invalidate cached evidence when the interpreter/dependency installation changes."""
    executable = Path(interpreter)
    files = [executable.resolve()]
    venv = executable.parent.parent
    if (venv / "pyvenv.cfg").exists():
        files.append(venv / "pyvenv.cfg")
    files.extend(venv.glob("lib/python*/site-packages/*.dist-info/METADATA"))
    if backend == "native":
        library = Path(metadata["library"])
        if sha256(library) != metadata["library_sha256"]:
            raise ValueError("current native library differs from the original training library")
        files.append(library)
    return fingerprint_files(files)


def report_markdown(result):
    rows = result["workers"]
    passed = [row for row in rows if row["status"] == "passed"]
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
    episode_count = sum(row.get("evaluated_episode_count", 0) for row in passed)
    lines = ["# 保存済み最終モデルの再ロード推論検証", "", f"集計日時: {result['generated_at_utc']}。",
             f"予定{len(rows)} workerモデルの状態: {json.dumps(counts, ensure_ascii=False)}。"
             f"成功モデルの照合episode数は{episode_count}。未完了runはpendingで残した。", "",
             "学習manifestのPythonと環境変数を使い、native LibTorchとSB3 PyTorchを別プロセスで読み込む。"
             "追加の推論workerは同時に最大1プロセス、数値計算のthreadは1。"
             "niceを10増加する処理はOSが許可する場合のみ行い、許可されなかった場合もJSONへ記録する。"
             "native並列構成も保存された各workerを個別に再ロードする。replay/optimizer状態を伴う学習再開の検証ではない。", "",
             f"照合対象は元の最終testの先頭{result['protocol']['episodes_per_worker']} episode"
             "（seed900000から）。returnの許容誤差は"
             f"abs={result['protocol']['absolute_tolerance']}、rel={result['protocol']['relative_tolerance']}"
             "によるmath.iscloseと同じ規則。episode長とseedは完全一致を必要とする。"
             "同じtest seedを使う目的は保存前後の推論再現性の確認だけで、設定調整・モデル選抜には用いない。", "",
             "checkpoint全componentのSHA-256・サイズ・mtimeを読み込み前後で照合する。"
             "nativeのsave_pathはNone、act_and_train/stop_episode/saveは呼べば失敗するguardを付け、actだけを使う。"
             "nativeは再ロードした学習counterが0、SB3は保存時counterが復元されるためその差分が0であることを確認する。"
             "SB3はpolicy state_dictのメモリ上hashも推論前後で一致を確認する。RND付きnativeはtarget/predictorも読み込むが、"
             "決定的なpolicy推論中にRND学習は呼ばない。", "",
             "既成功結果の再利用は、checkpoint・metadata・最終評価ログ・config・検証script・使用Python/設定の"
             "fingerprintが完全一致するときだけ行う。Python実行file・依存package METADATA・native libraryも照合する。"
             "元の検証日時を維持しcachedと明記する。`--force`で再実測できる。", "",
             "|実装 / 条件 / seed / worker|状態|episode数|return最大絶対差|長さ一致|checkpoint不変|新規更新数|cached|",
             "|---|---|---:|---:|---|---|---:|---|"]
    for row in rows:
        lines.append(f"|{row['backend']} / {row['condition']} / {row['seed']} / w{row['worker']} {row['algorithm']}|"
                     f"{row['status']}|{row.get('evaluated_episode_count', '—')}|{row.get('max_absolute_return_difference', '—')}|"
                     f"{row.get('all_lengths_match', '—')}|{row.get('checkpoint_unchanged', '—')}|"
                     f"{row.get('new_updates_during_verification', '—')}|{row.get('cached', False)}|")
    lines.extend(["", "詳細は[results.json](results.json)、各workerの個別JSONとsubprocess logにある。"
                  "再ロードに失敗した場合も元checkpointや学習設定は変更しない。", "",
                  "```sh", "python3 benchmarks/verify_checkpoint_reload.py", "```", ""])
    return "\n".join(lines)


def controller(args):
    args.output.mkdir(parents=True, exist_ok=True)
    script_hash = sha256(__file__)
    records, executed = [], 0
    result = {"generated_at_utc": utc_now(), "script_sha256": script_hash,
              "protocol": {"episodes_per_worker": args.episodes, "evaluation_seed_start": 900000,
                           "absolute_tolerance": args.atol, "relative_tolerance": args.rtol,
                           "maximum_additional_inference_processes": 1,
                           "training_allowed": False, "checkpoint_selection": "none; final models only"},
              "workers": records}
    def publish():
        result["generated_at_utc"] = utc_now()
        write_json(args.output / "results.json", result)
        (args.output / "REPORT.ja.md").write_text(report_markdown(result))
    for job in manifest_jobs(args.base):
        run_dir, backend = job["run_dir"], job["backend"]
        metadata = read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
        final = read_json(run_dir / "final.json") if (run_dir / "final.json").exists() else {}
        specs = specifications(job, metadata, args.base)
        complete = bool(final) and (final.get("status") == "complete" if backend == "native" else metadata.get("status") == "complete")
        failed = (run_dir / "failure.json").exists() or metadata.get("status") == "failed"
        for worker, spec in enumerate(specs):
            common = {"backend": backend, "condition": job["condition"], "case": job["case"],
                      "seed": job["seed"], "worker": worker, "algorithm": spec["algorithm"],
                      "run_dir": str(run_dir.resolve())}
            if not complete or failed:
                records.append({**common, "status": "pending" if not failed else "training_failed", "cached": False})
                continue
            if args.max_workers and executed >= args.max_workers:
                records.append({**common, "status": "pending_verification", "reason": "--max-workers limit", "cached": False})
                continue
            work_dir = args.output / "workers" / backend / job["condition"] / f"seed_{job['seed']}" / f"worker_{worker}"
            work_dir.mkdir(parents=True, exist_ok=True)
            output_path = work_dir / "result.json"
            try:
                checkpoints = checkpoint_paths(backend, run_dir, worker, spec, metadata)
                reference_files = [run_dir / "metadata.json", run_dir / "final.json"]
                if backend == "sb3":
                    reference_files += [run_dir / "config.json", run_dir / "eval_episodes.jsonl"]
                interpreter = job["command"][0]
                request = {**common, "metadata": metadata, "spec": spec,
                           "env_id": metadata.get("env_id", final.get("environment")),
                           "runtime_fingerprints": runtime_fingerprints(interpreter, metadata, backend),
                           "checkpoint_fingerprints": fingerprint_files(checkpoints),
                           "reference_fingerprints": fingerprint_files(reference_files),
                           "reference_episodes": reference_episodes(backend, run_dir, final, worker, args.episodes),
                           "absolute_tolerance": args.atol, "relative_tolerance": args.rtol}
                child_environment = dict(job.get("environment", {}))
                child_environment.update({name: "1" for name in THREAD_VARIABLES})
                if backend == "native":
                    child_environment["REINFORCEX_LIB"] = metadata["library"]
                key = cache_fingerprint(request, interpreter, child_environment, script_hash)
                cached = read_json(output_path) if output_path.exists() else {}
                if not args.force and cached.get("status") == "passed" and cached.get("cache_fingerprint") == key:
                    records.append({**common, **cached, "cached": True})
                    continue
                request_path = work_dir / "request.json"
                write_json(request_path, request)
                raw_output = work_dir / "worker_output.json"
                if raw_output.exists():
                    raw_output.unlink()
                command = [interpreter, str(Path(__file__).resolve()), "--worker-request", str(request_path.resolve()),
                           "--worker-output", str(raw_output.resolve())]
                environment = os.environ.copy()
                if backend == "sb3":
                    environment.pop("REINFORCEX_LIB", None)
                    environment.pop("DYLD_LIBRARY_PATH", None)
                environment.update(child_environment)
                log_path = work_dir / "subprocess.log"
                print(json.dumps({"verifying": common}), flush=True)
                with log_path.open("a") as log:
                    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=args.timeout)
                executed += 1
                worker_result = read_json(raw_output) if raw_output.exists() else {"status": "error", "error": "worker produced no result"}
                worker_result.update(cache_fingerprint=key, subprocess_returncode=completed.returncode,
                                     invocation=command, subprocess_environment=child_environment,
                                     runtime_fingerprints=request["runtime_fingerprints"],
                                     subprocess_log=str(log_path.resolve()), cached=False)
                if completed.returncode != 0 and worker_result["status"] == "passed":
                    worker_result.update(status="error", error="subprocess exit code was nonzero")
                # Protect the reference evaluation itself as well as weights.
                reference_after = fingerprint_files(reference_files)
                worker_result["reference_inputs_unchanged"] = reference_after == request["reference_fingerprints"]
                if not worker_result["reference_inputs_unchanged"]:
                    worker_result.update(status="failed", error="reference input changed during verification")
                write_json(output_path, worker_result)
                with (work_dir / "attempts.jsonl").open("a") as history:
                    history.write(json.dumps(worker_result, ensure_ascii=False, allow_nan=False) + "\n")
                records.append({**common, **worker_result})
                print(json.dumps({"verified": common, "status": worker_result["status"],
                                  "max_absolute_return_difference": worker_result.get("max_absolute_return_difference")}), flush=True)
            except Exception as error:
                records.append({**common, "status": "error", "error": repr(error), "traceback": traceback.format_exc(), "cached": False})
            publish()
    publish()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE / "checkpoint_reload")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-workers", type=int, default=0, help="0 checks all complete models; intended for initial smoke only")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even fingerprint-matched passed results")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100 or args.max_workers < 0 or args.timeout <= 0:
        parser.error("episodes must be 1..100, timeout positive, max-workers nonnegative")
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    if bool(args.worker_request) != bool(args.worker_output):
        parser.error("worker-request and worker-output must be passed together")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.worker_request:
        raise SystemExit(worker_mode(args.worker_request, args.worker_output))
    result = controller(args)
    counts = {status: sum(row["status"] == status for row in result["workers"])
              for status in sorted({row["status"] for row in result["workers"]})}
    print(json.dumps({"models": len(result["workers"]), "status_counts": counts,
                      "episodes_matched": sum(row.get("evaluated_episode_count", 0) for row in result["workers"] if row["status"] == "passed")}), flush=True)
    raise SystemExit(1 if any(row["status"] in ("failed", "error", "training_failed") for row in result["workers"]) else 0)
