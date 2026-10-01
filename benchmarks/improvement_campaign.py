"""Execute a saved list of study runs with bounded CPU concurrency and resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from improvement_configs import effective_configuration, study_request

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def prepare_job(job):
    required = {"case", "seed", "steps", "output", "library", "manifest"}
    optional = {"overrides", "stage", "workers", "eval-episodes", "validation-episodes",
                "checkpoints", "share-rnd", "serial-workers", "final-seed"}
    if not isinstance(job, dict) or required - set(job) or set(job) - required - optional:
        raise ValueError(f"missing or unknown job fields: {job}")
    for flag in ("share-rnd", "serial-workers"):
        if flag in job and type(job[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")
    if "final-seed" in job and job["final-seed"] is None:
        raise ValueError("final-seed must be an explicit confirmation block integer, not null")
    output, library, manifest_path = (ROOT / job[name] for name in ("output", "library", "manifest"))
    output, library, manifest_path = output.resolve(), library.resolve(strict=True), manifest_path.resolve(strict=True)
    if output == ROOT or output in ROOT.parents:
        raise ValueError("output cannot be the checkout or one of its parents")
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise ValueError(f"requested library differs from manifest: {library}")
    override_path = None if not job.get("overrides") else (ROOT / job["overrides"]).resolve(strict=True)
    overrides = None if override_path is None else json.loads(override_path.read_text())
    case, specs = effective_configuration(job["case"], override_path, job.get("workers"), job.get("share-rnd", False))
    request = study_request(job["case"], job["seed"], job["steps"], case, specs,
                            stage=job.get("stage", "development"), checkpoints=job.get("checkpoints", 10),
                            eval_episodes=job.get("eval-episodes", 100),
                            validation_episodes=job.get("validation-episodes", 10),
                            share_rnd=job.get("share-rnd", False), serial_workers=job.get("serial-workers", False),
                            final_seed=job.get("final-seed"))
    return {"job": job, "output": output, "library": library, "manifest_path": manifest_path,
            "manifest": manifest, "override_path": override_path, "overrides": overrides, "request": request}


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"saved run does not match requested {label}")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_completed(plan):
    """Resume only an identical measurement; never retrofit old run provenance."""
    output, expected = plan["output"], plan["request"]
    data = json.loads((output / "final.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    require_equal(metadata["library_sha256"], plan["manifest"]["library_sha256"], "library hash")
    require_equal(Path(metadata["library"]).resolve(), plan["library"], "library path")
    require_equal(metadata["overrides"], plan["overrides"], "overrides")
    history = read_jsonl(output / "progress_history.jsonl")
    if "request" in metadata:
        actual = metadata["request"]
    else:
        # The first baseline jobs predate request/source hashes. Recover only
        # identities actually recorded in their logs, without altering them.
        actual = {"case": metadata["case"], "seed": metadata["seed"],
                  "steps": metadata["requested_total_steps"], "worker_count": metadata["worker_count"],
                  "configuration": metadata["configuration"], "effective_agents": metadata["effective_agents"],
                  "stage": metadata["study_stage"], "checkpoints": len(history),
                  "eval_episodes": metadata["final_test_episodes"], "validation_episodes": metadata["validation_episodes"],
                  "share_rnd": metadata["share_rnd"], "serial_workers": not metadata["concurrent_workers"],
                  "validation_seed": metadata["validation_seed"], "final_seed": metadata["final_test_seed"]}
    require_equal(actual, expected, "measurement identity/configuration")
    require_equal(data["status"], "complete", "completion status")
    for field, key in (("case", "case"), ("seed", "seed"), ("study_stage", "stage"), ("actual_total_steps", "steps")):
        require_equal(data[field], expected[key], field)
    n, steps, checkpoints = expected["worker_count"], expected["steps"], expected["checkpoints"]
    require_equal(len(data["workers"]), n, "worker count")
    for index, worker in enumerate(data["workers"]):
        require_equal((worker["worker"], worker["algorithm"], worker["steps"]),
                      (index, expected["effective_agents"][index]["algorithm"], steps // n), "worker budget")
    require_equal(len(history), checkpoints, "checkpoint count")
    for index, checkpoint in enumerate(history, 1):
        require_equal((checkpoint["checkpoint"], checkpoint["aggregate_steps"]),
                      (index, index * steps // checkpoints), "checkpoint schedule")
    require_equal(len(data["test"]), n, "final evaluation worker count")
    final_split = "test" if expected["stage"] == "confirmation" else "development"
    for index, result in enumerate(data["test"]):
        require_equal((result["worker"], result["algorithm"], result["aggregate_steps"], result["split"],
                       result["seed_start"], result["episodes"], result["deterministic"], result["reward"]),
                      (index, expected["effective_agents"][index]["algorithm"], steps, final_split,
                       expected["final_seed"], expected["eval_episodes"], True, "raw"), "final evaluation protocol")
        require_equal((len(result["returns"]), len(result["lengths"])),
                      (expected["eval_episodes"], expected["eval_episodes"]), "final episode count")
    evaluations = read_jsonl(output / "evaluations.jsonl")
    require_equal(len(evaluations), n * (checkpoints + 2), "evaluation record count")
    for point in range(checkpoints + 1):
        for index in range(n):
            result = evaluations[point * n + index]
            require_equal((result["worker"], result["split"], result["aggregate_steps"],
                           result["seed_start"], result["episodes"], len(result["returns"]), len(result["lengths"]),
                           result["deterministic"], result["reward"]),
                          (index, "validation", point * steps // checkpoints, expected["validation_seed"],
                           expected["validation_episodes"], expected["validation_episodes"],
                           expected["validation_episodes"], True, "raw"), "validation protocol")
    require_equal(evaluations[-n:], data["test"], "final raw evaluation records")
    return {"output": str(output), "status": "cached",
            "source_hashes_recorded": "runner_source_sha256" in metadata}


def validate_outputs(plans):
    occupied = []
    for plan in plans:
        output = plan["output"]
        for path in (output, output.with_suffix(".launch.json"), output.with_suffix(".log")):
            for previous in occupied:
                if path == previous or path in previous.parents or previous in path.parents:
                    raise ValueError(f"job output paths overlap: {path} and {previous}")
            occupied.append(path)
    for plan in plans:
        output = plan["output"]
        if (output / "final.json").exists():
            plan["cached"] = verify_completed(plan)
        elif output.exists() or output.with_suffix(".launch.json").exists() or output.with_suffix(".log").exists():
            raise RuntimeError(f"incomplete output exists; archive explicitly before retry: {output}")


def command_for(plan):
    job = plan["job"]
    command = [sys.executable, str(ROOT / "benchmarks/improvement_run.py"),
               "--case", job["case"], "--seed", str(job["seed"]), "--steps", str(job["steps"]),
               "--output", str(plan["output"]), "--build-manifest", str(plan["manifest_path"]),
               "--library", str(plan["library"]), "--stage", job.get("stage", "development")]
    if plan["override_path"] is not None:
        command += ["--overrides", str(plan["override_path"])]
    for name in ("workers", "eval-episodes", "validation-episodes", "checkpoints", "final-seed"):
        if name in job:
            command += ["--" + name, str(job[name])]
    for flag in ("share-rnd", "serial-workers"):
        if job.get(flag, False):
            command += ["--" + flag]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if args.parallel <= 0 or not isinstance(jobs, list) or not jobs:
        parser.error("positive --parallel and a nonempty job list are required")
    plans = [prepare_job(job) for job in jobs]
    validate_outputs(plans)

    def execute(plan):
        if "cached" in plan:
            return plan["cached"]
        job, output = plan["job"], plan["output"]
        final = output / "final.json"
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        env["LIBTORCH"] = str(ROOT / "target/debug/build/torch-sys-c1854a431246b133/out/libtorch/libtorch")
        env["DYLD_LIBRARY_PATH"] = env["LIBTORCH"] + "/lib"
        env["LD_LIBRARY_PATH"] = env["DYLD_LIBRARY_PATH"]
        env["REINFORCEX_LIB"] = str(plan["library"])
        command = command_for(plan)
        output.parent.mkdir(parents=True, exist_ok=True)
        record = {"job": job, "command": command, "start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "source_sha256_at_launch": {name: hashlib.sha256((ROOT / "benchmarks" / name).read_bytes()).hexdigest()
                      for name in ("improvement_run.py", "improvement_configs.py", "improvement_campaign.py")}}
        with output.with_suffix(".launch.json").open("x") as file:
            file.write(json.dumps(record, indent=2) + "\n")
        with output.with_suffix(".log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            output.with_suffix(".launch.json").write_text(json.dumps(record, indent=2) + "\n")
            code = process.wait()
        result = {"output": str(output), "exit_code": code, "status": "failed"}
        if code == 0 and final.exists():
            try:
                verify_completed(plan)
                result["status"] = "complete"
            except (ValueError, KeyError, OSError) as error:
                result["verification_error"] = str(error)
        with LOCK:
            print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(execute, plans))
    args.jobs.with_suffix(".results.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
