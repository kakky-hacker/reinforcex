"""Run a checked-in benchmark job manifest, preserving logs and failed jobs.

Completed runs may be skipped; partial training is never resumed from a weights-
only checkpoint. A rerun requires a different output directory/manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(__file__).resolve().parents[1]
    def run(job):
        output = root / job["output"]
        if (output / "final.json").exists():
            return {"id": job["id"], "status": "already_complete"}
        if output.exists():
            return {"id": job["id"], "status": "incomplete_directory_requires_new_run"}
        output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"starting": job["id"], "time": time.time()}), flush=True)
        env = os.environ.copy()
        env.update(job.get("environment", {}))
        with output.with_suffix(".log").open("w") as log:
            result = subprocess.run(job["command"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"id": job["id"], "returncode": result.returncode,
                  "status": "complete" if (output / "final.json").exists() and result.returncode == 0 else "failed"}
        print(json.dumps(record), flush=True)
        return record
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run, job) for job in manifest["jobs"]]):
            results.append(future.result())
            args.manifest.with_suffix(".status.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r["status"] not in ("complete", "already_complete") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
