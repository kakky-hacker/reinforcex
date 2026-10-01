#!/usr/bin/env python3
"""Read inexpensive progress snapshots; unfinished episode steps are omitted."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"


def last_complete_line(path):
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        start = max(0, size - 16384)
        stream.seek(start)
        data = stream.read()
    lines = data.split(b"\n")[:-1]
    if start:
        lines = lines[1:]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "backends": {},
              "step_count_note": "Lower bound: active, unfinished episodes are omitted."}
    for backend in ("native", "sb3", "tianshou", "tianshou_dqn"):
        manifest_path = BASE / f"{backend}_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        try:
            execution_status = {row["id"]: row["status"] for row in json.loads(
                manifest_path.with_suffix(".status.json").read_text())}
        except (FileNotFoundError, json.JSONDecodeError):
            execution_status = {}
        total, observed, completed, active, failures = 0, 0, 0, [], []
        for job in manifest["jobs"]:
            directory = ROOT / job["output"]
            command = job["command"]
            requested = int(command[command.index("--steps") + 1])
            total += requested
            final = directory / "final.json"
            if final.exists():
                record = json.loads(final.read_text())
                observed += record.get("actual_total_steps", record.get("actual_steps", 0))
                completed += 1
                continue
            if ((directory / "failure.json").exists()
                    or execution_status.get(job["id"]) not in (None, "complete", "already_complete")):
                failures.append(job["id"])
            if not (directory / "metadata.json").exists():
                continue
            logs = (list(directory.glob("train_worker*.jsonl")) if backend == "native"
                    else [directory / "train_episodes.jsonl"])
            steps = []
            for log in logs:
                record = last_complete_line(log)
                steps.append(record.get("steps", record.get("env_steps", 0)))
            collected = sum(steps)
            observed += collected
            active.append({"id": job["id"], "observed_steps": collected,
                           "requested_steps": requested})
        report["backends"][backend] = {
            "completed": completed, "expected": len(manifest["jobs"]), "observed_steps": observed,
            "requested_steps": total, "active": active, "failure_files": failures}
    report["observed_steps"] = sum(b["observed_steps"] for b in report["backends"].values())
    report["requested_steps"] = sum(b["requested_steps"] for b in report["backends"].values())
    report["fraction"] = report["observed_steps"] / report["requested_steps"]
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
