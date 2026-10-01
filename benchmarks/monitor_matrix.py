"""Preserve native progress/statistics snapshots without modifying a running job."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    seen = {}
    with (args.root / "progress_history.jsonl").open("a", buffering=1) as log:
        while True:
            for path in (args.root / "runs/native").glob("*/*/progress.json"):
                try:
                    item = json.loads(path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                key = str(path.parent.relative_to(args.root))
                if seen.get(key) != item["aggregate_steps"]:
                    log.write(json.dumps({"run": key, "observed_at": time.time(), **item}, allow_nan=False) + "\n")
                    seen[key] = item["aggregate_steps"]
            finished = []
            for name in ("native", "sb3"):
                manifest = args.root / f"{name}_manifest.json"
                status = args.root / f"{name}_manifest.status.json"
                try:
                    finished.append(len(json.loads(manifest.read_text())["jobs"]) == len(json.loads(status.read_text())))
                except (FileNotFoundError, json.JSONDecodeError):
                    finished.append(False)
            if all(finished):
                break
            time.sleep(3)


if __name__ == "__main__":
    main()
