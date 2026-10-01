#!/usr/bin/env python3
"""Prepare relocated copies of the fixed 99-run campaign; never start training.

The default is a read-only dry run. --write creates a new campaign directory,
including launch_commands.json. Existing directories (even empty ones) are
rejected. Python environments and the native library must already exist.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path("reports/oss_benchmarks")
BACKENDS = {
    "native": (60, "run_native.py"),
    "sb3": (30, "run_sb3.py"),
    "tianshou": (3, "run_tianshou_discrete.py"),
    "tianshou_dqn": (6, "run_tianshou_dqn.py"),
}
COPY_FILES = ("source_snapshot_before.json", "benchmark_code_snapshot_frozen.json",
              "gymnasium_130_thresholds.json", "PROTOCOL.ja.md", "tianshou_dqn_protocol.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def option_index(command, name):
    if command.count(name) != 1:
        raise ValueError(f"Expected exactly one {name}: {command}")
    index = command.index(name) + 1
    if index == len(command):
        raise ValueError(f"Missing value for {name}")
    return index


def check_new_output(output, source):
    # lexists also rejects broken symlinks; resolve catches symlinked parents.
    if os.path.lexists(output):
        raise ValueError(f"Refusing existing output directory or symlink: {output}")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be outside the original campaign directory")


def prepare(args):
    checkout = args.checkout.expanduser().resolve(strict=True)
    source = args.source_base.expanduser().resolve(strict=True)
    output_arg = args.output.expanduser().absolute()
    if os.path.lexists(output_arg):
        raise ValueError(f"Refusing existing output directory or symlink: {output_arg}")
    output = output_arg.resolve()
    check_new_output(output, source)
    pythons = {}
    for backend in BACKENDS:
        # Do not resolve the final symlink: venv/bin/python must retain its venv.
        path = getattr(args, backend + "_python").expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Python executable does not exist or is not executable: {path}")
        pythons[backend] = str(path)
    library = args.native_library.expanduser().resolve(strict=True)
    libtorch = args.libtorch_dir.expanduser().resolve(strict=True)
    if not library.is_file() or not libtorch.is_dir():
        raise ValueError("--native-library must be a file and --libtorch-dir a directory")

    files = {name: (source / name).read_bytes() for name in COPY_FILES}
    for path in sorted((source / "configs").glob("*.json")):
        files[str(Path("configs") / path.name)] = path.read_bytes()
    if len([name for name in files if name.startswith("configs/")]) != 15:
        raise ValueError("The fixed campaign requires its 15 exported configurations")
    originals = {backend: json.loads((source / f"{backend}_manifest.json").read_text())
                 for backend in BACKENDS}
    expected_hashes = {}
    for name in COPY_FILES[:2]:
        expected_hashes.update(json.loads(files[name])["files"])
    for manifest in originals.values():
        expected_hashes.update(manifest.get("frozen_sha256", {}))
    mismatches = []
    for name, expected in expected_hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe snapshot path: {name}")
        path = checkout / relative
        actual = digest(path.read_bytes()) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": name, "expected_sha256": expected, "checkout_sha256": actual})
        # Copies of frozen campaign inputs must also match the saved hashes.
        if relative.is_relative_to(PREFIX):
            source_path = source / relative.relative_to(PREFIX)
            if not source_path.is_file() or digest(source_path.read_bytes()) != expected:
                raise ValueError(f"Frozen source campaign input changed: {source_path}")
    if mismatches:
        raise ValueError("Checkout differs from the frozen campaign; preparation refused: " +
                         json.dumps(mismatches, ensure_ascii=False))

    manifests, launches, changes, ids = {}, [], [], set()
    for backend, (count, runner) in BACKENDS.items():
        original = originals[backend]
        manifest = copy.deepcopy(original)
        jobs = manifest["jobs"]
        if len(jobs) != count or sum(job["steps"] for job in jobs) != manifest["total_steps"]:
            raise ValueError(f"Unexpected job count or total budget in {backend}")
        for field in ("protocol", "human_protocol"):
            if field in manifest:
                path = Path(manifest[field])
                manifest[field] = str(output / path.relative_to(PREFIX) if path.is_relative_to(PREFIX)
                                      else checkout / path)
        for job in jobs:
            before = copy.deepcopy(job)
            command = job["command"]
            for field in ("case", "condition"):
                if not isinstance(job[field], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", job[field]):
                    raise ValueError(f"{field} must be a single ASCII path component: {job[field]!r}")
            if command[1] != f"benchmarks/{runner}":
                raise ValueError(f"Unexpected runner for {job['id']}")
            if job["id"] != f"{backend}/{job['condition']}/{job['seed']}":
                raise ValueError(f"Job identity differs from backend/condition/seed: {job['id']}")
            if job["seed"] not in (42, 123, 2026) or job["id"] in ids:
                raise ValueError(f"Unexpected or duplicate seed/job: {job['id']}")
            ids.add(job["id"])
            budget = 204800 if job["case"].startswith("cartpole_") else (
                1024000 if job["case"].startswith("lunar_") else 2048000)
            if job["steps"] != budget or int(command[option_index(command, "--steps")]) != budget:
                raise ValueError(f"Unexpected fixed budget: {job['id']}")
            if int(command[option_index(command, "--seed")]) != job["seed"]:
                raise ValueError(f"Seed differs between manifest and command: {job['id']}")
            expected_output = PREFIX / "runs" / backend / job["condition"] / f"seed_{job['seed']}"
            if Path(job["output"]) != expected_output or command[option_index(command, "--output")] != job["output"]:
                raise ValueError(f"Unexpected original output: {job['id']}")
            target = output / expected_output.relative_to(PREFIX)
            if not target.resolve().is_relative_to(output):
                raise ValueError(f"Training output escapes the new campaign: {target}")
            job["output"] = str(target)
            command[0], command[1] = pythons[backend], str(checkout / "benchmarks" / runner)
            command[option_index(command, "--output")] = str(target)
            if backend != "native":
                index = option_index(command, "--config")
                expected_config = PREFIX / "configs" / f"{job['case']}.json"
                if Path(command[index]) != expected_config:
                    raise ValueError(f"Unexpected config path: {job['id']}")
                command[index] = str(output / expected_config.relative_to(PREFIX))
            elif command[option_index(command, "--case")] != job["case"]:
                raise ValueError(f"Native case differs: {job['id']}")
            environment = job.setdefault("environment", {})
            # run_matrix merges os.environ. Explicit empty values also protect
            # callers who invoke it directly instead of using our launch plan.
            environment["DYLD_LIBRARY_PATH"] = ""
            environment["LD_LIBRARY_PATH"] = ""
            if backend == "native":
                environment["REINFORCEX_LIB"] = str(library)
                environment[args.loader_var] = str(libtorch)
            changes.append({"id": job["id"], "original_command": before["command"],
                            "command": command, "original_environment": before.get("environment", {}),
                            "environment": environment})
        manifests[backend] = manifest
        files[f"{backend}_manifest.json"] = json_bytes(manifest)
        launches.append({"backend": backend, "cwd": str(checkout), "command": [
                         "/usr/bin/env", "-u", "DYLD_LIBRARY_PATH", "-u", "LD_LIBRARY_PATH", pythons[backend],
                         str(checkout / "benchmarks/run_matrix.py"), str(output / f"{backend}_manifest.json"),
                         "--jobs", "1" if backend.startswith("tianshou") else "4"]})
    if len(ids) != 99:
        raise ValueError("Expected exactly 99 unique jobs")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_base": str(source),
        "checkout": str(checkout), "output": str(output), "jobs": 99,
        "total_steps": sum(m["total_steps"] for m in manifests.values()),
        "original_manifest_sha256": {b: digest((source / f"{b}_manifest.json").read_bytes()) for b in BACKENDS},
        "copied_input_sha256": {n: digest(data) for n, data in files.items() if not n.endswith("_manifest.json")},
        "checkout_source_check": {"matched_files": len(expected_hashes), "mismatches": []},
        "native_library_sha256": digest(library.read_bytes()), "libtorch_directory": str(libtorch),
        "python_executables": pythons, "path_changes": changes,
        "limitations": [
            "Preparation only: no Python executable, shared library, training or evaluator is executed.",
            "Source hashes match; compiler, LibTorch binaries and installed package versions are not verified here.",
            "A different build, machine, runtime, or parallel schedule need not reproduce identical rewards.",
            "Launch commands clear inherited DYLD_LIBRARY_PATH/LD_LIBRARY_PATH; only native jobs receive the selected LibTorch directory.",
            "Native configuration is reconstructed by the unchanged runner and supplied library; exported configs are retained for audit.",
            "Run the two Tianshou launch commands sequentially; supplementary training must have at most one job in total.",
            "Tianshou audit/summary/checkpoint tools and finalizer retain original BASE assumptions; do not invoke their defaults on this new campaign.",
        ],
    }
    files["reproduction.json"] = json_bytes(provenance)
    files["launch_commands.json"] = json_bytes(launches)
    return output, files, provenance, launches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT)
    parser.add_argument("--source-base", type=Path, default=ROOT / PREFIX)
    parser.add_argument("--output", type=Path, required=True)
    for backend in BACKENDS:
        parser.add_argument("--" + backend.replace("_", "-") + "-python", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--libtorch-dir", type=Path, required=True,
                        help="Directory containing LibTorch shared libraries (typically LIBTORCH/lib)")
    parser.add_argument("--loader-var", choices=("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"),
                        default="DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    parser.add_argument("--write", action="store_true", help="Write prepared files only; never execute jobs")
    args = parser.parse_args(argv)
    try:
        output, files, provenance, launches = prepare(args)
        if args.write:
            check_new_output(output, args.source_base.expanduser().resolve())
            output.mkdir(parents=True, exist_ok=False)
            for name, data in files.items():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)
        print(json.dumps({"status": "prepared" if args.write else "dry_run", "output": str(output),
                          "jobs": 99, "files": len(files), "total_steps": provenance["total_steps"],
                          "checkout_source_check": provenance["checkout_source_check"],
                          "launch_commands": launches, "limitations": provenance["limitations"]},
                         indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
