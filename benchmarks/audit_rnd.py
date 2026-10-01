"""Read-only RND checkpoint and recorded-curiosity audit for completed runs.

Reconstruct initial RND tensors from the recorded seed/config and the exact same
FFI binary. The native manifest and exported configs enumerate expected runs,
including runs without metadata. Pending runs are allowed unless
--require-complete is set; mismatched libraries or invalid checkpoints always
fail. No learner, checkpoint, core source, or running benchmark is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import reinforcex_ffi as rx


def digest(path):
    with zipfile.ZipFile(path) as archive:
        payloads = sorted(archive.read(name) for name in archive.namelist() if "/data/" in name)
    if not payloads:
        raise ValueError(f"no tensor payloads in {path}")
    return {"sha256": hashlib.sha256(b"".join(payloads)).hexdigest(),
            "tensor_count": len(payloads), "tensor_bytes": sum(map(len, payloads))}


def history(root):
    entries, malformed = {}, 0
    path = root / "progress_history.jsonl"
    if not path.exists():
        return entries, malformed
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
            entries.setdefault(item["run"], {})[item["aggregate_steps"]] = item
        except (json.JSONDecodeError, KeyError, TypeError):
            malformed += 1
            continue
    return entries, malformed


def rnd_workers(specs):
    return [i for i, spec in enumerate(specs) if spec.get("rnd_config") is not None]


def manifest_expectations(root, manifest_path):
    """Enumerate RND runs before metadata exists, using exported configs + CLI topology."""
    manifest = json.loads(manifest_path.read_text())
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--share-rnd", action="store_true")
    expected = {}
    for job in manifest["jobs"]:
        settings = json.loads((root / "configs" / f"{job['case']}.json").read_text())
        specs = settings["agents"]
        options, _ = parser.parse_known_args(job.get("command", []))
        if options.workers is not None:
            if len(specs) != 1 or options.workers <= 0:
                raise ValueError(f"invalid worker override in {job['id']}")
            specs = specs * options.workers
        workers = rnd_workers(specs)
        if options.share_rnd and len(workers) != len(specs):
            raise ValueError(f"shared RND requires RND on every worker: {job['id']}")
        if not workers:
            continue
        run = Path(job["output"])
        run = (ROOT / run).resolve() if not run.is_absolute() else run.resolve()
        key = str(run.relative_to(root))
        if key in expected:
            raise ValueError(f"duplicate RND run in manifest: {key}")
        expected[key] = {"case": job["case"], "seed": job["seed"], "steps": job["steps"],
                         "workers": workers, "worker_count": len(specs), "share_rnd": options.share_rnd,
                         "rnd_configs": {i: specs[i]["rnd_config"] for i in workers},
                         "module_count": 1 if options.share_rnd else len(workers)}
    return expected


def initial_digests(lib, settings, seed):
    rnd_config = rx.RxRndConfig()
    for name, value in settings.items():
        setattr(rnd_config, name, value)
    with tempfile.TemporaryDirectory(prefix="reinforcex-rnd-audit-") as temp:
        rx.manual_seed(lib, seed)
        rnd = rx.create_rnd(lib, rnd_config, temp, None)
        try:
            rnd.save()
        finally:
            rnd.close()
        return {name: digest(Path(temp) / name) for name in ("rnd_target.ot", "rnd_predictor.ot")}


def audit_run(root, key, expected, observed, lib, library_hash):
    run = root / key
    result = {"run": key, "expected_from_manifest": expected is not None,
              "expected_modules": expected["module_count"] if expected else None}
    if (run / "failure.json").exists():
        return {**result, "status": "run_failed", "errors": ["training failure.json exists"]}
    if not (run / "metadata.json").exists():
        if (run / "final.json").exists():
            return {**result, "status": "audit_error", "errors": ["final exists without metadata"]}
        return {**result, "status": "not_started"}
    try:
        metadata = json.loads((run / "metadata.json").read_text())
        if metadata["library_sha256"] != library_hash:
            return {**result, "status": "library_mismatch",
                    "expected_sha256": metadata["library_sha256"], "actual_sha256": library_hash,
                    "errors": ["loaded library differs from training library"]}
        specs = metadata["effective_agents"]
        workers = rnd_workers(specs)
        if expected:
            if (workers != expected["workers"] or len(specs) != expected["worker_count"]
                    or metadata["share_rnd"] != expected["share_rnd"]
                    or metadata["case"] != expected["case"] or metadata["seed"] != expected["seed"]
                    or metadata["requested_total_steps"] != expected["steps"]
                    or any(specs[i]["rnd_config"] != expected["rnd_configs"][i] for i in workers)):
                return {**result, "status": "manifest_mismatch",
                        "errors": ["recorded run topology/config/seed/budget differs from manifest"]}
        if not workers:
            return {**result, "status": "audit_error", "errors": ["RND run has no RND workers"]}
        final_path = run / "final.json"
        if not final_path.exists():
            return {**result, "status": "in_progress"}
        final = json.loads(final_path.read_text())
        if (final.get("status") != "complete"
                or final["actual_total_steps"] != metadata["requested_total_steps"]):
            return {**result, "status": "audit_error", "errors": ["final run is not complete at its required budget"]}
        final_workers = {item["worker"]: item for item in final["workers"]}
        if len(final_workers) != len(specs) or set(final_workers) != set(range(len(specs))):
            return {**result, "status": "audit_error", "errors": ["final worker set is incomplete or duplicated"]}
        if (len(final["workers"]) != len(specs)
                or sum(worker["steps"] for worker in final_workers.values()) != final["actual_total_steps"]):
            return {**result, "status": "audit_error", "errors": ["final worker step totals are inconsistent"]}
        checkpoints = sorted(observed.get(key, {}).values(), key=lambda item: item["aggregate_steps"])
        traces = []
        for worker in workers:
            points = []
            for checkpoint in checkpoints:
                recorded = {item["worker"]: item for item in checkpoint["workers"]}
                if worker not in recorded:
                    continue
                stats = recorded[worker]["statistics"]
                if "intrinsic_reward_mean" in stats:
                    points.append({"checkpoint": checkpoint["checkpoint"],
                                   "aggregate_steps": checkpoint["aggregate_steps"],
                                   "worker_steps": recorded[worker]["steps"],
                                   "updates": stats["updates"],
                                   "last_rollout_intrinsic_mean": stats["intrinsic_reward_mean"],
                                   "coefficient": stats["curiosity_coefficient"]})
            traces.append({"worker": worker, "points": points,
                           "missing_early_checkpoints": not points or points[0]["checkpoint"] > 1,
                           "final_statistics": final_workers[worker]["statistics"]})
        modules = []
        errors = []
        for worker in (workers[:1] if metadata["share_rnd"] else workers):
            settings = specs[worker]["rnd_config"]
            seed = metadata["worker_seeds"][worker]["rnd_seed"]
            initial = initial_digests(lib, settings, seed)
            trained = {name: digest(run / f"rnd_worker{worker}" / name) for name in initial}
            modules.append({"owner_worker": worker, "initial_seed": seed,
                            "shared_by_workers": workers if metadata["share_rnd"] else [worker],
                            "initial": initial, "trained": trained,
                            "target_unchanged": initial["rnd_target.ot"] == trained["rnd_target.ot"],
                            "predictor_changed": initial["rnd_predictor.ot"] != trained["rnd_predictor.ot"]})
            if not modules[-1]["target_unchanged"]:
                errors.append(f"worker {worker}: target changed")
            if not modules[-1]["predictor_changed"]:
                errors.append(f"worker {worker}: predictor did not change")
        return {**result, "status": "audited", "modules": modules, "worker_traces": traces,
                "errors": errors, "actual_total_steps": final["actual_total_steps"]}
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, zipfile.BadZipFile) as error:
        return {**result, "status": "audit_error", "errors": [f"{type(error).__name__}: {error}"]}


def build_report(root, lib, require_complete=False, manifest_path=None):
    root = root.resolve()
    manifest_path = manifest_path or root / "native_manifest.json"
    observed, malformed = history(root)
    library_hash = hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
    expected, errors = None, []
    if manifest_path.exists():
        try:
            expected = manifest_expectations(root, manifest_path)
        except (OSError, ValueError, KeyError, TypeError, argparse.ArgumentError) as error:
            errors.append(f"manifest: {type(error).__name__}: {error}")
    elif require_complete:
        errors.append(f"required manifest is missing: {manifest_path}")
    keys = set(expected or {})
    for path in sorted((root / "runs/native").glob("*/*/metadata.json")):
        key = str(path.parent.relative_to(root))
        try:
            if rnd_workers(json.loads(path.read_text())["effective_agents"]):
                keys.add(key)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{key}: cannot inspect metadata: {error}")
    results = [audit_run(root, key, (expected or {}).get(key), observed, lib, library_hash)
               for key in sorted(keys)]
    audited = [item for item in results if item["status"] == "audited"]
    modules = [module for item in audited for module in item["modules"]]
    failures = [{"run": item["run"], "status": item["status"], "errors": item["errors"]}
                for item in results if item.get("errors")]
    pending = [item["run"] for item in results if item["status"] in ("not_started", "in_progress")]
    unexpected = [key for key in sorted(keys) if expected is not None and key not in expected]
    expected_modules = sum(item["module_count"] for item in expected.values()) if expected is not None else None
    complete = (expected is not None and not pending and not failures and not errors and not unexpected
                and len(audited) == len(expected) and len(modules) == expected_modules)
    if require_complete and not complete:
        errors.append("required RND run/module coverage is incomplete or failed")
    report = {"library": str(lib._name), "library_sha256": library_hash,
              "manifest": str(manifest_path), "require_complete": require_complete,
              "expected_runs": len(expected) if expected is not None else None,
              "expected_modules": expected_modules, "pending_runs": pending,
              "unexpected_runs": unexpected, "failed_runs": failures, "errors": errors,
              "complete": complete, "passed": not errors and not failures,
              "completed_runs_audited": len(audited), "modules_audited": len(modules),
              "targets_unchanged": sum(m["target_unchanged"] for m in modules),
              "predictors_changed": sum(m["predictor_changed"] for m in modules),
              "malformed_history_lines_skipped": malformed, "results": results,
              "limitations": ["The history may omit early checkpoints if monitoring started late.",
                               "Reported curiosity is the latest completed rollout mean, not a run-wide average.",
                               "Novel states can raise curiosity, so a monotonic decrease is not required.",
                               "Tensor payload multisets are compared; ZIP timestamps/serialization IDs are ignored.",
                               "Pending runs are allowed by default; --require-complete requires every manifest RND run/module.",
                               "Target stability and predictor mutation do not establish a performance benefit."]}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "reports/oss_benchmarks")
    parser.add_argument("--manifest", type=Path, help="Defaults to ROOT/native_manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail unless all RND runs and modules expected by the manifest are audited successfully")
    args = parser.parse_args(argv)
    output = args.output or args.root / "rnd_posthoc_audit.json"
    report = build_report(args.root, rx.load_reinforcex(), args.require_complete, args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {report['completed_runs_audited']}/{report['expected_runs']} runs / "
          f"{report['modules_audited']}/{report['expected_modules']} RND modules: "
          f"{report['targets_unchanged']} targets unchanged, {report['predictors_changed']} predictors changed; "
          f"pending={len(report['pending_runs'])}, passed={report['passed']}, complete={report['complete']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
