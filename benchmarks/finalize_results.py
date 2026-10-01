#!/usr/bin/env python3
"""Wait for this fixed 99-run campaign, then audit, verify and render its results.

--wait performs no analysis until all four manifests and final artifacts are
complete. This script never starts, resumes or changes a training run. Its
canonical input layout remains reports/oss_benchmarks in this checkout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/oss_benchmarks"
EXPECTED_RUNS = {"native": 60, "sb3": 30, "tianshou": 3, "tianshou_dqn": 6}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_report(base, report):
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = base / "finalization.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_campaign(base):
    manifests = {backend: read(base / f"{backend}_manifest.json") for backend in EXPECTED_RUNS}
    ids, outputs = [], []
    for backend, expected in EXPECTED_RUNS.items():
        jobs = manifests[backend]["jobs"]
        require(len(jobs) == expected, f"{backend}: expected {expected} manifest jobs")
        require(sum(j["steps"] for j in jobs) == manifests[backend]["total_steps"], f"{backend}: step sum mismatch")
        ids.extend(j["id"] for j in jobs)
        outputs.extend(j["output"] for j in jobs)
    require(len(set(ids)) == len(set(outputs)) == 99, "Expected 99 unique job IDs and outputs")
    return manifests


def campaign_status(base=BASE, root=ROOT):
    manifests = load_campaign(base)
    states = {}
    for backend, manifest in manifests.items():
        try:
            records = read(base / f"{backend}_manifest.status.json")
        except (FileNotFoundError, json.JSONDecodeError):
            records = []  # A live launcher rewrites this JSON non-atomically.
        by_id = {record["id"]: record for record in records}
        failures, completed, pending = [], [], []
        for job in manifest["jobs"]:
            path = root / job["output"]
            launcher_status = by_id.get(job["id"], {}).get("status")
            try:
                metadata = read(path / "metadata.json") if (path / "metadata.json").exists() else {}
                final = read(path / "final.json") if (path / "final.json").exists() else {}
            except json.JSONDecodeError:
                pending.append(job["id"])
                continue
            if ((path / "failure.json").exists() or metadata.get("status") == "failed"
                    or launcher_status not in (None, "complete", "already_complete")):
                failures.append({"id": job["id"], "launcher_status": launcher_status})
                continue
            final_complete = final.get("status") == "complete" if backend == "native" else metadata.get("status") == "complete"
            if launcher_status in ("complete", "already_complete") and final and final_complete:
                completed.append(job["id"])
            else:
                pending.append(job["id"])
        states[backend] = {"completed": len(completed), "expected": EXPECTED_RUNS[backend],
                           "pending": pending, "failures": failures}
    return states


def coverage(base, root, manifests):
    all_runs, main_runs, models, main_models, rnd = set(), set(), set(), set(), {}
    for backend, manifest in manifests.items():
        for job in manifest["jobs"]:
            run = (backend, job["condition"], job["seed"])
            all_runs.add(run)
            specs = read(base / "configs" / f"{job['case']}.json")["agents"]
            if backend == "native":
                command = job["command"]
                if "--workers" in command:
                    specs *= int(command[command.index("--workers") + 1])
                owners = [i for i, spec in enumerate(specs) if spec.get("rnd_config") is not None]
                if owners:
                    key = str((root / job["output"]).resolve().relative_to(base.resolve()))
                    rnd[key] = {0} if "--share-rnd" in command else set(owners)
            else:
                specs = specs[:1]
            worker_ids = {(*run, i) for i in range(len(specs))}
            models.update(worker_ids)
            if backend in ("native", "sb3"):
                main_runs.add(run)
                main_models.update(worker_ids)
    require(len(main_runs) == 90 and len(all_runs) == 99, "Unexpected run coverage")
    require(len(main_models) == 132 and len(models) == 141, "Unexpected worker coverage")
    require(len(rnd) == 12 and sum(map(len, rnd.values())) == 15, "Unexpected RND module coverage")
    return {"all_runs": all_runs, "main_runs": main_runs, "models": models, "main_models": main_models, "rnd": rnd}


def exact_rows(rows, expected, fields, label):
    observed = [tuple(row[field] for field in fields) for row in rows]
    require(len(observed) == len(set(observed)) and set(observed) == expected,
            f"{label}: missing, duplicate or unexpected identities")


def validate_artifact(kind, base, expected, manifests):
    if kind in ("main_audit", "tianshou_audit", "tianshou_dqn_audit"):
        name = {"main_audit": "results_audit.json", "tianshou_audit": "tianshou_audit.json",
                "tianshou_dqn_audit": "tianshou_dqn_audit.json"}[kind]
        data = read(base / name)
        backends = ("native", "sb3") if kind == "main_audit" else (kind.removesuffix("_audit"),)
        ids = {(j["id"],) for b in backends for j in manifests[b]["jobs"]}
        require(data.get("status") == "passed", f"{kind}: audit not passed")
        require(not data.get("global_errors") and not data.get("manifest_errors"), f"{kind}: audit errors")
        exact_rows(data["runs"], ids, ("id",), kind)
        require(all(r["status"] == "passed" and not r.get("errors") for r in data["runs"]), f"{kind}: incomplete/invalid run")
        return {"passed_runs": len(ids)}
    if kind == "rnd":
        data = read(base / "rnd_posthoc_audit.json")
        require(data.get("passed") is True and data.get("complete") is True and data.get("require_complete") is True,
                "RND audit must be complete, not merely passed with pending runs")
        for field in ("errors", "pending_runs", "failed_runs", "unexpected_runs"):
            require(not data.get(field), f"RND {field} is not empty")
        exact_rows(data["results"], {(key,) for key in expected["rnd"]}, ("run",), "RND runs")
        for run in data["results"]:
            require(run["status"] == "audited" and not run.get("errors"), "RND run not audited")
            exact_rows(run["modules"], {(i,) for i in expected["rnd"][run["run"]]}, ("owner_worker",), "RND modules")
            require(all(m["target_unchanged"] is True and m["predictor_changed"] is True for m in run["modules"]),
                    "RND target/predictor invariant failed")
        require(data.get("completed_runs_audited") == 12 and data.get("modules_audited") == 15,
                "RND summary must cover 12 runs / 15 modules")
        return {"audited_runs": 12, "audited_modules": 15}
    if kind == "reload":
        data = read(base / "checkpoint_reload/results.json")
        rows = data["workers"]
        exact_rows(rows, expected["main_models"], ("backend", "condition", "seed", "worker"), "Reload models")
        require(data["protocol"].get("episodes_per_worker") == 10 and data["protocol"].get("training_allowed") is False,
                "Reload protocol must be ten inference-only episodes per model")
        for row in rows:
            require(row["status"] == "passed" and row.get("evaluated_episode_count") == 10, "Reload pending/failed/incomplete model")
            require(row.get("new_updates_during_verification") == 0 and row.get("training_and_save_calls") == {"training": 0, "save": 0},
                    "Reload performed training or saving")
            require(all(row.get(k) is True for k in ("checkpoint_unchanged", "reference_inputs_unchanged", "statistics_unchanged", "all_lengths_match")),
                    "Reload immutability check failed")
            episodes = row["episodes"]
            require(len(episodes) == 10 and [e["seed"] for e in episodes] == list(range(900000, 900010)), "Reload episode/seed mismatch")
            require(all(e.get(k) is True for e in episodes for k in ("seed_matches", "length_matches", "return_matches")),
                    "Reload episode mismatch")
        return {"passed_models": 132, "pending_models": 0, "matched_episodes": 1320}
    if kind in ("tianshou_reload", "tianshou_dqn_reload"):
        backend = kind.removesuffix("_reload")
        data = read(base / f"{backend}_checkpoint_audit.json")
        ids = {(j["seed"],) for j in manifests[backend]["jobs"]} if backend == "tianshou" else {(j["id"],) for j in manifests[backend]["jobs"]}
        exact_rows(data["runs"], ids, ("seed",) if backend == "tianshou" else ("id",), kind)
        require(data.get("status") == "passed" and all(r["status"] == "passed" and r["episodes"] == 100 and not r["mismatches"] for r in data["runs"]),
                f"{kind}: incomplete checkpoint inference verification")
        return {"passed_models": len(ids), "matched_episodes": 100 * len(ids)}
    if kind == "distribution":
        data = read(base / "evaluation_distribution.json")
        require(not data["errors"], "Distribution errors")
        exact_rows(data["runs"], expected["all_runs"], ("backend", "condition", "seed"), "Distribution runs")
        exact_rows(data["workers"], expected["models"], ("backend", "condition", "seed", "worker"), "Distribution models")
        require(all(r["status"] == "complete" and not r["errors"] for r in data["runs"]), "Distribution run incomplete")
        for row in data["workers"]:
            require(row["status"] == "complete" and not row["errors"] and row["episodes"] == 100 and row["evaluation_seed_start"] == 900000,
                    "Distribution model incomplete")
            require(len(row["evaluation_returns"]) == len(row["evaluation_lengths"]) == 100 and
                    all(math.isfinite(value) for value in row["evaluation_returns"]), "Distribution missing/nonfinite episodes")
        return {"complete_runs": 99, "complete_models": 141, "evaluation_episodes": 14100}
    if kind == "summary":
        data = read(base / "summary.json")
        exact_rows(data["runs"], expected["main_runs"], ("backend", "condition", "seed"), "Summary runs")
        require(not data["errors"] and all(r["valid_final"] for r in data["runs"]), "Summary contains invalid/incomplete runs")
        require(len(data["groups"]) == 33 and all(g["complete"] and g["n"] == 3 for g in data["groups"]), "Summary seed groups incomplete")
        return {"valid_runs": 90, "complete_groups": 33}
    if kind == "stability":
        data = read(base / "stability.json")
        require(not data.get("errors"), "Stability discovery errors")
        exact_rows(data["workers"], expected["main_models"], ("backend", "condition", "seed", "worker"), "Stability models")
        require(all(r["diagnostics_status"] == "complete" and r["final_test_valid"] for r in data["workers"]), "Stability diagnostics incomplete")
        return {"complete_models": 132}
    raise ValueError(f"Unknown artifact validation: {kind}")


def command_plan(manifests):
    clean_env = os.environ.copy()
    for name in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "REINFORCEX_LIB"):
        clean_env.pop(name, None)
    native = manifests["native"]["jobs"][0]
    native_env = {**clean_env, **native.get("environment", {})}
    native_python = native["command"][0]
    ti_python = manifests["tianshou"]["jobs"][0]["command"][0]
    plot_python = manifests["tianshou_dqn"]["jobs"][0]["command"][0]
    plan = []
    def add(script, validator=None, python=sys.executable, env=None, extra=()):
        plan.append({"command": [python, "benchmarks/" + script + ".py", *extra],
                     "validator": validator, "environment": env if env is not None else clean_env})
    add("audit_results", "main_audit")
    add("audit_tianshou", "tianshou_audit")
    add("audit_tianshou_dqn", "tianshou_dqn_audit")
    add("audit_rnd", "rnd", native_python, native_env, ("--require-complete",))
    add("verify_checkpoint_reload", "reload")
    add("verify_tianshou_checkpoints", "tianshou_reload", ti_python)
    add("verify_tianshou_dqn_checkpoints", "tianshou_dqn_reload", plot_python)
    add("analyze_evaluation_distribution", "distribution")
    add("render_oss_report", "summary", plot_python)
    add("analyze_stability", "stability")
    add("summarize_tianshou", python=plot_python)
    add("summarize_tianshou_dqn", python=plot_python)
    add("render_rnd_report", python=plot_python)
    add("render_overview", python=plot_python)
    return plan


def finalize(base, root, states, report):
    report.update(status="in_progress", commands=[], validations={})
    write_report(base, report)
    try:
        require(set(states) == set(EXPECTED_RUNS) and all(
            states[backend]["expected"] == count and states[backend]["completed"] == count
            and not states[backend]["pending"] and not states[backend]["failures"]
            for backend, count in EXPECTED_RUNS.items()), "All 99 training runs must be complete before finalizing")
        manifests = load_campaign(base)
        expected = coverage(base, root, manifests)
        plan = command_plan(manifests)
        for step in plan:
            print("Running " + " ".join(step["command"]), flush=True)
            result = subprocess.run(step["command"], cwd=root, env=step["environment"])
            report["commands"].append({"command": step["command"], "returncode": result.returncode})
            require(result.returncode == 0, f"Command failed: {step['command']}")
            if step["validator"]:
                report["validations"][step["validator"]] = validate_artifact(step["validator"], base, expected, manifests)
            write_report(base, report)
        # Recheck coverage after rendering; successful process exit alone is insufficient.
        for kind in report["validations"]:
            report["validations"][kind] = validate_artifact(kind, base, expected, manifests)
        report.update(status="passed", campaign=states, performance_claim="Integrity complete; reward-threshold results remain in the performance reports.")
        write_report(base, report)
        print("All 99 runs audited; 132 native/SB3 reloads, 15 RND modules and 14,100 final evaluation episodes verified.", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(base, report)
        print(report["error"], file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    report = {"status": "waiting", "expected_runs": 99,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    previous = None
    while True:
        try:
            states = campaign_status(BASE, ROOT)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.update(status="failed", error=f"Campaign preflight: {type(error).__name__}: {error}")
            write_report(BASE, report)
            print(report["error"], file=sys.stderr, flush=True)
            return 1
        report["campaign"] = states
        if states != previous:
            print(json.dumps(states), flush=True)
            write_report(BASE, report)
            previous = states
        if any(state["failures"] for state in states.values()):
            report.update(status="failed", error="A training job failed; inspect its log before finalizing.")
            write_report(BASE, report)
            return 1
        if all(state["completed"] == state["expected"] for state in states.values()):
            return finalize(BASE, ROOT, states, report)
        if not args.wait:
            print("Training is incomplete; use --wait or return after all four matrices finish.", file=sys.stderr)
            return 2
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
