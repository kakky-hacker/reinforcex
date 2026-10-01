"""Finalization safety tests: only temporary files and mocked subprocesses.

No learner, native library, model inference or production report is invoked.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks import finalize_results as finalizer


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "reports/oss_benchmarks"
        self.base.mkdir(parents=True)
        for name in [f"{b}_manifest.json" for b in finalizer.EXPECTED_RUNS]:
            self.write(name, finalizer.read(finalizer.BASE / name))
        for config in (finalizer.BASE / "configs").glob("*.json"):
            self.write("configs/" + config.name, finalizer.read(config))
        self.manifests = finalizer.load_campaign(self.base)
        self.expected = finalizer.coverage(self.base, self.root, self.manifests)
        self.states = {backend: {"completed": count, "expected": count, "pending": [], "failures": []}
                       for backend, count in finalizer.EXPECTED_RUNS.items()}
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))

    def write(self, name, data):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def complete_training(self, omit_backend=None):
        for backend, manifest in self.manifests.items():
            records = []
            for job in manifest["jobs"]:
                if backend == omit_backend:
                    continue
                path = self.root / job["output"]
                path.mkdir(parents=True, exist_ok=True)
                for name in ("metadata.json", "final.json"):
                    (path / name).write_text('{"status":"complete"}')
                records.append({"id": job["id"], "status": "complete"})
            self.write(f"{backend}_manifest.status.json", records)

    def valid_artifacts(self):
        for name, backends in (("results_audit.json", ("native", "sb3")),
                               ("tianshou_audit.json", ("tianshou",)),
                               ("tianshou_dqn_audit.json", ("tianshou_dqn",))):
            self.write(name, {"status": "passed", "runs": [
                {"id": j["id"], "status": "passed", "errors": []}
                for b in backends for j in self.manifests[b]["jobs"]]})
        self.write("rnd_posthoc_audit.json", {
            "passed": True, "complete": True, "require_complete": True,
            "completed_runs_audited": 12, "modules_audited": 15,
            "results": [{"run": run, "status": "audited", "errors": [], "modules": [
                {"owner_worker": i, "target_unchanged": True, "predictor_changed": True} for i in owners]}
                for run, owners in sorted(self.expected["rnd"].items())]})
        models = [dict(zip(("backend", "condition", "seed", "worker"), key))
                  for key in sorted(self.expected["main_models"])]
        self.write("checkpoint_reload/results.json", {
            "protocol": {"episodes_per_worker": 10, "training_allowed": False},
            "workers": [{**row, "status": "passed", "evaluated_episode_count": 10,
                         "new_updates_during_verification": 0,
                         "training_and_save_calls": {"training": 0, "save": 0},
                         "checkpoint_unchanged": True, "reference_inputs_unchanged": True,
                         "statistics_unchanged": True, "all_lengths_match": True,
                         "episodes": [{"seed": seed, "seed_matches": True, "length_matches": True,
                                       "return_matches": True} for seed in range(900000, 900010)]}
                        for row in models]})
        for backend in ("tianshou", "tianshou_dqn"):
            self.write(f"{backend}_checkpoint_audit.json", {"status": "passed", "runs": [
                {"id": j["id"], "seed": j["seed"], "status": "passed", "episodes": 100, "mismatches": []}
                for j in self.manifests[backend]["jobs"]]})
        run_rows = [dict(zip(("backend", "condition", "seed"), key))
                    for key in sorted(self.expected["all_runs"])]
        self.write("evaluation_distribution.json", {
            "errors": [], "runs": [{**row, "status": "complete", "errors": []} for row in run_rows],
            "workers": [{**dict(zip(("backend", "condition", "seed", "worker"), key)),
                         "status": "complete", "errors": [], "episodes": 100, "evaluation_seed_start": 900000,
                         "evaluation_returns": [0.0] * 100, "evaluation_lengths": [1] * 100}
                        for key in sorted(self.expected["models"])]})
        self.write("summary.json", {"errors": [], "runs": [
            {**row, "valid_final": True} for row in run_rows if row["backend"] in ("native", "sb3")],
            "groups": [{"complete": True, "n": 3} for _ in range(33)]})
        self.write("stability.json", {"workers": [
            {**row, "diagnostics_status": "complete", "final_test_valid": True} for row in models]})

    def call_main(self, args):
        with patch.object(finalizer, "BASE", self.base), patch.object(finalizer, "ROOT", self.root):
            return finalizer.main(args)

    def run_finalizer(self, side_effect=None):
        with patch.object(finalizer.subprocess, "run", return_value=SimpleNamespace(returncode=0),
                          side_effect=side_effect) as subprocess_run:
            code = finalizer.finalize(self.base, self.root, self.states, {})
        return code, subprocess_run

    def test_93_completed_is_pending_and_starts_no_analysis(self):
        self.complete_training(omit_backend="tianshou_dqn")
        with patch.object(finalizer.subprocess, "run") as run:
            self.assertEqual(self.call_main([]), 2)
        run.assert_not_called()
        report = finalizer.read(self.base / "finalization.json")
        self.assertEqual(report["status"], "waiting")
        self.assertEqual(sum(s["completed"] for s in report["campaign"].values()), 93)
        self.assertEqual(len(report["campaign"]["tianshou_dqn"]["pending"]), 6)

    def test_wait_does_not_analyze_while_pending(self):
        self.complete_training(omit_backend="tianshou_dqn")
        class StopTestWait(Exception):
            pass
        with patch.object(finalizer.subprocess, "run") as run, \
                patch.object(finalizer.time, "sleep", side_effect=StopTestWait):
            with self.assertRaises(StopTestWait):
                self.call_main(["--wait"])
        run.assert_not_called()
        self.assertEqual(finalizer.read(self.base / "finalization.json")["status"], "waiting")

    def test_launcher_complete_without_final_is_pending(self):
        self.complete_training()
        job = self.manifests["native"]["jobs"][0]
        (self.root / job["output"] / "final.json").unlink()
        state = finalizer.campaign_status(self.base, self.root)
        self.assertEqual(state["native"]["completed"], 59)
        self.assertEqual(state["native"]["pending"], [job["id"]])

    def test_invalid_preflight_cannot_leave_a_stale_pass(self):
        self.write("finalization.json", {"status": "passed"})
        self.write("tianshou_dqn_manifest.json", {"jobs": [], "total_steps": 0})
        with patch.object(finalizer.subprocess, "run") as run:
            self.assertEqual(self.call_main([]), 1)
        run.assert_not_called()
        self.assertEqual(finalizer.read(self.base / "finalization.json")["status"], "failed")

    def test_failed_launcher_blocks_analysis_even_with_final_artifacts(self):
        self.complete_training()
        records = finalizer.read(self.base / "native_manifest.status.json")
        records[0]["status"] = "failed"
        self.write("native_manifest.status.json", records)
        with patch.object(finalizer.subprocess, "run") as run:
            self.assertEqual(self.call_main([]), 1)
        run.assert_not_called()
        self.assertEqual(finalizer.read(self.base / "finalization.json")["status"], "failed")

    def test_complete_artifacts_pass_with_all_coverage_checks(self):
        self.valid_artifacts()
        code, run = self.run_finalizer()
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 14)
        report = finalizer.read(self.base / "finalization.json")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["validations"]["reload"]["passed_models"], 132)
        self.assertEqual(report["validations"]["rnd"]["audited_modules"], 15)
        self.assertEqual(report["validations"]["distribution"]["evaluation_episodes"], 14100)

    def test_zero_exit_with_one_pending_reload_cannot_pass(self):
        self.valid_artifacts()
        data = finalizer.read(self.base / "checkpoint_reload/results.json")
        data["workers"][-1]["status"] = "pending"
        self.write("checkpoint_reload/results.json", data)
        code, run = self.run_finalizer()
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 5)
        self.assertEqual(finalizer.read(self.base / "finalization.json")["status"], "failed")

    def test_rnd_permissive_pass_and_missing_module_both_rejected(self):
        self.valid_artifacts()
        path = "rnd_posthoc_audit.json"
        data = finalizer.read(self.base / path)
        data.update(complete=False, pending_runs=["one pending run"])
        self.write(path, data)
        with self.assertRaisesRegex(ValueError, "must be complete"):
            finalizer.validate_artifact("rnd", self.base, self.expected, self.manifests)
        data.update(complete=True, pending_runs=[])
        data["results"][0]["modules"].pop()
        self.write(path, data)
        with self.assertRaisesRegex(ValueError, "RND modules"):
            finalizer.validate_artifact("rnd", self.base, self.expected, self.manifests)

    def test_same_model_count_with_duplicate_identity_is_rejected(self):
        self.valid_artifacts()
        data = finalizer.read(self.base / "checkpoint_reload/results.json")
        data["workers"][-1] = data["workers"][0]
        self.write("checkpoint_reload/results.json", data)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            finalizer.validate_artifact("reload", self.base, self.expected, self.manifests)

    def test_141_models_with_one_missing_evaluation_episode_is_rejected(self):
        self.valid_artifacts()
        data = finalizer.read(self.base / "evaluation_distribution.json")
        data["workers"][-1]["evaluation_returns"].pop()
        self.write("evaluation_distribution.json", data)
        with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
            finalizer.validate_artifact("distribution", self.base, self.expected, self.manifests)

    def test_stability_discovery_error_is_not_hidden_by_complete_rows(self):
        self.valid_artifacts()
        data = finalizer.read(self.base / "stability.json")
        data["errors"] = ["duplicate discovered run"]
        self.write("stability.json", data)
        with self.assertRaisesRegex(ValueError, "discovery errors"):
            finalizer.validate_artifact("stability", self.base, self.expected, self.manifests)

    def test_mutation_during_render_is_detected_by_final_recheck(self):
        self.valid_artifacts()
        def execute(command, **kwargs):
            if command[1].endswith("render_overview.py"):
                data = finalizer.read(self.base / "checkpoint_reload/results.json")
                data["workers"][-1]["status"] = "pending"
                self.write("checkpoint_reload/results.json", data)
            return SimpleNamespace(returncode=0)
        code, run = self.run_finalizer(execute)
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 14)
        self.assertEqual(finalizer.read(self.base / "finalization.json")["status"], "failed")

    def test_plan_is_serial_analysis_and_isolates_native_loader(self):
        with patch.dict(finalizer.os.environ, {"DYLD_LIBRARY_PATH": "contaminated", "LD_LIBRARY_PATH": "contaminated"}):
            plan = finalizer.command_plan(self.manifests)
        scripts = [Path(step["command"][1]).stem for step in plan]
        self.assertEqual(len(scripts), len(set(scripts)))
        self.assertTrue({"audit_tianshou_dqn", "summarize_tianshou_dqn", "verify_checkpoint_reload",
                         "analyze_evaluation_distribution", "render_rnd_report"}.issubset(scripts))
        self.assertFalse(any(s.startswith("run_") for s in scripts))
        for step in plan:
            if Path(step["command"][1]).stem == "audit_rnd":
                self.assertIn("--require-complete", step["command"])
                self.assertEqual(step["command"][0], self.manifests["native"]["jobs"][0]["command"][0])
            else:
                self.assertNotIn("DYLD_LIBRARY_PATH", step["environment"])
                self.assertNotIn("LD_LIBRARY_PATH", step["environment"])


if __name__ == "__main__":
    unittest.main()
