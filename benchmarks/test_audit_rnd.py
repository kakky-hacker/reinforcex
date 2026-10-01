"""Artifact fixtures for strict RND audit completion and failure exit semantics."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("audit_rnd", Path(__file__).with_name("audit_rnd.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = self.root / "library"
        self.library.write_bytes(b"test library")
        self.lib = types.SimpleNamespace(_name=str(self.library))
        self.library_hash = hashlib.sha256(self.library.read_bytes()).hexdigest()
        self.jobs = []
        self.rnd_config = {"obs_size": 4, "feature_size": 8, "hidden_layers": 1,
                           "hidden_size": 16, "learning_rate": .001, "update_interval": 128}
        self.spec = {"algorithm": "ppo", "rnd_config": self.rnd_config}
        self.write(self.root / "configs/lunar_rnd.json", {"agents": [self.spec]})
        self.initial = {}
        for name, payload in (("rnd_target.ot", b"target"), ("rnd_predictor.ot", b"predictor")):
            path = self.root / "initial" / name
            self.tensor(path, payload)
            self.initial[name] = audit.digest(path)
        self.patcher = patch.object(audit, "initial_digests", return_value=self.initial)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def tensor(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("module/data/0", payload)

    def add_run(self, condition="lunar_rnd", workers=1, share=False, state="complete"):
        run = self.root / "runs/native" / condition / "seed_42"
        total = 1024 * workers
        command = ["python", "runner", "--workers", str(workers)]
        if share:
            command.append("--share-rnd")
        self.jobs.append({"id": condition, "case": "lunar_rnd", "seed": 42, "steps": total,
                          "output": str(run), "command": command})
        self.write(self.root / "native_manifest.json", {"jobs": self.jobs})
        if state == "not_started":
            return run
        self.write(run / "metadata.json", {
            "case": "lunar_rnd", "seed": 42, "requested_total_steps": total,
            "library_sha256": self.library_hash, "effective_agents": [self.spec] * workers,
            "share_rnd": share, "worker_seeds": [{"rnd_seed": 1000042 + 10000 * i} for i in range(workers)],
        })
        if state == "in_progress":
            return run
        self.write(run / "final.json", {"status": "complete", "actual_total_steps": total,
                                       "workers": [{"worker": i, "steps": 1024,
                                                    "statistics": {"updates": 1}} for i in range(workers)]})
        for i in range(1 if share else workers):
            self.tensor(run / f"rnd_worker{i}/rnd_target.ot", b"target")
            self.tensor(run / f"rnd_worker{i}/rnd_predictor.ot", b"trained predictor")
        return run

    def report(self, strict=False):
        return audit.build_report(self.root, self.lib, require_complete=strict)

    def exit_code(self, strict=False):
        argv = ["--root", str(self.root)] + (["--require-complete"] if strict else [])
        with patch.object(audit.rx, "load_reinforcex", return_value=self.lib), contextlib.redirect_stdout(io.StringIO()):
            return audit.main(argv)

    def test_complete_independent_and_shared_module_counts(self):
        self.add_run("independent", workers=2)
        self.add_run("shared", workers=2, share=True)
        report = self.report(True)
        self.assertTrue(report["passed"])
        self.assertTrue(report["complete"])
        self.assertEqual((report["expected_runs"], report["expected_modules"]), (2, 3))
        self.assertEqual((report["modules_audited"], report["targets_unchanged"], report["predictors_changed"]), (3, 3, 3))
        self.assertEqual(self.exit_code(True), 0)

    def test_unstarted_metadata_is_enumerated_and_strictly_required(self):
        self.add_run("done")
        self.add_run("unstarted", workers=2, state="not_started")
        report = self.report()
        self.assertEqual((report["expected_runs"], report["expected_modules"]), (2, 3))
        self.assertEqual(len(report["pending_runs"]), 1)
        self.assertEqual(report["results"][1]["status"], "not_started")
        self.assertEqual(self.exit_code(), 0)
        self.assertEqual(self.exit_code(True), 1)

    def test_started_pending_is_allowed_only_by_default(self):
        self.add_run(state="in_progress")
        self.assertEqual(self.report()["results"][0]["status"], "in_progress")
        self.assertEqual(self.exit_code(), 0)
        self.assertEqual(self.exit_code(True), 1)

    def test_library_mismatch_fails_even_for_pending_run(self):
        run = self.add_run(state="in_progress")
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["library_sha256"] = "other library"
        self.write(run / "metadata.json", metadata)
        self.assertEqual(self.report()["results"][0]["status"], "library_mismatch")
        self.assertEqual(self.exit_code(), 1)

    def test_changed_target_fails(self):
        run = self.add_run()
        self.tensor(run / "rnd_worker0/rnd_target.ot", b"changed target")
        self.assertEqual(self.exit_code(), 1)
        self.assertEqual(self.report()["targets_unchanged"], 0)

    def test_unchanged_predictor_fails(self):
        run = self.add_run()
        self.tensor(run / "rnd_worker0/rnd_predictor.ot", b"predictor")
        self.assertEqual(self.exit_code(), 1)
        self.assertEqual(self.report()["predictors_changed"], 0)

    def test_missing_module_checkpoint_fails(self):
        run = self.add_run(workers=2)
        (run / "rnd_worker1/rnd_predictor.ot").unlink()
        self.assertEqual(self.exit_code(), 1)
        self.assertEqual(self.report()["results"][0]["status"], "audit_error")

    def test_manifest_topology_mismatch_fails(self):
        run = self.add_run(workers=2)
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["share_rnd"] = True
        self.write(run / "metadata.json", metadata)
        self.assertEqual(self.exit_code(), 1)
        self.assertEqual(self.report()["results"][0]["status"], "manifest_mismatch")

    def test_missing_manifest_cannot_pass_strict_mode(self):
        self.add_run()
        (self.root / "native_manifest.json").unlink()
        self.assertEqual(self.exit_code(), 0)
        self.assertEqual(self.exit_code(True), 1)

    def test_truncated_history_does_not_hide_valid_checkpoint(self):
        self.add_run()
        (self.root / "progress_history.jsonl").write_text('{"unfinished":\n{}\n')
        report = self.report(True)
        self.assertTrue(report["passed"])
        self.assertEqual(report["malformed_history_lines_skipped"], 2)

    def test_failed_job_is_not_pending(self):
        run = self.add_run(state="in_progress")
        self.write(run / "failure.json", {"error": "training failed"})
        self.assertEqual(self.exit_code(), 1)
        self.assertEqual(self.report()["results"][0]["status"], "run_failed")

    def test_wrong_final_budget_fails(self):
        run = self.add_run()
        final = json.loads((run / "final.json").read_text())
        final["actual_total_steps"] -= 1
        self.write(run / "final.json", final)
        self.assertEqual(self.exit_code(), 1)


if __name__ == "__main__":
    unittest.main()
