"""Real-campaign relocation tests; no training, imports of ML libraries or FFI."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import prepare_reproduction as subject


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "original"
        self.source.mkdir()
        original = subject.ROOT / subject.PREFIX
        for name in subject.COPY_FILES:
            shutil.copyfile(original / name, self.source / name)
        shutil.copytree(original / "configs", self.source / "configs")
        for backend in subject.BACKENDS:
            shutil.copyfile(original / f"{backend}_manifest.json", self.source / f"{backend}_manifest.json")
        self.checkout = self.base / "other checkout"
        hashes = {}
        for name in subject.COPY_FILES[:2]:
            hashes.update(json.loads((self.source / name).read_text())["files"])
        for backend in subject.BACKENDS:
            hashes.update(json.loads((self.source / f"{backend}_manifest.json").read_text()).get("frozen_sha256", {}))
        for name in hashes:
            target = self.checkout / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(subject.ROOT / name, target)
        self.python_target = self.base / "base-python"
        self.python_target.write_text("#!/bin/sh\nexit 99\n")
        self.python_target.chmod(0o700)
        self.python = self.base / "venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.symlink_to(self.python_target)
        self.library = self.base / "native library.so"
        self.library.write_bytes(b"test-only nonexecutable library")
        self.libtorch = self.base / "libtorch libraries"
        self.libtorch.mkdir()
        self.output = self.base / "new results"
        self.args = argparse.Namespace(checkout=self.checkout, source_base=self.source,
                                       output=self.output, native_library=self.library,
                                       libtorch_dir=self.libtorch, loader_var="LD_LIBRARY_PATH")
        for backend in subject.BACKENDS:
            setattr(self.args, backend + "_python", self.python)

    def argv(self):
        values = ["--checkout", str(self.checkout), "--source-base", str(self.source),
                  "--output", str(self.output), "--native-library", str(self.library),
                  "--libtorch-dir", str(self.libtorch), "--loader-var", "LD_LIBRARY_PATH"]
        for backend in subject.BACKENDS:
            values.extend(["--" + backend.replace("_", "-") + "-python", str(self.python)])
        return values

    def source_hashes(self):
        return {str(p.relative_to(self.source)): subject.digest(p.read_bytes())
                for p in self.source.rglob("*") if p.is_file()}

    def test_99_jobs_keep_all_non_path_arguments_and_configs(self):
        before = self.source_hashes()
        output, files, provenance, launches = subject.prepare(self.args)
        self.assertFalse(output.exists())
        self.assertEqual(provenance["jobs"], 99)
        self.assertEqual(provenance["total_steps"], 119193600)
        self.assertEqual(len(launches), 4)
        for backend in subject.BACKENDS:
            old = json.loads((self.source / f"{backend}_manifest.json").read_text())
            new = json.loads(files[f"{backend}_manifest.json"])
            for a, b in zip(old["jobs"], new["jobs"]):
                for key in ("id", "case", "condition", "seed", "steps"):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(b["command"][0], str(self.python))
                self.assertNotEqual(b["command"][0], str(self.python_target))
                permitted = {0, 1, a["command"].index("--output") + 1}
                if "--config" in a["command"]:
                    permitted.add(a["command"].index("--config") + 1)
                self.assertEqual(len(a["command"]), len(b["command"]))
                for index, (left, right) in enumerate(zip(a["command"], b["command"])):
                    if index not in permitted:
                        self.assertEqual(left, right)
                env = dict(b["environment"])
                if backend == "native":
                    self.assertEqual(env.pop("LD_LIBRARY_PATH"), str(self.libtorch.resolve()))
                    self.assertEqual(env.pop("DYLD_LIBRARY_PATH"), "")
                    self.assertEqual(env.pop("REINFORCEX_LIB"), str(self.library.resolve()))
                    expected = {k: v for k, v in a["environment"].items()
                                if k not in ("DYLD_LIBRARY_PATH", "REINFORCEX_LIB")}
                    self.assertEqual(env, expected)
                else:
                    self.assertEqual(env.pop("LD_LIBRARY_PATH"), "")
                    self.assertEqual(env.pop("DYLD_LIBRARY_PATH"), "")
                    self.assertEqual(env, a["environment"])
        for name in files:
            if name.startswith("configs/") or name in subject.COPY_FILES:
                self.assertEqual(files[name], (self.source / name).read_bytes())
        self.assertEqual(before, self.source_hashes())

    def test_dry_run_then_explicit_write_never_creates_training_outputs(self):
        with redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(subject.main(self.argv()), 0)
        self.assertEqual(json.loads(capture.getvalue())["status"], "dry_run")
        self.assertFalse(self.output.exists())
        before = self.source_hashes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(subject.main(self.argv() + ["--write"]), 0)
        self.assertTrue((self.output / "reproduction.json").is_file())
        self.assertFalse((self.output / "runs").exists())
        self.assertEqual(before, self.source_hashes())
        # No implicit launch: the dummy executable exits 99 if invoked.
        self.assertEqual(len(list(self.output.glob("*_manifest.json"))), 4)

    def test_existing_empty_output_and_broken_symlink_are_refused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "existing output"):
            subject.prepare(self.args)
        self.output.rmdir()
        self.output.symlink_to(self.base / "missing")
        with self.assertRaisesRegex(ValueError, "existing output"):
            subject.prepare(self.args)

    def test_output_inside_original_campaign_is_refused(self):
        self.args.output = self.source / "new"
        with self.assertRaisesRegex(ValueError, "outside the original"):
            subject.prepare(self.args)

    def test_changed_checkout_is_refused_with_specific_hash_difference(self):
        path = self.checkout / "benchmarks/native_configs.py"
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "native_configs.py.*expected_sha256.*checkout_sha256"):
            subject.prepare(self.args)
        self.assertFalse(self.output.exists())

    def test_manifest_output_or_budget_corruption_is_refused(self):
        path = self.source / "native_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["jobs"][0]["output"] = "../existing-results"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected original output"):
            subject.prepare(self.args)
        manifest = json.loads((subject.ROOT / subject.PREFIX / "native_manifest.json").read_text())
        manifest["jobs"][0]["steps"] = 1
        manifest["total_steps"] = sum(j["steps"] for j in manifest["jobs"])
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected fixed budget"):
            subject.prepare(self.args)

    def test_missing_runtime_path_is_refused_without_write(self):
        self.args.sb3_python = self.base / "missing-python"
        with self.assertRaisesRegex(ValueError, "Python executable"):
            subject.prepare(self.args)
        self.assertFalse(self.output.exists())

    def test_consistently_rewritten_traversal_condition_is_refused(self):
        path = self.source / "native_manifest.json"
        manifest = json.loads(path.read_text())
        job = manifest["jobs"][0]
        job["condition"] = "../../../../escaped"
        job["id"] = f"native/{job['condition']}/{job['seed']}"
        job["output"] = str(subject.PREFIX / "runs/native" / job["condition"] / f"seed_{job['seed']}")
        job["command"][job["command"].index("--output") + 1] = job["output"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "single ASCII path component"):
            subject.prepare(self.args)
        self.assertFalse(self.output.exists())

    def test_reference_jobs_cannot_inherit_native_libtorch_loader_paths(self):
        _, files, _, launches = subject.prepare(self.args)
        for launch in launches:
            self.assertEqual(launch["command"][:5], ["/usr/bin/env", "-u", "DYLD_LIBRARY_PATH",
                                                      "-u", "LD_LIBRARY_PATH"])
        for backend in ("sb3", "tianshou", "tianshou_dqn"):
            for job in json.loads(files[f"{backend}_manifest.json"])["jobs"]:
                inherited = {"DYLD_LIBRARY_PATH": "/old/native/torch2.7", "LD_LIBRARY_PATH": "/old/native/torch2.7"}
                inherited.update(job["environment"])
                self.assertEqual(inherited["DYLD_LIBRARY_PATH"], "")
                self.assertEqual(inherited["LD_LIBRARY_PATH"], "")


if __name__ == "__main__":
    unittest.main()
