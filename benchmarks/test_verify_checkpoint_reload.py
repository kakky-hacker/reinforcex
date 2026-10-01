"""Regression checks for inference-only checkpoint verification bookkeeping."""
import copy
import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import verify_checkpoint_reload as reload_check


class ReloadVerificationTests(unittest.TestCase):
    def test_return_tolerances_do_not_hide_seed_or_length_mismatches(self):
        expected = [{"seed": 900000, "return": 1000.0, "length": 100}]
        observed = [{"seed": 900001, "return": 1000.00009, "length": 99}]
        row = reload_check.compare_episodes(observed, expected, 1e-6, 1e-7)[0]
        self.assertTrue(row["return_matches"])
        self.assertFalse(row["seed_matches"])
        self.assertFalse(row["length_matches"])
        observed[0]["return"] = 1000.001
        self.assertFalse(reload_check.compare_episodes(observed, expected, 1e-6, 1e-7)[0]["return_matches"])
        with self.assertRaises(ValueError):
            reload_check.compare_episodes([], expected, 1e-6, 1e-7)

    def test_absolute_tolerance_and_nonfinite_values(self):
        expected = [{"seed": 900000, "return": 0.0, "length": 1}]
        observed = [{"seed": 900000, "return": 0.5e-6, "length": 1}]
        self.assertTrue(reload_check.compare_episodes(observed, expected, 1e-6, 1e-7)[0]["return_matches"])
        observed[0]["return"] = float("nan")
        self.assertFalse(reload_check.compare_episodes(observed, expected, 1e-6, 1e-7)[0]["return_matches"])

    def test_native_uses_first_ten_final_episodes_including_low_returns(self):
        record = {"worker": 2, "seed_start": 900000, "episodes": 100, "reward": "raw",
                  "deterministic": True, "split": "test", "returns": list(range(-10, 90)),
                  "lengths": list(range(1, 101))}
        final = {"test": [record]}
        rows = reload_check.reference_episodes("native", Path("."), final, 2, 10)
        self.assertEqual([row["return"] for row in rows], list(range(-10, 0)))
        self.assertEqual([row["seed"] for row in rows], list(range(900000, 900010)))
        for field, value in [("seed_start", 800000), ("episodes", 10), ("reward", "shaped"),
                             ("deterministic", False), ("split", "validation")]:
            invalid = copy.deepcopy(final)
            invalid["test"][0][field] = value
            with self.assertRaises(ValueError):
                reload_check.reference_episodes("native", Path("."), invalid, 2, 10)
        with self.assertRaises(ValueError):
            reload_check.reference_episodes("native", Path("."), {"test": [record, record]}, 2, 10)

    def test_sb3_uses_final_seed_prefix_not_log_order_or_validation(self):
        final = {"final_evaluation": {"seed_start": 900000, "episodes": 100,
                                      "deterministic": True, "phase": "final"}}
        records = [{"phase": "final", "seed": 900000 + i, "return": i - 10, "length": i + 1}
                   for i in reversed(range(100))]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "eval_episodes.jsonl"
            def write(rows):
                log.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            write(records + [{"phase": "validation", "seed": 800000, "return": 9999, "length": 1}])
            actual = reload_check.reference_episodes("sb3", directory, final, 0, 10)
            self.assertEqual([row["return"] for row in actual], list(range(-10, 0)))
            write(records[:-1])
            with self.assertRaises(ValueError):
                reload_check.reference_episodes("sb3", directory, final, 0, 10)
            write(records[:-1] + [records[0]])
            with self.assertRaises(ValueError):
                reload_check.reference_episodes("sb3", directory, final, 0, 10)

    def test_checkpoint_components_include_shared_rnd_target_and_predictor(self):
        paths = reload_check.checkpoint_paths("native", Path("run"), 3,
                                             {"algorithm": "ppo", "rnd_config": {}}, {"share_rnd": True})
        self.assertEqual(paths, [Path("run/worker3.ot"), Path("run/rnd_worker0/rnd_predictor.ot"),
                                 Path("run/rnd_worker0/rnd_target.ot")])
        paths = reload_check.checkpoint_paths("native", Path("run"), 1, {"algorithm": "sac"}, {})
        self.assertEqual(len(paths), 4)
        self.assertIn(Path("run/worker1_temperature.ot"), paths)

    def test_file_mutation_and_cache_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "weights.ot"
            checkpoint.write_bytes(b"first")
            before = reload_check.fingerprint_files([checkpoint])
            checkpoint.write_bytes(b"other")
            after = reload_check.fingerprint_files([checkpoint])
            self.assertNotEqual(before, after)
            request = {"checkpoint_fingerprints": before, "reference_fingerprints": {"ref": "a"},
                       "runtime_fingerprints": {"python": "a"}}
            original = reload_check.cache_fingerprint(request, "python", {}, "script")
            for field, replacement in [("checkpoint_fingerprints", after), ("reference_fingerprints", {"ref": "b"}),
                                       ("runtime_fingerprints", {"python": "b"})]:
                altered = {**request, field: replacement}
                self.assertNotEqual(original, reload_check.cache_fingerprint(altered, "python", {}, "script"))
            self.assertNotEqual(original, reload_check.cache_fingerprint(request, "python", {}, "new-script"))
            self.assertNotEqual(original, reload_check.cache_fingerprint(request, "other-python", {}, "script"))

    def test_flat_and_nested_ctypes_configuration_restoration(self):
        class Base(ctypes.Structure):
            _fields_ = [("width", ctypes.c_int), ("discount", ctypes.c_float)]
        class Wrapped(ctypes.Structure):
            _anonymous_ = ("base",)
            _fields_ = [("base", Base), ("target_interval", ctypes.c_int)]
        config = reload_check.assign_structure(Wrapped(), {"width": 64, "discount": 0.99, "target_interval": 8})
        self.assertEqual(config.width, 64)
        self.assertAlmostEqual(config.discount, 0.99)
        self.assertEqual(config.target_interval, 8)
        reload_check.assign_structure(config, {"base": {"width": 128}})
        self.assertEqual(config.width, 128)
        with self.assertRaises(ValueError):
            reload_check.assign_structure(config, {"made_up_config": 1})

    def test_native_library_drift_rejected_before_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable, library = directory / "python", directory / "libreinforcex.dylib"
            executable.write_bytes(b"interpreter")
            library.write_bytes(b"original-library")
            metadata = {"library": str(library), "library_sha256": reload_check.sha256(library)}
            files = reload_check.runtime_fingerprints(str(executable), metadata, "native")
            self.assertIn(str(library.resolve()), files)
            library.write_bytes(b"other-library")
            with self.assertRaises(ValueError):
                reload_check.runtime_fingerprints(str(executable), metadata, "native")


if __name__ == "__main__":
    unittest.main()
