"""Freeze the exact source and native binary used by an improvement-study run."""
import argparse
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/core_improvements_20261001"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths():
    result = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for folder in ("core", "ffi", "examples"):
        result += [p for p in (ROOT / folder).rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix not in (".ot", ".pyc")]
    result += list((ROOT / "benchmarks").glob("*improvement*.py"))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        parser.error("name must contain letters, numbers, underscores or hyphens")
    destination = STUDY / "builds" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    files = source_paths()
    contents = {str(p.relative_to(ROOT)): p.read_bytes() for p in files}
    initial = {path: hashlib.sha256(data).hexdigest() for path, data in contents.items()}
    for path in files:
        target = destination / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[str(path.relative_to(ROOT))])
    env = dict(os.environ)
    env.update(RUSTUP_HOME="/private/tmp/reinforcex-rustup-20261001",
               CARGO_TARGET_DIR=str(ROOT / "target/core-improvements"),
               LIBTORCH=str(ROOT / "target/debug/build/torch-sys-c1854a431246b133/out/libtorch/libtorch"))
    env["DYLD_LIBRARY_PATH"] = env["LIBTORCH"] + "/lib"
    command = ["cargo", "+1.88.0", "build", "--offline", "--release", "-p", "reinforcex_ffi"]
    with (destination / "build.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    final = {str(p.relative_to(ROOT)): digest(p) for p in source_paths()}
    compiled = lambda name: Path(name).suffix in (".rs", ".toml", ".lock")
    if {p: sha for p, sha in initial.items() if compiled(p)} != {p: sha for p, sha in final.items() if compiled(p)}:
        raise RuntimeError("compiled source changed during build; do not use this artifact, build a fresh variant")
    binary = destination / "libreinforcex.dylib"
    shutil.copy2(ROOT / "target/core-improvements/release/libreinforcex.dylib", binary)
    manifest = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_sha256": initial, "library_sha256": digest(binary),
                "noncompiled_source_changed_during_build": sorted(
                    p for p in set(initial) | set(final) if not compiled(p) and initial.get(p) != final.get(p)),
                "source_snapshot_timing": "captured before build; each training run records its actual Python runner sources separately",
                "build_command": command, "rust_toolchain": "1.88.0", "libtorch": "2.7.0",
                "baseline_manifest": str((STUDY / "baseline/manifest.json").relative_to(ROOT)),
                "baseline_manifest_sha256": digest(STUDY / "baseline/manifest.json")}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    patches = []
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in ("core", "ffi", "examples"):
            continue
        before = STUDY / "baseline/source" / relative
        if path.suffix not in (".rs", ".py", ".h", ".toml"):
            continue
        patches.extend(difflib.unified_diff(
            before.read_text().splitlines(keepends=True) if before.exists() else [],
            (destination / "source" / relative).read_text().splitlines(keepends=True), fromfile="baseline/" + str(relative),
            tofile="improved/" + str(relative)))
    (destination / "changes_from_start.patch").write_text("".join(patches))
    print(json.dumps({"build": str(destination), "library_sha256": manifest["library_sha256"]}))


if __name__ == "__main__":
    main()
