"""Provenance for v0.2 runs: source and output hashes, machine/package info, and the confirmatory-run lock.

`confirmatory_lock(addendum)` refuses to run unless the experiment's addendum and all source files are committed and
the working tree is clean, so a confirmatory result can always be traced to the exact preregistered code.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("src/apm", "src/apm_research", "src/apm_testbeds", "experiments/F_behavioral_learning")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    out = {}
    for d in SOURCE_DIRS:
        for f in sorted((ROOT / d).rglob("*.py")):
            out[f.relative_to(ROOT).as_posix()] = sha256(f)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "machine": platform.machine(), "cpus": os.cpu_count(), "commit": git("rev-parse", "HEAD"),
                            "dirty": bool(git("status", "--porcelain")), "start_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name in ("numpy", "torch", "mujoco", "onnxruntime"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            info[name] = None
    return info


def confirmatory_lock(addendum: str | Path) -> dict[str, Any]:
    """Raise unless the addendum is committed and the tree is clean; returns the provenance record."""
    path = Path(addendum)
    rel = path.resolve().relative_to(ROOT).as_posix()
    if not path.exists() or not git("ls-files", rel):
        raise RuntimeError(f"confirmatory run refused: addendum {rel} is not committed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("confirmatory run refused: uncommitted changes to tracked files")
    return {**environment(), "addendum": rel, "addendum_sha256": sha256(path), "sources": source_hashes()}


def write_json(path: str | Path, data: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    return sha256(path)
