"""Programme H provenance: F/G's confirmatory lock, plus H source hashes and the NERVA checkout's state."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from apm_research import provenance as F
from apm_research.provenance import environment, sha256, write_json  # noqa: F401  (re-exported)

SOURCE_DIRS = ("src/apm", "src/apm_research", "src/apm_testbeds", "src/apm_select", "src/apm_express",
               "experiments/H_expressive_intelligence")


def source_hashes() -> dict[str, str]:
    out = {}
    for d in SOURCE_DIRS:
        for f in sorted((F.ROOT / d).rglob("*.py")):
            out[f.relative_to(F.ROOT).as_posix()] = sha256(f)
    return out


def nerva_state() -> dict[str, Any]:
    """NERVA commit and cleanliness (no paths recorded); empty if NERVA_ROOT is unset."""
    root = os.environ.get("NERVA_ROOT")
    if not root:
        return {}

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(Path(root)), *args], capture_output=True, text=True).stdout.strip()

    return {"nerva_commit": git("rev-parse", "HEAD"), "nerva_dirty": bool(git("status", "--porcelain"))}


def confirmatory_lock(addendum: str | Path) -> dict[str, Any]:
    """Refuses unless the addendum is committed and no tracked file is modified; adds H source and NERVA state."""
    rec = F.confirmatory_lock(addendum)
    rec["sources"] = source_hashes()
    rec.update(nerva_state())
    if rec.get("nerva_dirty"):
        raise RuntimeError("confirmatory run refused: the NERVA checkout has uncommitted changes")
    return rec
