"""Programme G provenance: F's confirmatory lock, with G's source directories hashed as well."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from apm_research import provenance as F
from apm_research.provenance import environment, sha256, write_json  # noqa: F401  (re-exported)

SOURCE_DIRS = ("src/apm", "src/apm_research", "src/apm_testbeds", "src/apm_select",
               "experiments/G_architecture_selection")


def source_hashes() -> dict[str, str]:
    out = {}
    for d in SOURCE_DIRS:
        for f in sorted((F.ROOT / d).rglob("*.py")):
            out[f.relative_to(F.ROOT).as_posix()] = sha256(f)
    return out


def confirmatory_lock(addendum: str | Path) -> dict[str, Any]:
    """Refuses unless the addendum is committed and no tracked file is modified; adds G source hashes."""
    rec = F.confirmatory_lock(addendum)
    rec["sources"] = source_hashes()
    return rec
