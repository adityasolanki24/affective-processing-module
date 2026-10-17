"""Hash every programme-J raw run file (`runs/J/`, git-ignored) into `manifests/raw_data.json`.

    python j_manifest.py

Per file: path relative to the repo, git-normalised SHA-256 (CRLF → LF), size and line count. A result can be checked
against the raw data it came from without committing the data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments" / "J_validation_transfer" / "manifests" / "raw_data.json"


def main() -> None:
    rows = []
    for f in sorted((REPO / "runs" / "J").rglob("*")):
        if f.is_file() and f.suffix in (".jsonl", ".json"):
            b = f.read_bytes().replace(b"\r\n", b"\n")
            rows.append({"path": f.relative_to(REPO).as_posix(), "sha256": hashlib.sha256(b).hexdigest(),
                         "bytes": len(b), "lines": b.count(b"\n")})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"files": rows}, indent=1) + "\n", encoding="utf-8")
    print(len(rows), "files ->", OUT)


if __name__ == "__main__":
    main()
