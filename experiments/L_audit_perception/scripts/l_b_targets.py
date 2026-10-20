"""L protocol §B.2: recovery targets calibrated on partition V (safe history only), frozen before X analysis.

    python l_b_targets.py --runs LB1=<V.jsonl> LB2=<V.jsonl> LB3=<V.jsonl> --out configs/LB_recovery_targets.json

Target per body and system: the mean B1 stopping distance at encounters >= 10 of the safe history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LATE = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="EXP=path")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out: dict = {"rule": "mean B1 d_stop, encounters >= 10, safe history, partition V", "targets": {}, "sources": {}}
    for item in a.runs:
        exp, path = item.split("=", 1)
        p = Path(path)
        d: dict = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            assert r["partition"] == "V" and r["history"] == "safe", (r["partition"], r["history"])
            d.setdefault(r["system"], []).extend(x["d_stop"] for x in r["rows"] if x["entity"] == "B1"
                                                 and x["k"] >= LATE)
        out["targets"][exp] = {s: round(float(np.mean(v)), 4) for s, v in sorted(d.items())}
        out["sources"][exp] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                               "n_values": {s: len(v) for s, v in d.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["targets"], indent=1))


if __name__ == "__main__":
    main()
