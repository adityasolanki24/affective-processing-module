"""Writes configs/G6_systems.json by the rule in prereg/G6_ADDENDUM.md (validation records only).

    python scripts/g6_systems.py --models runs/G45/models
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("runs/G45/models"))
    args = ap.parse_args()
    g1 = json.loads((HERE / "configs" / "G1_selected.json").read_text())
    g2b = json.loads((HERE / "configs" / "G2b_selected.json").read_text())
    a0 = {k: v for k, v in g1["systems"][g1["best_simple"]].items() if k != "config_id"}
    rules = {"A0": a0, "R1": {k: v for k, v in g1["systems"]["R1"].items() if k != "config_id"},
             "A0_mem": {**a0, "memory": g2b["systems"]["M1t_probeU"]["memory"]},
             "A1": {k: v for k, v in g1["systems"]["R6"].items() if k != "config_id"}, "B0": {"kind": "b0"}}
    val = defaultdict(list)
    for f in args.models.glob("*_s?.json"):
        r = json.loads(f.read_text())
        val[f"{r['arch']}_{r['method']}"].append(r["best"]["J"])
    best = {a: max((c for c in val if c.startswith(a + "_")), key=lambda c: np.mean(val[c]))
            for a in ("A2", "A3", "A4") if any(c.startswith(a + "_") for c in val)}
    out = {"rule": "best mean best-validation J cell per architecture", "validation": {c: float(np.mean(v))
                                                                                     for c, v in val.items()},
           "rules": rules, "models": sorted(best.values())}
    (HERE / "configs" / "G6_systems.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"models": out["models"], "validation": out["validation"]}, indent=1))


if __name__ == "__main__":
    main()
