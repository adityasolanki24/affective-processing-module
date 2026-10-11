"""E7 analysis: per fault, (1) exceptions that reached the harness and its GO fallbacks, (4) collision rate of B1
and B2 relative to B0 on the same faulted lifetimes (paired cluster bootstrap; criterion: CI upper ≤ +5%).

    python scripts/e7_analyze.py --partition P
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
from apm_research.provenance import write_json  # noqa: E402
from apm_research.stats import paired_lifetimes  # noqa: E402

MARGIN = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/E7"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    data: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(args.runs / args.partition / "lifetimes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            data[r["fault"]][r["system"]][r["index"]] = r
    res: dict = {"partition": args.partition, "faults": {}}
    for fault, systems in data.items():
        row: dict = {}
        for s, lives in systems.items():
            tot = {k: sum(r["totals"][k] for r in lives.values()) for k in ("collisions", "minutes")}
            row[s] = {"lifetimes": len(lives), "exceptions": int(sum(r["exceptions"] for r in lives.values())),
                      "fallbacks": int(sum(r["fallbacks"] for r in lives.values())),
                      "collisions_per_min": tot["collisions"] / max(1e-9, tot["minutes"])}
        for s in ("B1", "B2"):
            if s in systems and "B0" in systems:
                common = sorted(set(systems[s]) & set(systems["B0"]))
                a = {k: np.array([systems[s][i]["totals"][k] for i in common], float)
                     for k in systems[s][common[0]]["totals"]}
                b = {k: np.array([systems["B0"][i]["totals"][k] for i in common], float)
                     for k in systems["B0"][common[0]]["totals"]}
                c = paired_lifetimes(a, b)
                row[f"{s}_vs_B0"] = c
                row[f"criterion4_{s}"] = bool(c["collision_rate_relative"]["ci95"][1] <= MARGIN)
        res["faults"][fault] = row
    res["criterion4_all"] = bool(all(v for f in res["faults"].values() for k, v in f.items()
                                     if k.startswith("criterion4_")))
    out = args.out or HERE / "results" / "E7" / f"e7_{args.partition}.json"
    print(write_json(out, res))
    print(json.dumps({f: {k: (v["collision_rate_relative"]["estimate"], v["collision_rate_relative"]["ci95"])
                          for k, v in row.items() if k.endswith("_vs_B0")} | {"exceptions": {s: row[s]["exceptions"]
                          for s in ("B0", "B1", "B2") if s in row}} for f, row in res["faults"].items()}, indent=1))


if __name__ == "__main__":
    main()
