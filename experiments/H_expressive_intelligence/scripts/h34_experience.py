"""H3/H4: generate physically measured experience streams (robot lifetimes of encounters).

For each partition, history and lifetime:
1. draw the world schedule (identity, zone, jolted) from `apm_express.develop`;
2. measure every encounter's outcome in NERVA's simulation: the robot standing, pushed at m* if jolted, 4 s
   observed.

All encounters are independent simulations, so they run in parallel.

    NERVA_ROOT=... PYTHONPATH="src;<NERVA>" <nerva-venv>/python scripts/h34_experience.py --partition P \
        --lifetimes 2 --out results/H34/P
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h0_interface import _init  # noqa: E402
from h1_repertoire import S  # noqa: E402

from apm_express.develop import HISTORIES, N_ENCOUNTERS, kind_of, measure_encounter, schedule  # noqa: E402


def lifetime_plan(partition: str, history: str, index: int) -> list[dict]:
    ss = S.world_seed("H4", partition, HISTORIES.index(history) * 100_000 + index)
    rng = np.random.default_rng(ss)
    plan = []
    for k, (ident, zone, jolted) in enumerate(schedule(history, rng)):
        plan.append({"k": k, "identity": ident, "kind": kind_of(ident), "zone": zone, "jolted": jolted,
                     "direction": float(rng.uniform(-math.pi, math.pi)),
                     "sim_seed": S.sim_seed("H4", partition, (HISTORIES.index(history) * 100_000 + index) * 100 + k)})
    return plan


def measure(job: dict) -> dict:
    tilt, speed = measure_encounter(job["sim_seed"], job["jolted"], job["direction"])
    return {**job, "peak_tilt": tilt, "peak_speed": speed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["P", "T", "V", "X", "S"])
    ap.add_argument("--lifetimes", type=int, required=True)
    ap.add_argument("--histories", default=",".join(HISTORIES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--index-offset", type=int, default=0, help="fresh lifetime indices (H4b: 1000)")
    ap.add_argument("--addendum", default="H34_ADDENDUM.md")
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock, environment, nerva_state
    lock = (confirmatory_lock(HERE / "prereg" / a.addendum) if a.partition in ("X", "S")
            else {**environment(), **nerva_state(), "partition": a.partition})
    jobs, keys = [], []
    for h in a.histories.split(","):
        for i in range(a.index_offset, a.index_offset + a.lifetimes):
            for e in lifetime_plan(a.partition, h, i):
                jobs.append(e)
                keys.append((h, i))
    start = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        rows = pool.map(measure, jobs, chunksize=4)
    lifetimes: dict = {}
    for (h, i), r in zip(keys, rows, strict=True):
        lifetimes.setdefault(f"{h}/{i}", {"history": h, "index": i, "events": []})["events"].append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "experience.json"
    path.write_text(json.dumps({"partition": a.partition, "n_encounters": N_ENCOUNTERS, "lock": lock,
                                "wall_s": round(time.time() - start, 1), "lifetimes": list(lifetimes.values())}),
                    encoding="utf-8")
    print(json.dumps({"lifetimes": len(lifetimes), "encounters": len(rows), "wall_s": round(time.time() - start, 1),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
