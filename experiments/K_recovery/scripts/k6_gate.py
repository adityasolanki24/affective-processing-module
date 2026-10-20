"""K6 gate G0 (`prereg/K6_HUMANOID.md`): validate the humanoid walking controller before any APM data.

    python k6_gate.py --n 50 --out results/K6_gate.json [--render <dir>]

Encounters with random entity type (calm/pushy), stop distance (grid proximity), tempo and 1–3 windows, on seeds
K5/P (the humanoid's own pilot partition is unused by other robots: experiment K5, partition P, indices 1000+).

**Pass** if all of the following hold:
- 0 falls;
- arrival in ≥ 98% of encounters;
- |stopping distance − requested| ≤ 0.1 m in ≥ 90% of encounters.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / "src"))


def _seeds():
    spec = importlib.util.spec_from_file_location("k_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def one(i: int) -> dict:
    from apm_transfer.expression_map import GRID, stop_distance
    from apm_transfer.humanoid_world import closed_loop_encounter
    from apm_transfer.nerva_world import Camera
    seed = int(_seeds().world_seed("K5", "P", 1000 + i).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    gi = int(rng.integers(len(GRID)))
    etype = "pushy" if rng.random() < 0.5 else "calm"
    w = int(rng.integers(1, 4))
    ev, out, truth = closed_loop_encounter(seed, etype, GRID[gi], windows=w, camera=Camera())
    req = stop_distance(GRID[gi].proximity)
    return {"i": i, "type": etype, "grid_i": gi, "requested": round(req, 3), "d_stop": out["final_distance"],
            "arrived": out["arrived"], "t_arrive": out["t_arrive"], "fell": out["fell"], "trips": out["trips"],
            "contacts": truth["contacts"], "max_push": truth["max_push"], "windows": len(out["windows"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with Pool(min(a.workers, a.n)) as pool:
        rows = pool.map(one, range(a.n))
    arrived = [r for r in rows if r["arrived"]]
    acc = np.mean([abs(r["d_stop"] - r["requested"]) <= 0.1 for r in arrived]) if arrived else 0.0
    res = {"n": a.n, "falls": int(sum(r["fell"] for r in rows)), "arrival_rate": len(arrived) / a.n,
           "stop_within_0.1m": float(acc), "trips": int(sum(r["trips"] > 0 for r in rows)),
           "contacts_pushy": float(np.mean([r["contacts"] for r in rows if r["type"] == "pushy"] or [0])),
           "rows": rows}
    res["G0_pass"] = res["falls"] == 0 and res["arrival_rate"] >= 0.98 and res["stop_within_0.1m"] >= 0.90
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
