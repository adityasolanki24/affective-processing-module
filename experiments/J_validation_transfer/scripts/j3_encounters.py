"""J3: generate standing NERVA encounters (robot-observable evidence plus evaluation-only truth). Needs NERVA's venv.

    python j3_encounters.py --partition P --n 200 --distances 0.2,0.4,0.6 --types calm,gentle,forceful --out <jsonl>

One row per encounter:
- partition, index, type, distance;
- evidence: peak gyro, peak horizontal acceleration deviation, stabilisation time, contact changes, effort ratio,
  tilt peak, fell;
- truth: lunges, contacts, max push, summed push.

Seeds come from J3/<partition>/<index>.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _seeds():
    spec = importlib.util.spec_from_file_location("j_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def job(args: tuple) -> dict:
    partition, index, etype, distance = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from apm_transfer.nerva_world import standing_encounter
    S = _seeds()
    seed = int(S.world_seed("J3", partition, index).generate_state(1)[0])
    ev, truth = standing_encounter(seed, etype, distance)
    return {"partition": partition, "index": index, "type": etype, "distance": distance, "evidence": ev.as_list(),
            "truth": truth}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--distances", default="0.2,0.4,0.6")
    ap.add_argument("--types", default="calm,gentle,forceful")
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs, k = [], a.first
    for etype in a.types.split(","):
        for d in (float(x) for x in a.distances.split(",")):
            for _ in range(a.n):
                jobs.append((a.partition, k, etype, d))
                k += 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with Pool(a.workers) as pool, open(a.out, "w", encoding="utf-8") as fh:
        for row in pool.imap_unordered(job, jobs, chunksize=4):
            fh.write(json.dumps(row) + "\n")
    print("done", len(jobs), "encounters in", round(time.time() - t, 1), "s ->", a.out)


if __name__ == "__main__":
    main()
