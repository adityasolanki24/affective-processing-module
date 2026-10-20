"""Programme I runner (all experiments). Writes one JSON line per lifetime and a run_meta.json.

    python experiments/I_autonomous_revision/scripts/i_run.py --experiment I1 --partition P --n 6 \
        --systems experiments/I_autonomous_revision/configs/I1_pilot.json --scenarios S1,S2,S3

Partitions X and S require `--addendum` (committed) and a clean tree (`apm_revise.provenance.confirmatory_lock`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i_common import REPO, init, run_job  # noqa: E402

from apm_revise.provenance import confirmatory_lock, environment, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--systems", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated subset of system names")
    ap.add_argument("--scenarios", default="S1,S2,S3,S4,S5,S6,S7,S8,S9")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "I")
    ap.add_argument("--tag", default="")
    ap.add_argument("--addendum", default="")
    a = ap.parse_args()
    if a.partition in ("X", "S"):
        if not a.addendum:
            raise SystemExit("confirmatory partitions need --addendum")
        provenance = confirmatory_lock(REPO / "experiments" / "I_autonomous_revision" / "prereg" / a.addendum)
    else:
        provenance = environment()
    systems = json.loads(a.systems.read_text())["systems"]
    if a.only:
        systems = {k: v for k, v in systems.items() if k in a.only.split(",")}
    jobs = [{"spec": {**spec, "label": name}, "scenario": sc, "experiment": a.experiment, "partition": a.partition,
             "index": a.first + i}
            for name, spec in systems.items() for sc in a.scenarios.split(",") for i in range(a.n)]
    out_dir = a.out / a.experiment / (a.partition + (f"_{a.tag}" if a.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    n_done = 0
    with Pool(a.workers, initializer=init) as pool, open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
            n_done += 1
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "systems": systems, "n": a.n, "first": a.first,
                                           "scenarios": a.scenarios.split(","), "jobs": n_done,
                                           "wall_s": round(time.time() - start, 1)})
    print("done", n_done, "lifetimes in", round(time.time() - start, 1), "s ->", out_dir)


if __name__ == "__main__":
    main()
