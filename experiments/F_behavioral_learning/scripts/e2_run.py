"""E2 runs: every system on the same lifetimes (common random numbers). Partition X needs the committed addendum.

Systems: B0, B1, B2 (tuned) and B2 ablations (no PAD, no tonic appraisal, no entity-memory predictions, no persistent
affect, shuffled affect history = B2 readouts replayed from lifetime (i + 1) mod N), plus the evaluation-only Oracle.

    python scripts/e2_run.py --partition P --n 40
    python scripts/e2_run.py --partition X --n <from addendum>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from e2_common import run_jobs  # noqa: E402

from apm_research.provenance import confirmatory_lock, environment, write_json  # noqa: E402

EPISODES = 10


def system_specs(tuned: dict, oracle_rates: dict | None) -> dict[str, dict]:
    b1, b2 = tuned["best"]["B1"], tuned["best"]["B2"]
    lam1 = {"lam_c": b1["lam_c"], "lam_s": b1["lam_s"]}
    lam2 = {"lam_c": b2["lam_c"], "lam_s": b2["lam_s"], "k_pad": b2["k_pad"], "k_t": b2["k_t"]}
    specs = {"B0": {"system": "B0"}, "B1": {"system": "B1", **lam1}, "B2": {"system": "B2", **lam2},
             "B2_noPAD": {"system": "B2", **lam2, "k_pad": 0.0},
             "B2_noTonic": {"system": "B2", **lam2, "tonic": False},
             "B2_noMemoryFrames": {"system": "B2", **lam2, "memory_frames": False},
             "B2_noPersistence": {"system": "B2", **lam2, "persistent": False}}
    if oracle_rates is not None:
        specs["Oracle"] = {"system": "B1", **lam1, "oracle_by_disposition": oracle_rates}
    for k, v in specs.items():
        v["label"] = k
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="E2")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--predictor", default="runs/E2/predictor.json")
    ap.add_argument("--tuned", type=Path, default=HERE / "configs" / "E2_tuned.json")
    ap.add_argument("--systems", default="B0,B1,B2,B2_noPAD,B2_noTonic,B2_noMemoryFrames,B2_noPersistence,"
                                        "B2_shuffled,Oracle")
    ap.add_argument("--out", type=Path, default=Path("runs/E2"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    provenance = (confirmatory_lock(HERE / "prereg" / f"{args.experiment}_ADDENDUM.md")
                  if args.partition in ("X", "S") else environment())
    start = time.time()
    tuned = json.loads(args.tuned.read_text())
    oracle = json.loads((HERE / "configs" / "oracle_rates.json").read_text())["rates_by_disposition"] \
        if (HERE / "configs" / "oracle_rates.json").exists() else None
    specs = system_specs(tuned, oracle)
    wanted = args.systems.split(",")
    plain = [s for s in wanted if s in specs]
    jobs = [{"spec": specs[s], "experiment": args.experiment, "partition": args.partition, "index": i,
             "n_episodes": EPISODES} for s in plain for i in range(args.n)]
    jobs.sort(key=lambda j: j["spec"]["system"] != "B2")
    results = run_jobs(jobs, args.predictor, args.workers)
    if "B2_shuffled" in wanted:
        traces = {r["index"]: r["readouts"] for r in results if r["label"] == "B2"}
        shuffled = [{"spec": {**specs["B2"], "label": "B2_shuffled", "trace": traces[(i + 1) % args.n]},
                     "experiment": args.experiment, "partition": args.partition, "index": i, "n_episodes": EPISODES}
                    for i in range(args.n)]
        results += run_jobs(shuffled, args.predictor, args.workers)
    out_dir = args.out / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            r.pop("readouts", None)
            fh.write(json.dumps(r, default=float) + "\n")
    meta = {"provenance": provenance, "systems": {k: v for k, v in specs.items() if k in wanted},
            "n": args.n, "episodes": EPISODES, "wall_s": round(time.time() - start, 1)}
    meta["systems"].get("B2_shuffled", {}).pop("trace", None)
    write_json(out_dir / "run_meta.json", meta)
    print("done", len(results), meta["wall_s"])


if __name__ == "__main__":
    main()
