"""G1 evaluation: every selected system on the same lifetimes (common random numbers), plus negative controls.

Systems come from a JSON file of specs (`--systems`): for partition P a pilot file (programme F's tuned values),
for X/S `configs/G1_selected.json` (written from tuning/ES results and committed with prereg/G1_ADDENDUM.md).
Negative controls (evaluation only, no tuning) for R6 and for the best simple family named in the file:
  <name>_shuffled  the modulation trace (m, tendency bonuses) recorded on lifetime (i + 1) mod N, replayed
  <name>_shifted   the system's own trace on the same lifetime, rotated by one episode (120 decisions)

    python scripts/g1_run.py --partition P --n 40 --systems configs/G1_pilot_specs.json
    python scripts/g1_run.py --partition X --n <addendum> --systems configs/G1_selected.json
    python scripts/g1_run.py --partition S --n <addendum> --systems configs/G1_selected.json \
        --conditions miscal_x3,dense,sparse --no-controls
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from g_common import PREDICTOR_SHA256, run_jobs  # noqa: E402

from apm_select.provenance import confirmatory_lock, environment, sha256, write_json  # noqa: E402

EPISODES, DECISIONS_PER_EPISODE = 10, 120


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--systems", type=Path, required=True)
    ap.add_argument("--conditions", default="nominal")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/G1"))
    args = ap.parse_args()
    provenance = (confirmatory_lock(HERE / "prereg" / "G1_ADDENDUM.md") if args.partition in ("X", "S")
                  else environment())
    assert sha256(Path("runs/E2/predictor.json")) == PREDICTOR_SHA256, "frozen predictor changed"
    start = time.time()
    cfg = json.loads(args.systems.read_text())
    systems: dict = cfg["systems"]
    controls = [] if args.no_controls else [n for n in ("R6", cfg.get("best_simple")) if n and n in systems]
    out_dir = args.out / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for cond in args.conditions.split(","):
        jobs = [{"spec": {**spec, "label": name}, "experiment": "G1", "partition": args.partition, "index": i,
                 "n_episodes": EPISODES, "condition": cond, "keep_readouts": name in controls}
                for name, spec in systems.items() for i in range(args.n)]
        jobs.sort(key=lambda j: j["spec"]["kind"] != "b2v")
        res = run_jobs(jobs, args.workers, out_dir / f"lifetimes_{cond}_base.jsonl")
        traces = {(r["label"], r["index"]): r["readouts"] for r in res if "readouts" in r}
        cjobs = []
        for name in controls:
            shift = DECISIONS_PER_EPISODE
            for i in range(args.n):
                own = traces[(name, i)]
                for suffix, trace in (("shuffled", traces[(name, (i + 1) % args.n)]),
                                      ("shifted", own[shift:] + own[:shift])):
                    cjobs.append({"spec": {**systems[name], "label": f"{name}_{suffix}", "trace": trace},
                                  "experiment": "G1", "partition": args.partition, "index": i,
                                  "n_episodes": EPISODES, "condition": cond})
        cres = run_jobs(cjobs, args.workers, out_dir / f"lifetimes_{cond}_controls.jsonl") if cjobs else []
        all_results += [{k: v for k, v in r.items() if k != "readouts"} for r in res + cres]
    with open(out_dir / "lifetimes.jsonl", "w", encoding="utf-8") as fh:
        for r in all_results:
            fh.write(json.dumps(r) + "\n")
    write_json(out_dir / "run_meta.json", {"provenance": provenance, "systems_file": str(args.systems),
                                           "systems_sha256": sha256(args.systems), "systems": systems,
                                           "controls": controls, "n": args.n, "episodes": EPISODES,
                                           "conditions": args.conditions.split(","),
                                           "wall_s": round(time.time() - start, 1)})
    print("done", len(all_results), round(time.time() - start, 1))


if __name__ == "__main__":
    main()
