"""G7 lifelong analysis (prereg/G7_ADDENDUM.md): after each phase's update, every learning system vs the frozen L0 on
the current-phase and first-phase frozen suites (paired lifetimes): mean J difference, collision rate, completion.

Success (L2, the gated system): in at least one phase a *promoted* update improves the current-phase suite's mean J by
≥ 5% of L0's with bootstrap CI lower > 0, with collision r CI upper ≤ +5%, and the first-phase suite's mean J is not
worse by more than 5% (CI lower > −5%). Rejected updates never count as learning.

    python scripts/g7_analyze.py --partition X
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from apm_select.provenance import write_json  # noqa: E402
from apm_select.stats import compare_systems, strip  # noqa: E402


def j_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000) -> dict:
    rng = np.random.default_rng(0)
    d = a - b
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    base = float(b.mean())
    rel = draws / abs(base)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "rel": float(d.mean() / abs(base)), "rel_ci95": [float(np.percentile(rel, 2.5)),
                                                             float(np.percentile(rel, 97.5))], "L0_mean": base}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="X")
    ap.add_argument("--runs", type=Path, default=Path("runs/G7"))
    args = ap.parse_args()
    meta = json.loads((args.runs / f"run_meta_{args.partition}.json").read_text())
    rows: dict = defaultdict(lambda: defaultdict(dict))
    for line in (args.runs / f"evals_{args.partition}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[(r["after_phase"], r["suite"])][r["system"]][r["index"]] = r["totals"]
    res: dict = {"log": meta["log"], "comparisons": {}}
    for (phase, suite), by in rows.items():
        idx = sorted(by["L0"])
        arr = {s: {k: np.array([by[s][i][k] for i in idx]) for k in by[s][idx[0]]} for s in by}
        for s in by:
            if s == "L0":
                continue
            c = compare_systems(arr[s], arr["L0"])
            res["comparisons"][f"{phase}|{suite}|{s}"] = {"J": j_diff(arr[s]["J"], arr["L0"]["J"]),
                                                         "collision_rel": c["collision_rate"]["rel"],
                                                         "completion_abs": c["completion"]["abs"]}
    promoted = {(e["phase"], e["system"]) for e in meta["log"] if e.get("event") == "promoted"}
    success = []
    for phase in ("PH2", "PH3"):
        if (phase, "L2") not in promoted:
            continue
        cur = res["comparisons"].get(f"{phase}|{phase}|L2")
        old = res["comparisons"].get(f"{phase}|PH1|L2")
        if cur and old and cur["J"]["rel"] >= 0.05 and cur["J"]["ci95"][0] > 0 \
                and cur["collision_rel"]["ci95"][1] <= 0.05 and old["J"]["rel_ci95"][0] > -0.05:
            success.append(phase)
    res["promotions"] = sorted(f"{p}:{s}" for p, s in promoted)
    res["L2_success_phases"] = success
    res["lifelong_success"] = bool(success)
    print(write_json(HERE / "results" / "G7" / f"g7_{args.partition}.json", strip(res)))
    print(json.dumps({"promotions": res["promotions"], "success": success,
                      "J_rel": {k: (round(v["J"]["rel"], 3), [round(x, 3) for x in v["J"]["rel_ci95"]],
                                    round(v["collision_rel"]["estimate"], 3)) for k, v in res["comparisons"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
