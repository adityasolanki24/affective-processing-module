"""L protocol §E: fair comparison of expression controllers. The script has two modes: tune on partition T, then
analyse on X.

    python e_controllers.py tune --runs <LE_T_*.jsonl> --out configs/LE_tuned.json
    python e_controllers.py analyse --runs <LE_X_*.jsonl> --tuned configs/LE_tuned.json --out results/E_controllers.json

**Task cost** per encounter is LB4's: approach time + 4 s × (windows − 1) + 20 s × strong perturbations
+ 12.6 s × (1 − measured success). Each lifetime contributes its mean. Comparisons are paired over
(history, index) with bootstrap 95% CIs.

**Tuning (equal budgets).**
- E1 and E2 each evaluate 12 configurations per task on partition T (risky_B and safe).
- Per task, the configuration with the lowest mean cost is chosen. Ties go to the earlier, simpler grid entry.
- E3 and E4 have no free parameters, because their task values are frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
K = REPO / "experiments" / "K_recovery"
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import k3b_analyze as K3B  # noqa: E402
import k_analyze as KA  # noqa: E402

ORDER = ("E0", "E1", "E2", "E3", "E4")  # simplest first
NAMES = {"L0_VIS": "E0", "DT_IMUVIS": "E3", "DT_IMUVIS_VOI": "E4"}
EQUIV_S = 1.0


def load(paths: list[Path]) -> dict:
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())["tasks"]
    K3B.MEASURED = True
    d: dict = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            lt = json.loads(line)
            spec = tasks[lt["task"]]
            rows = lt["rows"]
            d[lt["task"]][lt["system"]][(lt["history"], lt["index"])] = {
                "cost": float(np.mean([K3B.enc_cost(x, spec) for x in rows])),
                "contacts": float(np.mean([x["contacts"] for x in rows])),
                "success": float(np.mean([K3B.success(x, spec) for x in rows])),
                "falls": int(sum(x["fell"] for x in rows))}
    return d


def tune(a: argparse.Namespace) -> None:
    from apm_transfer.controllers import E1_GRID, E2_GRID
    d = load(a.runs)
    out: dict = {"rule": "per task, lowest mean T cost; ties to the earlier grid entry", "tuned": {}, "table": {}}
    for task, sy in sorted(d.items()):
        out["tuned"][task] = {}
        for ctl, grid in (("E1", E1_GRID), ("E2", E2_GRID)):
            names = [f"{ctl}:" + "/".join(f"{k}={v:g}" for k, v in g.items()) for g in grid]
            costs = [float(np.mean([v["cost"] for v in sy[n].values()])) if n in sy else np.inf for n in names]
            best = int(np.argmin(costs))
            out["tuned"][task][ctl] = names[best]
            out["table"].setdefault(task, {})[ctl] = dict(zip(names, [round(c, 3) for c in costs], strict=True))
            print(f"{task:10} {ctl}: {names[best]} cost {costs[best]:.2f} (range {min(costs):.2f}-{max(costs):.2f})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


def analyse(a: argparse.Namespace) -> None:
    d = load(a.runs)
    tuned = json.loads(a.tuned.read_text())["tuned"]
    res: dict = {"tasks": {}}
    better_tasks = 0
    for task, sy in sorted(d.items()):
        ids = {**{k: v for k, v in NAMES.items()}, tuned[task]["E1"]: "E1", tuned[task]["E2"]: "E2"}
        by = {ids[s]: lts for s, lts in sy.items() if s in ids}
        o: dict = {"systems": {e: {m: float(np.mean([v[m] for v in by[e].values()])) for m in
                                   ("cost", "contacts", "success")} | {"falls": int(sum(v["falls"] for v in
                                                                                        by[e].values()))}
                               for e in ORDER if e in by}, "configs": {"E1": tuned[task]["E1"],
                                                                        "E2": tuned[task]["E2"]}}

        def diff(x: str, y: str, m: str, by: dict = by) -> dict:
            keys = sorted(set(by[x]) & set(by[y]))
            return KA.boot([by[x][k][m] - by[y][k][m] for k in keys])
        o["pairs"] = {f"{x}-{y}": {m: diff(x, y, m) for m in ("cost", "contacts")}
                      for x in ("E3", "E4") for y in ("E0", "E1", "E2") if x in by and y in by}
        o["pairs"]["E4-E3"] = {m: diff("E4", "E3", m) for m in ("cost", "contacts")} if "E4" in by else {}
        beats = {}
        for x in ("E3", "E4"):
            beats[x] = all(o["pairs"][f"{x}-{y}"]["cost"]["ci95"][1] < 0
                           and o["pairs"][f"{x}-{y}"]["contacts"]["ci95"][1] <= 0.05 for y in ("E0", "E1", "E2"))
        o["beats_calibrated_simple"] = beats
        better_tasks += int(beats["E3"] or beats["E4"])
        # recommended: the simplest controller equivalent to (within EQUIV_S of) the lowest-cost one, not worse
        best = min(o["systems"], key=lambda e: o["systems"][e]["cost"])
        rec = best
        for e in ORDER:
            if e == best:
                break
            dd = diff(e, best, "cost")
            if dd["ci95"][1] is not None and dd["ci95"][1] <= EQUIV_S:
                rec = e
                break
        o["lowest_cost"], o["recommended"] = best, rec
        res["tasks"][task] = o
        print(f"{task:10} " + " ".join(f"{e} {o['systems'][e]['cost']:.2f}/{o['systems'][e]['contacts']:.2f}"
                                       for e in o["systems"]) + f" | beats {beats} | rec {rec}")
    n = len(res["tasks"])
    res["answer_better_than_calibrated_simple"] = ("yes" if better_tasks >= 3 else "partly" if better_tasks >= 1
                                                   else "no") if n == 4 else f"{better_tasks} of {n}"
    res["tasks_better"] = better_tasks
    res["E5_learned_justified"] = better_tasks == 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(res["answer_better_than_calibrated_simple"], "| E5 justified:", res["E5_learned_justified"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tune", "analyse"))
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--tuned", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tune(a) if a.mode == "tune" else analyse(a)


if __name__ == "__main__":
    main()
