"""M6 analysis (PROTOCOL.md §6-§7): per-robot hypotheses A1-D3, cross-robot E, safety.

    python experiments/M_release/scripts/m6_analyze.py --runs <dir or jsonl ...> --tasks <robot>=<task json> ...
        --out results/m6/analysis.json

Input: m5_lifetime.py JSONL records (one lifetime per line). The unit is the lifetime; systems are paired by
(robot, scene, history, index).

Measures (§6), per encounter, with d the abstract stopping distance from the robot's own odometry:
- `contacts`: the person's lunges that reached the robot (PhysX truth);
- `success`: d <= d_task + 0.05;
- realised cost = arrival time + price_s x contacts + retry_s x (1 - success). An encounter without arrival
  counts the approach budget (45 s) as its time.

Tests (§7): one-sided paired sign-flip permutation test on per-lifetime differences (10 000 permutations, seed
20261019), Holm across A1-D3 within each robot (alpha 0.05), and a 95% bootstrap CI of the mean difference
(10 000 resamples).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SWITCH = 10
APPROACH_BUDGET_S = 45.0
ALPHA = 0.05
SEED = 20261019


def load(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            if f.parent.name == "logs":
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if "rows" in rec and "system" in rec:
                    out.append(rec)
    return out


def enc_cost(r: dict, task: dict) -> float:
    t = r["t_arrive"] if r.get("t_arrive") is not None else APPROACH_BUDGET_S
    success = r["d_stop"] <= task["d_task"] + 0.05
    return float(t + task["price_s"] * r["contacts"] + task["retry_s"] * (1.0 - success))


def b1_after(rows: list[dict], phase: str, skip: int = 2) -> list[dict]:
    """B1 encounters before/after the switch, from the (skip+1)-th B1 meeting of that phase on."""
    sel = [r for r in rows if r["entity"] == "B1" and ((r["k"] < SWITCH) if phase == "pre" else (r["k"] >= SWITCH))]
    return sel[skip:]


def metric(rec: dict, name: str, task: dict) -> float | None:
    rows = rec["rows"]
    if name in ("A", "A_contacts"):  # change_B, pre-switch B1 contacts per encounter
        sel = b1_after(rows, "pre")
        return float(np.mean([r["contacts"] for r in sel])) if sel else None
    if name == "B_late":  # late_B, post-switch B1 contacts per encounter
        sel = b1_after(rows, "post")
        return float(np.mean([r["contacts"] for r in sel])) if sel else None
    if name == "B_change":  # change_B, post-switch B1 success rate
        sel = b1_after(rows, "post")
        return float(np.mean([r["d_stop"] <= task["d_task"] + 0.05 for r in sel])) if sel else None
    if name == "cost":
        return float(np.mean([enc_cost(r, task) for r in rows])) if rows else None
    raise ValueError(name)


# (id, metric, history filter, scene filter, DT vs, direction: "less" = DT smaller is the prediction)
HYPOTHESES = [
    ("A1", "A", "change_B", "warehouse", "NOMEM", "less"),
    ("A2", "A", "change_B", "warehouse", "SHUFFLED", "less"),
    ("B1", "B_late", "late_B", "warehouse", "FROZEN", "less"),
    ("B2", "B_change", "change_B", "warehouse", "FROZEN", "greater"),
    ("C1", "cost", None, "warehouse", "FIXED", "less"),
    ("C2", "cost", None, "warehouse", "NEUTRAL", "less"),
    ("C3", "cost", None, "warehouse", "LOOKUP", "less"),
    ("C4", "cost", None, "warehouse", "MONOTONE", "less"),
    ("D1", "cost", None, "hospital", "REACTIVE", "less"),
    ("D2", "cost", None, "hospital", "MONOTONE", "less"),
    ("D3", "cost", None, "home", "REACTIVE", "less"),
]


def paired(recs: list[dict], robot: str, scene: str, history: str | None, other: str, name: str,
           task: dict) -> np.ndarray:
    """Per-pair differences DT - other (a lifetime index and history define a pair)."""
    idx: dict[tuple, dict] = {}
    for r in recs:
        if r["robot"] != robot or r["scene"] != scene or (history and r["history"] != history):
            continue
        sysname = r["system"].split(":")[0]
        if sysname not in ("DT", other):
            continue
        idx.setdefault((r["history"], r["index"]), {})[sysname] = r
    diffs = []
    for pair in idx.values():
        if "DT" in pair and other in pair:
            a, b = metric(pair["DT"], name, task), metric(pair[other], name, task)
            if a is not None and b is not None:
                diffs.append(a - b)
    return np.asarray(diffs, float)


def perm_p(d: np.ndarray, direction: str, n: int = 10_000) -> float:
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(SEED)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * np.abs(d)).mean(1)
    return float(((null <= obs) if direction == "less" else (null >= obs)).mean())


def boot_ci(d: np.ndarray, n: int = 10_000) -> list[float]:
    if len(d) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED + 1)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps: dict[str, float]) -> dict[str, bool]:
    order = sorted((p, k) for k, p in ps.items() if np.isfinite(p))
    out = {k: False for k in ps}
    m = len(order)
    for i, (p, k) in enumerate(order):
        if p <= ALPHA / (m - i):
            out[k] = True
        else:
            break
    return out


def safety(recs: list[dict]) -> dict:
    s: dict = {}
    for r in recs:
        key = f'{r["robot"]}/{r["scene"]}/{r["system"]}'
        e = s.setdefault(key, {"lifetimes": 0, "encounters": 0, "falls": 0, "trips": 0, "contacts": 0,
                               "vetoes": 0, "fall_list": []})
        e["lifetimes"] += 1
        e["encounters"] += len(r["rows"])
        e["trips"] += sum(x["trips"] for x in r["rows"])
        e["contacts"] += sum(x["contacts"] for x in r["rows"])
        e["vetoes"] += r.get("vetoes", 0)
        for x in r["rows"]:
            if x["fell"]:
                e["falls"] += 1
                e["fall_list"].append([r["history"], r["index"], x["k"], x["entity"]])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True, help="robot=task_config.json (the robot's frozen values)")
    ap.add_argument("--task", default="handoff")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recs = load(a.runs)
    tasks = {kv.split("=")[0]: json.loads(Path(kv.split("=")[1]).read_text())["tasks"][a.task] for kv in a.tasks}
    res: dict = {"lifetimes": len(recs), "robots": {}}
    for robot in sorted({r["robot"] for r in recs}):
        task = tasks[robot]
        hyp, ps = {}, {}
        for hid, name, hist, scene, other, direction in HYPOTHESES:
            d = paired(recs, robot, scene, hist, other, name, task)
            p = perm_p(d, direction)
            hyp[hid] = {"vs": other, "measure": name, "scene": scene, "history": hist, "n_pairs": int(len(d)),
                        "mean_diff_DT_minus_other": float(d.mean()) if len(d) else None, "ci95": boot_ci(d),
                        "p_one_sided": p, "predicted": direction}
            ps[hid] = p
        sig = holm(ps)
        for hid, h in hyp.items():
            md = h["mean_diff_DT_minus_other"]
            wrong = md is not None and ((md > 0) if h["predicted"] == "less" else (md < 0))
            p_wrong = perm_p(paired(recs, robot, h["scene"], h["history"], h["vs"], h["measure"], task),
                             "greater" if h["predicted"] == "less" else "less")
            h["holm_pass"] = sig[hid]
            h["failed_wrong_direction"] = bool(wrong and np.isfinite(p_wrong) and p_wrong <= ALPHA)
        res["robots"][robot] = hyp
    E = {}
    for hid in ("A1", "B1", "B2", "C2", "D1"):
        E[hid] = sum(bool(res["robots"][rb][hid]["holm_pass"]) for rb in res["robots"])
    res["E_robots_passing"] = E
    verdict = {}
    for hid, *_ in HYPOTHESES:
        n = sum(bool(res["robots"][rb][hid]["holm_pass"]) for rb in res["robots"])
        failed = any(res["robots"][rb][hid]["failed_wrong_direction"] for rb in res["robots"])
        nrob = len(res["robots"])
        verdict[hid] = ("Failed" if failed and n == 0 else "Demonstrated" if n == nrob and nrob == 3
                        else "Partially demonstrated" if n > 0 else "Not demonstrated")
    res["verdicts"] = verdict
    res["safety"] = safety(recs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for robot, hyp in res["robots"].items():
        print(robot)
        for hid, h in hyp.items():
            md = h["mean_diff_DT_minus_other"]
            print(f"  {hid} DT vs {h['vs']:9} n={h['n_pairs']:2} diff={md if md is None else round(md, 3)} "
                  f"CI={np.round(h['ci95'], 3).tolist()} p={h['p_one_sided']:.4f} pass={h['holm_pass']}")
    print("verdicts", verdict)


if __name__ == "__main__":
    main()
