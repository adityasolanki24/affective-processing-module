"""Physically measured expression in the matched present task (from recorded traces, not plans).

    python experiments/M_release/scripts/expression_measures.py --records <expression_v2 *_<seed>.json> ... \
        --out results/remediation/expression_measures.json

Per present encounter and condition (neutral, or memory after a given history):
- observe_s: time standing and watching before moving;
- heading_sweep_rad: range of body heading while observing (wheeled gaze);
- head_yaw_span_rad: range of head yaw over the encounter (T1, NERVA);
- guard and open posture amplitude while interacting (T1: arm and waist offsets; NERVA: neck/head pitch);
- approach straightness and mean speed (the robot's own pose trace);
- stopping gap to the person's surface (measured);
- retreat_m and turn_s when leaving; total encounter time; falls.

Matching check: memory conditions must not differ from neutral by more than 0.05 m/s in approach speed. The
stopping-gap difference is reported (command-matched, not exactly realised).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def measures(row: dict, robot: str) -> dict:
    e = row.get("expression") or {}
    tr = e.get("trace") or []
    ph = np.array([t["phase"] for t in tr])
    yaw = np.unwrap(np.array([t["pose"][2] for t in tr])) if tr else np.zeros(0)
    upper = [t["upper"] for t in tr]
    out = {"observe_s": e.get("observe_s"), "fell": row.get("fell")}
    obs = ph == "observe" if len(ph) else np.zeros(0, bool)
    out["heading_sweep_rad"] = float(np.ptp(yaw[obs])) if obs.any() else 0.0
    if upper and len(upper[0]):
        U = np.array(upper, float)
        inter = U[ph == "interact"] if (ph == "interact").any() else U[:0]
        if robot == "humanoid":
            out["head_yaw_span_rad"] = float(np.ptp(U[:, 0]))
            out["guard_posture_rad"] = float(np.mean(np.abs(inter[:, [2, 4, 6, 8]]))) if len(inter) else 0.0
            out["torso_turn_rad"] = float(np.mean(np.abs(inter[:, 10]))) if len(inter) else 0.0
        else:  # NERVA: neck_pitch, head_pitch, head_yaw, head_roll
            out["head_yaw_span_rad"] = float(np.ptp(U[:, 2]))
            out["neck_pitch_rad"] = float(np.mean(inter[:, 0])) if len(inter) else 0.0
            out["head_pitch_rad"] = float(np.mean(inter[:, 1])) if len(inter) else 0.0
            out["head_roll_rad"] = float(np.mean(np.abs(inter[:, 3]))) if len(inter) else 0.0
    m = row.get("motion") or {}
    out["approach_straightness"] = m.get("straightness")
    out["approach_speed_m_s"] = m.get("mean_speed")
    md = row.get("measured_distance") or {}
    out["stop_gap_m"] = md.get("radial_surface_gap_m")
    dep = e.get("depart") or {}
    out["retreat_m"], out["turn_s"] = dep.get("retreat_m"), dep.get("turn_s")
    out["encounter_s"] = round(tr[-1]["t"] - tr[0]["t"], 2) if tr else None
    return out


def summarise(vals: list[dict]) -> dict:
    keys = sorted({k for v in vals for k in v if k != "fell"})
    s = {k: round(float(np.mean([v[k] for v in vals if v.get(k) is not None])), 3)
         for k in keys if any(v.get(k) is not None for v in vals)}
    s["falls"] = int(sum(bool(v.get("fell")) for v in vals))
    s["n"] = len(vals)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res: dict = {}
    for f in a.records:
        d = json.loads(f.read_text())
        robot, hist = d["robot"], d["history"]
        r = res.setdefault(robot, {"learned": {}, "conditions": {}})
        r["learned"][hist] = {k: round(v, 3) for k, v in d["learned_state"].items()}
        for cond, rows in d["present"].items():
            name = "neutral" if cond == "neutral" else f"memory_after_{hist}"
            r["conditions"].setdefault(name, []).extend(measures(x, robot) for x in rows)
    for r in res.values():
        r["summary"] = {c: summarise(v) for c, v in r["conditions"].items()}
        neu = r["summary"].get("neutral", {})
        r["matching"] = {c: {"approach_speed_diff_m_s": round(s.get("approach_speed_m_s", 0) -
                                                             neu.get("approach_speed_m_s", 0), 3),
                             "stop_gap_diff_m": round((s.get("stop_gap_m") or 0) - (neu.get("stop_gap_m") or 0), 3)}
                         for c, s in r["summary"].items() if c != "neutral"}
        del r["conditions"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for robot, r in res.items():
        print("==", robot, "learned", r["learned"])
        for c, s in r["summary"].items():
            print(f"  {c:22}", s)
        print("  matching", r["matching"])


if __name__ == "__main__":
    main()
