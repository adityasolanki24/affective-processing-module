"""Compact learning-and-recovery traces from the demo records (for the figures and the report).

    python experiments/M_release/scripts/demo_traces.py --demo humanoid=<..._demo.json> nova_carter=<...> \
        nerva=<...> [--failed humanoid=<..._demo.json> ...] --out results/remediation/demo_traces.json

Per meeting:
- the person's behaviour (the demo's script, pushy then calm);
- the robot's learned wariness of that person before and after the meeting, read from its own experience memory;
- the disturbance and movement evidence it holds;
- whether it fell;
- contacts (evaluation truth).
Failed attempts are listed with the meeting at which they ended.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows_of(path: Path) -> dict:
    d = json.loads(path.read_text())
    demo = d["demo"]
    rows = []
    for r in demo["rows"]:
        b, a = r["readout_before"], r["readout_after"]
        rows.append({"meeting": r["k"] + 1, "person": r["person"],
                     "wariness_before": round(b["wariness"], 4), "wariness_after": round(a["wariness"], 4),
                     "disturbance_p_after": round(a["disturbance_p"], 4), "movement_p_after": round(a["movement_p"], 4),
                     "fell": r["fell"], "contacts_truth": r["contacts_truth"]})
    return {"seed": d["seed"], "pushy_n": demo["pushy_n"], "calm_n": demo["calm_n"],
            "continuous_physics": demo.get("continuous_physics", True),
            "repositioned_meetings": demo.get("repositioned_meetings", False), "meetings": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", nargs="+", required=True, help="robot=path")
    ap.add_argument("--failed", nargs="*", default=[], help="robot=path of attempts that ended early")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out: dict = {"source": "isaac/expression_v2.py --demo (Isaac Sim 5.1 / PhysX); DEVIATIONS D8", "robots": {},
                 "failed_attempts": []}
    for item in a.demo:
        robot, path = item.split("=", 1)
        out["robots"][robot] = rows_of(Path(path))
    for item in a.failed:
        robot, path = item.split("=", 1)
        t = rows_of(Path(path))
        out["failed_attempts"].append({"robot": robot, "seed": t["seed"],
                                       "repositioned_meetings": t["repositioned_meetings"],
                                       "ended_at_meeting": len(t["meetings"]),
                                       "fell": any(m["fell"] for m in t["meetings"])})
    a.out.write_text(json.dumps(out, indent=1))
    for robot, t in out["robots"].items():
        print(robot, [m["wariness_after"] for m in t["meetings"]])


if __name__ == "__main__":
    main()
