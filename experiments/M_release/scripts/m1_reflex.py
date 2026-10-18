"""M1: NERVA falls under clustered sub-cap pushes. Characterise them and choose a push reflex for the physical
safety layer.

    python m1_reflex.py --n-normal 30 --workers 60 --out out/m1_reflex.jsonl   (NERVA venv)

**Fall cases.** Every NERVA fall in programme L's test data, replayed exactly through `k_run.job`:
- LE-X 23 (handoff, E3 and E4);
- LE-X 22 (inspect, E2);
- LE-X 23 change_B (inspect, E1);
- LC4-X 24 (IMU + vision VOI).

**Normal operation.** LE-X indices 0..n-1, risky_B and safe, E3 on handoff. This measures what a stricter reflex
costs: trips, holds and interaction completion.

**Variants.** The independent monitor's stop thresholds (tilt in degrees, angular speed in rad/s). While tripped,
NERVA rests in its neutral pose and resumes after 0.5 s below 12°.
- K/L default: (25, 5.0);
- T20, T15, T12: tilt only;
- T15W4, T15W3: tilt and angular speed.

A local check showed that (10, 2.0) trips on normal gait in every encounter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
K = REPO / "experiments" / "K_recovery"
L = REPO / "experiments" / "L_audit_perception"
J = REPO / "experiments" / "J_validation_transfer"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(K / "scripts"))
VARIANTS = {"default": None, "T20": (20.0, 5.0), "T15": (15.0, 5.0), "T12": (12.0, 5.0), "T15W4": (15.0, 4.0),
            "T15W3": (15.0, 3.0)}
TUNED = json.loads((L / "configs" / "LE_tuned.json").read_text())["tuned"]
FALL_CASES = [("LE", 23, "risky_B", "DT_IMUVIS", "handoff"), ("LE", 23, "change_B", "DT_IMUVIS_VOI", "handoff"),
              ("LE", 22, "risky_B", TUNED["inspect"]["E2"], "inspect"),
              ("LE", 23, "change_B", TUNED["inspect"]["E1"], "inspect"),
              ("LC4", 24, "risky_B", "DT_IMUVIS_VOI", "handoff")]


def channels_file(variant: str, tmp: Path) -> str:
    base = json.loads((J / "configs" / "H_admitted_interface.json").read_text())
    if VARIANTS[variant] is not None:
        base["interface"]["stop_tilt_deg"], base["interface"]["stop_angular_speed"] = VARIANTS[variant]
    p = tmp / f"interface_{variant}.json"
    p.write_text(json.dumps(base))
    return str(p)


def run(args: tuple) -> dict:
    exp, index, history, system, task, variant, chan, kind = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import k_run
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())
    cfg = {"types_model": str(L / "configs" / "LC_nerva_vision_types.json"),
           "edges": str(J / "configs" / "J4_edges.json"),
           "time_model": str(J / "configs" / "J6_model.json"), "channels": chan, "task": task,
           "task_spec": tasks["tasks"][task], "voi": tasks["voi"], "n_enc": 40, "robot": "nerva",
           "transition_fix": True, "seeds": str(L / "seeds.py"), "vision": {}, "legacy_task_scoring": True}
    r = k_run.job((exp, "X", index, history, system, cfg))
    rows = r["rows"]
    b1 = [x for x in rows if x["entity"] == "B1"]
    return {"kind": kind, "variant": variant, "exp": exp, "index": index, "history": history, "system": system,
            "task": task, "fell": any(x["fell"] for x in rows),
            "fall_k": next((x["k"] for x in rows if x["fell"]), None),
            "encounters": len(rows), "trip_encounters": sum(x["trips"] > 0 for x in rows),
            "contacts": sum(x["contacts"] for x in rows), "holds_B1": sum(x["no_safe_action"] for x in b1),
            "hold_completed_calm": [x["hold_completed"] for x in rows if x["type"] == "calm"],
            "arrived": sum(x["arrived"] for x in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-normal", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    jobs = []
    for v in VARIANTS:
        ch = channels_file(v, tmp)
        jobs += [(e, i, h, s, t, v, ch, "fall_case") for e, i, h, s, t in FALL_CASES]
        jobs += [("LE", i, h, "DT_IMUVIS", "handoff", v, ch, "normal") for i in range(a.n_normal)
                 for h in ("risky_B", "safe")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p, open(a.out, "w", encoding="utf-8") as f:
        for r in p.imap_unordered(run, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
    rows = [json.loads(x) for x in a.out.read_text().splitlines()]
    for v in VARIANTS:
        fc = [r for r in rows if r["variant"] == v and r["kind"] == "fall_case"]
        nm = [r for r in rows if r["variant"] == v and r["kind"] == "normal"]
        hc = [c for r in nm for c in r["hold_completed_calm"]]
        print(f"{v:8} fall cases fell {sum(r['fell'] for r in fc)}/{len(fc)} | "
              f"normal: falls {sum(r['fell'] for r in nm)}"
              f"/{len(nm)}, trip encounters {sum(r['trip_encounters'] for r in nm)}/{sum(r['encounters'] for r in nm)},"
              f" contacts {sum(r['contacts'] for r in nm)}, calm hold completion {sum(hc) / max(len(hc), 1):.3f}")


if __name__ == "__main__":
    main()
