"""L audit A7 (part 2): the humanoid controller's operating envelope, measured on a battery of conditions.

    python l_a7_humanoid_battery.py --seeds 20 --workers 3 --out results/A7_humanoid_battery.json   (hum venv)

Every run is 12 s of the plain-MuJoCo humanoid (`HumanoidRobot`) with no APM in the loop. The independent supervisor
(tilt > 25 deg -> zero command for 1 s) stays on, as in K6. Per condition:
- the fall rate (tilt > 50 deg or pelvis < 0.45 m) and supervisor trips;
- the tracking error over the last 6 s: mean heading-frame forward speed minus the command, and the same for yaw rate.

Conditions:
- speeds 0-0.5 m/s straight; turning at 0.2 m/s;
- stop/start every 1.5 s;
- perturbed initial joints and base velocity;
- repeated pushes every 1-2 s in random directions, at five magnitudes, while walking, turning and standing;
- single pushes at the training interval (5-10 s), at 0.5-1.0 m/s;
- training-level and doubled observation noise.

K's entity impulses are capped at 0.10 m/s. This battery establishes the margin beyond that cap. It is a simulation
envelope and does not certify the controller.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
T_RUN = 12.0


def conditions() -> dict[str, dict]:
    c: dict[str, dict] = {}
    for v in (0.0, 0.074, 0.15, 0.3, 0.5):
        c[f"speed_{v}"] = {"v": v}
    for w in (-1.0, -0.5, 0.5, 1.0):
        c[f"turn_{w}"] = {"v": 0.2, "w": w}
    c["stop_start_1.5s"] = {"v": 0.3, "stop_start": 1.5}
    for s in (0.05, 0.1):
        c[f"init_joint_noise_{s}"] = {"v": 0.2, "init": s}
    for m in (0.1, 0.3, 0.5, 0.8, 1.2):
        c[f"push_walk_{m}"] = {"v": 0.2, "push": m}
        c[f"push_turn_{m}"] = {"v": 0.2, "w": 0.8, "push": m}
        c[f"push_stand_{m}"] = {"v": 0.0, "push": m}
    for m in (0.5, 0.8, 1.0):  # the training recipe's push interval (5-10 s): one push per run, at 5-10 s
        c[f"push_sparse_walk_{m}"] = {"v": 0.2, "push": m, "interval": (5.0, 10.0)}
    for lv in (1.0, 2.0):
        c[f"obs_noise_{lv}"] = {"v": 0.2, "noise": lv}
        c[f"obs_noise_{lv}_push_0.3"] = {"v": 0.2, "noise": lv, "push": 0.3}
    return c


def run(args: tuple[str, dict, int]) -> dict:
    from apm_transfer.humanoid_world import HumanoidRobot
    name, cfg, seed = args
    rng = np.random.default_rng([seed, 2026, 1018])
    r = HumanoidRobot(seed, obs_noise=cfg.get("noise", 0.0))
    if cfg.get("init"):
        r.data.qpos[7:] += rng.normal(0, cfg["init"], r.model.nu)
        r.data.qvel[:2] += rng.uniform(-0.2, 0.2, 2)
        r.mj.mj_forward(r.model, r.data)
    n = int(T_RUN / r.dt)
    lo, hi = cfg.get("interval", (1.0, 2.0))
    next_push = rng.uniform(lo, hi)
    fell, max_tilt, vf, wz = False, 0.0, [], []
    pushes = 0
    for k in range(n):
        t = k * r.dt
        v, w = cfg.get("v", 0.0), cfg.get("w", 0.0)
        if cfg.get("stop_start") and int(t / cfg["stop_start"]) % 2 == 1:
            v, w = 0.0, 0.0
        r.apply(v, w)
        if cfg.get("push") and t >= next_push:
            a = rng.uniform(0, 2 * math.pi)
            r.push(cfg["push"] * math.cos(a), cfg["push"] * math.sin(a))
            pushes += 1
            next_push = t + rng.uniform(lo, hi)
        row = r.advance()
        max_tilt = max(max_tilt, row["tilt"])
        if r.fallen(row):
            fell = True
            break
        if t >= T_RUN / 2:
            _, _, yaw = r.pose()
            vx, vy = r.data.qvel[0], r.data.qvel[1]
            vf.append(vx * math.cos(yaw) + vy * math.sin(yaw))
            wz.append(r.data.qvel[5])
    track = not cfg.get("stop_start") and not cfg.get("push") and not fell
    return {"cond": name, "seed": seed, "fell": fell, "fall_t": round(k * r.dt, 2) if fell else None,
            "trips": r.trips, "max_tilt": round(max_tilt, 1), "pushes": pushes,
            "v_err": round(float(np.mean(vf)) - cfg.get("v", 0.0), 4) if track else None,
            "w_err": round(float(np.mean(wz)) - cfg.get("w", 0.0), 4) if track else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cs = conditions()
    jobs = [(name, cfg, s) for name, cfg in cs.items() for s in range(a.seeds)]
    with Pool(a.workers) as p:
        rows = p.map(run, jobs, chunksize=4)
    summary = {}
    for name, cfg in cs.items():
        rr = [x for x in rows if x["cond"] == name]
        ve = [x["v_err"] for x in rr if x["v_err"] is not None]
        we = [x["w_err"] for x in rr if x["w_err"] is not None]
        summary[name] = {"cfg": cfg, "n": len(rr), "falls": sum(x["fell"] for x in rr),
                         "trips_per_run": round(float(np.mean([x["trips"] for x in rr])), 2),
                         "max_tilt_deg_p95": round(float(np.percentile([x["max_tilt"] for x in rr], 95)), 1),
                         "v_err_mean": round(float(np.mean(ve)), 3) if ve else None,
                         "w_err_mean": round(float(np.mean(we)), 3) if we else None}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"t_run_s": T_RUN, "seeds": a.seeds, "summary": summary, "runs": rows}, indent=1),
                     encoding="utf-8")
    for name, s in summary.items():
        print(f"{name:26} falls {s['falls']:2}/{s['n']}  trips {s['trips_per_run']:.2f}"
              f"  tilt95 {s['max_tilt_deg_p95']:5.1f}  v_err {s['v_err_mean']}  w_err {s['w_err_mean']}")


if __name__ == "__main__":
    main()
