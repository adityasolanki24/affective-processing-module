"""L protocol §C.1: perception quality of the rendered-vision pipeline against the abstract camera and ground truth.

    python c_perception.py --robot nerva --experiment LC1 --partition X --n 400 --workers 120 --out <jsonl>
    python c_perception.py --analyse <jsonl> [<jsonl> ...] --out results/C_perception.json

**Encounters.**
- Single encounters, alternating calm and pushy.
- The robot stops at 0.3, 0.4 or 0.6 m, or at the 0.9 m stand-off. It senses for 3 windows, never leaving early.
- Each encounter is simulated twice with the same seed: once with the rendered camera (`apm_transfer.vision`), once
  with the abstract camera. Physics, entity behaviour and robot motion are identical between the two; only the
  sensor differs.

**Metrics** (truth is used here only):
- **Recall:** the share of *visible* lunges matched by a perceived event within ±0.4 s. Visible means in view and
  displacing the column by at least 5 cm, the depth threshold. Below that, no depth sensor at this noise level can
  see the motion. Recall over all moving lunges is reported too.
- **False events:** perceived events matching no lunge, per 4 s window.
- **Reach error:** perceived minus visible reach, for matched events. A lunge that reaches the robot can look no
  longer than the gap to the robot.
- **Information:** KL(calm ‖ pushy) of the per-window code, in nats, per stopping distance, for vision and for the
  abstract camera. Codes use add-0.5 smoothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parents[1]
STOPS = ((14, 0.0, "0.3"), (10, 0.0, "0.4"), (2, 0.0, "0.6"), (2, 0.3, "0.9"))  # (grid index, stand-off, label)
TOL_S = 0.4
MIN_DISP = 0.05  # m: the perception's depth threshold (VisionCamera.tau)


def _seeds():
    spec = importlib.util.spec_from_file_location("l_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def one(args: tuple) -> dict:
    robot, exp, part, i = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from apm_transfer.expression_map import GRID
    from apm_transfer.nerva_world import Camera
    from apm_transfer.vision import VisionCamera
    kw: dict = {}
    if robot == "nerva":
        from apm_transfer.nerva_world import closed_loop_encounter as enc
        cfg = REPO / "experiments" / "J_validation_transfer" / "configs" / "H_admitted_interface.json"
        kw["channels"] = json.loads(cfg.read_text())["interface"]
        dt = 0.02
    elif robot == "wheeled":
        from apm_transfer.wheeled_world import DT as dt
        from apm_transfer.wheeled_world import closed_loop_encounter as enc
    else:
        from apm_transfer.humanoid_world import DT as dt
        from apm_transfer.humanoid_world import closed_loop_encounter as enc
    base = int(_seeds().world_seed(exp, part, i).generate_state(1)[0])
    etype = "pushy" if i % 2 else "calm"
    gi, standoff, label = STOPS[(i // 2) % len(STOPS)]
    seed = int(np.random.default_rng(base).integers(2**31))
    _, out_v, truth_v = enc(seed, etype, GRID[gi], windows=3, camera=VisionCamera(), abort_on_detect=False,
                            standoff=standoff, **kw)
    _, out_a, truth_a = enc(seed, etype, GRID[gi], windows=3, camera=Camera(), abort_on_detect=False,
                            standoff=standoff, **kw)
    return {"robot": robot, "index": i, "type": etype, "stop": label, "dt": dt, "d_stop": out_v["final_distance"],
            "vision_codes": out_v["camera"], "abstract_codes": out_a["camera"], "vision": out_v["vision"],
            "abstract_reaches": out_a["camera_reaches"], "same_world": truth_v["pushes"] == truth_a["pushes"],
            "contacts": truth_v["contacts"], "fell": out_v["fell"]}


def kl(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = (p_counts + 0.5) / (p_counts + 0.5).sum()
    q = (q_counts + 0.5) / (q_counts + 0.5).sum()
    return float(np.sum(p * np.log(p / q)))


def analyse(rows: list[dict]) -> dict:
    from apm_transfer.vision import ENTITY_R
    out: dict = {}
    for robot in sorted({r["robot"] for r in rows}):
        rr = [r for r in rows if r["robot"] == robot]
        o: dict = {"encounters": len(rr), "same_world_share": float(np.mean([r["same_world"] for r in rr])),
                   "falls": int(sum(r["fell"] for r in rr))}
        vis_n = matched = false = windows = moving = moving_hit = 0
        err: list[float] = []
        for r in rr:
            tol = int(round(TOL_S / r["dt"]))
            for truth, events in zip(r["vision"]["truth"], r["vision"]["events"], strict=True):
                windows += 1
                used = set()
                for t in truth:
                    if t["in_view"] and t["moves"]:
                        moving += 1
                        moving_hit += any(e["k0"] - tol <= t["k"] <= e["k1"] + tol for e in events)
                    if not (t["in_view"] and t["visible_reach"] - ENTITY_R >= MIN_DISP):
                        continue
                    vis_n += 1
                    for j, e in enumerate(events):
                        if j not in used and e["k0"] - tol <= t["k"] <= e["k1"] + tol:
                            used.add(j)
                            matched += 1
                            err.append(e["reach_est"] - t["visible_reach"])
                            break
                for j, e in enumerate(events):
                    if j in used:
                        continue
                    if not any(e["k0"] - tol <= t["k"] <= e["k1"] + tol for t in truth):
                        false += 1
        o["recall"] = matched / max(vis_n, 1)
        o["visible_lunges"] = vis_n
        o["recall_all_moving"] = moving_hit / max(moving, 1)
        o["false_events_per_window"] = false / max(windows, 1)
        o["reach_error"] = {"bias": float(np.mean(err)) if err else None,
                            "mae": float(np.mean(np.abs(err))) if err else None, "n": len(err)}
        info = {}
        for stop in sorted({r["stop"] for r in rr}):
            s = [r for r in rr if r["stop"] == stop]
            res = {}
            for key in ("vision_codes", "abstract_codes"):
                cnt = {t: np.bincount([c for r in s if r["type"] == t for c in r[key]], minlength=8).astype(float)
                       for t in ("calm", "pushy")}
                res[key.split("_")[0]] = {"kl_calm_pushy": round(kl(cnt["calm"], cnt["pushy"]), 4),
                                          "calm": cnt["calm"].astype(int).tolist(),
                                          "pushy": cnt["pushy"].astype(int).tolist()}
            info[stop] = res
        o["information_nats_per_window"] = info
        out[robot] = o
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=("nerva", "wheeled", "humanoid"))
    ap.add_argument("--experiment")
    ap.add_argument("--partition", default="X")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--analyse", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.analyse:
        rows = [json.loads(line) for p in a.analyse for line in p.read_text(encoding="utf-8").splitlines()]
        res = analyse(rows)
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        for robot, o in res.items():
            print(robot, {k: v for k, v in o.items() if k != "information_nats_per_window"})
            for stop, inf in o["information_nats_per_window"].items():
                print(f"   stop {stop}: KL vision {inf['vision']['kl_calm_pushy']:.3f}  "
                      f"abstract {inf['abstract']['kl_calm_pushy']:.3f}")
        return
    jobs = [(a.robot, a.experiment, a.partition, a.first + i) for i in range(a.n)]
    with Pool(a.workers) as p, open(a.out, "w", encoding="utf-8") as f:
        for r in p.imap_unordered(one, jobs, chunksize=2):
            f.write(json.dumps(r) + "\n")
    print("done", a.n, "->", a.out)


if __name__ == "__main__":
    main()
