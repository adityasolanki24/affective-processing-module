"""Programme K closed-loop NERVA lifetimes: recovery from negative experience (K2/K3) and task studies (K3b).

    python k_run.py --experiment K3 --partition X --first 0 --n 50 --systems DT,DT_VOI,... --histories ... \
        --task handoff --types-model configs/K_types.json --out <jsonl> [--addendum PROTOCOL.md]

**Lifetime.** 40 encounters. Each encounter meets B1 (probability 0.5), A1 or C1 (0.25 each); only B1's behaviour
depends on the history. Histories (B1's physical type before / after encounter 16):

| history | before | after |
|---|---|---|
| risky_B | pushy | pushy |
| safe | calm | calm |
| change_B | pushy | calm |
| late_B | calm | pushy |

**Robots.** `--robot nerva` (default): the NERVA biped (`nerva_world`). `--robot wheeled`: the MuJoCo
differential-drive robot (`wheeled_world`), with its own acc8 edges, time model and type model fitted on its own
exploration (`k_fit.py --robot wheeled`). The APM core (memory, decision layer) is the same code.

**Systems.** All share the world, the memory class (`TypeStateModel`, EM types fitted on K-T) and the seeds. Only the
decision policy, the forgetting rate and the sensing differ:

| system | intent | λ | sensing | information gathering |
|---|---|---|---|---|
| L0 | H's fixed lookup | 0.02 | IMU | none |
| DT_STATIC | decision-theoretic, task values frozen | 0 | IMU | none |
| DT | same | 0.02 | IMU | none |
| DT_VOI | same + value of information | 0.02 | IMU | protected observation windows, approach geometry |
| DT_UNC | same, w = 3 when belief entropy > 0.6 | 0.02 | IMU | entropy-triggered windows |
| ORACLE | DT on the canonical belief of the *true current* type | – | – | evaluation reference only |
| DT_CAM | DT | 0.02 | IMU + camera extension | none (one window) |
| DT_CAM_VOI | DT_VOI | 0.02 | IMU + camera extension | as DT_VOI, plus OBSERVE: watch from 0.9 m, out of reach |
| RANDINTENT | random grid intent, random windows 1–3 | 0.02 | IMU + camera | exploration data for fitting (K-T) |

**Memory updates** come only from what the robot observed:
- for each sensing window with ≥ 25 quiet frames, an IMU category `band · 8 + acc8`, with the band from the
  robot's own stopping distance;
- with the camera, a camera code per window (column 24 + code).

Nothing observed means nothing learned. Lunge streams are drawn for the longest possible encounter, so every system
meets the same entity behaviour, given the same seed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
J = REPO / "experiments" / "J_validation_transfer"
N_ENC, SWITCH = 40, 16
ENTITIES = (("A1", "A"), ("B1", "B"), ("C1", "C"))
P_ENTITY = (0.25, 0.5, 0.25)
DIST_EDGES = (0.3, 0.5)
CAM_OFFSET = 24
SYSTEMS = {
    "L0": {"policy": "L0", "lam": 0.02, "camera": False},
    "DT_STATIC": {"policy": "greedy", "lam": 0.0, "camera": False},
    "DT": {"policy": "greedy", "lam": 0.02, "camera": False},
    "DT_VOI": {"policy": "voi", "lam": 0.02, "camera": False},
    "DT_UNC": {"policy": "uncertain", "lam": 0.02, "camera": False},
    "ORACLE": {"policy": "oracle", "lam": 0.02, "camera": False},
    "DT_CAM": {"policy": "greedy", "lam": 0.02, "camera": True},
    "DT_CAM_VOI": {"policy": "voi", "lam": 0.02, "camera": True},
    "RANDINTENT": {"policy": "random", "lam": 0.02, "camera": True},
}
RENDER = None  # k_demo.py only: callable(...) -> clip with kwargs()/close(row), or None


def _seeds():
    spec = importlib.util.spec_from_file_location("k_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def btype(history: str, k: int) -> str:
    if history == "risky_B":
        return "pushy"
    if history == "change_B":
        return "pushy" if k < SWITCH else "calm"
    if history == "late_B":
        return "calm" if k < SWITCH else "pushy"
    return "calm"


def job(args: tuple) -> dict:
    experiment, partition, index, history, system, cfg = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from apm_express.states import Probe, lookup_intent
    from apm_revise.belief import TypeModel
    from apm_transfer.expression_map import GRID, TimeModel, band_of
    from apm_transfer.infoseek import OBSERVE, Emissions, Task, choose
    from apm_transfer.nerva_memory import TypeStateModel
    from apm_transfer.nerva_world import MIN_QUIET_FRAMES, Camera
    if cfg.get("robot", "nerva") == "wheeled":
        from apm_transfer.wheeled_world import closed_loop_encounter
    else:
        from apm_transfer.nerva_world import closed_loop_encounter

    spec = SYSTEMS[system]
    S = _seeds()
    base = int(S.world_seed(experiment, partition, index).generate_state(1)[0])
    rng = np.random.default_rng(base)
    order = [ENTITIES[int(i)] for i in rng.choice(3, N_ENC, p=P_ENTITY)][:cfg.get("n_enc", N_ENC)]
    enc_seeds = rng.integers(0, 2**31, N_ENC)
    explore = np.random.default_rng(base + 2)
    decide_rng = np.random.default_rng(base + 3)
    mcfg = json.loads(Path(cfg["types_model"]).read_text())
    if "model" not in mcfg:  # a plain J type model (IMU blocks only): bootstrap for RANDINTENT exploration
        mcfg = {"model": mcfg, "oracle": {}}
    model = TypeModel.from_dict(mcfg["model"])
    em = Emissions.from_model(model.E, camera_block=model.E.shape[1] > 24)
    em_imu = Emissions(em.imu, None)
    edges = json.loads(Path(cfg["edges"]).read_text())["acc8"]
    tm = TimeModel(json.loads(Path(cfg["time_model"]).read_text())["time_coef"])
    task = Task(cfg["task"], **{k: v for k, v in cfg["task_spec"].items() if k != "why"})
    voi_cfg = cfg["voi"]
    nerva = cfg.get("robot", "nerva") == "nerva"
    channels = json.loads(Path(cfg["channels"]).read_text())["interface"] if nerva else None
    mem = TypeStateModel(model, spec["lam"], kind_prior=False)
    camera = Camera() if spec["camera"] else None
    rows = []
    t0 = time.time()
    for k, (ident, kind) in enumerate(order):
        etype = btype(history, k) if ident == "B1" else "calm"
        b = mem.belief(ident, kind)
        rd = mem.readout(Probe(ident, kind, "plain"))
        diag: dict = {}
        grid_i = None
        standoff = 0.0
        if spec["policy"] == "L0":
            intent, w = lookup_intent(rd), 1
        elif spec["policy"] == "random":
            grid_i, w = int(explore.integers(len(GRID))), int(explore.integers(1, 4))
            intent = GRID[grid_i]
        else:
            bb = np.asarray(mcfg["oracle"][etype]) if spec["policy"] == "oracle" else b
            pol = "greedy" if spec["policy"] == "oracle" else spec["policy"]
            grid_i, w, diag = choose(pol, bb, model.pi, tm, em if spec["camera"] else em_imu, task, lam=spec["lam"],
                                     horizon=voi_cfg["horizon"], gap=voi_cfg["gap"], u_trigger=voi_cfg["u_trigger"],
                                     rng=decide_rng)
            intent = GRID[2] if grid_i == OBSERVE else GRID[grid_i]
        standoff = diag.get("standoff", 0.0)
        clip = RENDER(system, history, index, k, ident, etype, b, rd, intent, w) if RENDER is not None else None
        ev, out, truth = closed_loop_encounter(int(enc_seeds[k]), etype, intent, channels, windows=w, camera=camera,
                                               standoff=standoff, **(clip.kwargs() if clip is not None else {}))
        d = out["final_distance"]
        band = int(np.searchsorted(DIST_EDGES, d))
        imu = [int(np.searchsorted(edges, e[1], side="right")) for e in out["windows"] if e[10] >= MIN_QUIET_FRAMES]
        cams = list(out.get("camera", []))
        if standoff > 0:
            imu = []  # out of every type's reach: IMU silence is not evidence of safety
        cats = [band * 8 + c for c in imu] + ([CAM_OFFSET + c for c in cams] if em.cam is not None else [])
        detected = any(bool(e[9]) for e in out["windows"] if e[10] >= MIN_QUIET_FRAMES)
        mem.observe_many(ident, kind, cats, float(detected))
        post = mem.belief(ident, kind)
        rows.append({"k": k, "entity": ident, "type": etype, "b_before": [round(float(v), 5) for v in b],
                     "b_after": [round(float(v), 5) for v in post], "p_before": round(rd.p, 5),
                     "u_before": round(rd.u, 4), "grid_i": grid_i, "proximity": round(intent.proximity, 3),
                     "tempo": round(intent.tempo, 3), "hesitation": round(intent.hesitation, 3),
                     "windows_planned": w, "windows_done": len(out["windows"]), "aborted": out["aborted"],
                     "imu_cats": imu, "cam_codes": cams,
                     "imu_peaks": [round(e[1], 5) for e in out["windows"] if e[10] >= MIN_QUIET_FRAMES],
                     "cam_reaches": out.get("camera_reaches"),
                     "informative": len(cats), "detected": detected, "d_stop": d, "band": band,
                     "intended_band": band_of(intent.proximity), "contacts": truth["contacts"],
                     "contacts_dwell": truth["contacts_dwell"], "max_push": truth["max_push"],
                     "arrived": out["arrived"], "t_arrive": out["t_arrive"], "dwell_s": out["dwell_s"],
                     "trips": out["trips"], "fell": out["fell"], "task_success": task.p_success(d),
                     "voi": round(diag.get("voi", 0.0), 3), "standoff": standoff})
        if clip is not None:
            clip.close(rows[-1])
        if out["fell"]:
            break  # a fall ends the lifetime; it is counted
    return {"experiment": experiment, "partition": partition, "index": index, "history": history, "system": system,
            "robot": cfg.get("robot", "nerva"),
            "task": cfg["task"], "rows": rows, "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="K3")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--histories", default="risky_B,safe,change_B,late_B")
    ap.add_argument("--task", default="handoff")
    ap.add_argument("--tasks-config", default=str(HERE / "configs" / "K_tasks.json"))
    ap.add_argument("--types-model", default=str(HERE / "configs" / "K_types.json"))
    ap.add_argument("--edges", default=str(J / "configs" / "J4_edges.json"))
    ap.add_argument("--time-model", default=str(J / "configs" / "J6_model.json"))
    ap.add_argument("--channels", default=str(J / "configs" / "H_admitted_interface.json"))
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--n-enc", type=int, default=N_ENC, help="shorter lifetimes, for smoke tests only")
    ap.add_argument("--robot", choices=("nerva", "wheeled"), default="nerva")
    ap.add_argument("--addendum", default="", help="confirmatory run: committed protocol (clean tree required)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    from apm_transfer.provenance import confirmatory_lock, environment, write_json
    prov = confirmatory_lock(HERE / a.addendum) if a.addendum else environment()
    tasks = json.loads(Path(a.tasks_config).read_text())
    cfg = {"types_model": a.types_model, "edges": a.edges, "time_model": a.time_model, "channels": a.channels,
           "task": a.task, "task_spec": tasks["tasks"][a.task], "voi": tasks["voi"], "n_enc": a.n_enc,
           "robot": a.robot}
    jobs = [(a.experiment, a.partition, a.first + i, h, s, cfg) for i in range(a.n) for h in a.histories.split(",")
            for s in a.systems.split(",")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out.with_suffix(".meta.json"), {"provenance": prov, "args": {k: str(v) for k, v in vars(a).items()},
                                                 "cfg": cfg})
    t = time.time()
    with Pool(a.workers) as pool, open(a.out, "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
    print("done", len(jobs), "lifetimes in", round(time.time() - t, 1), "s ->", a.out)


if __name__ == "__main__":
    main()
