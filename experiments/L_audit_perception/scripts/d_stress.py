"""L protocol §D: behavioural learning when the entity's behaviour, the world or the sensing changes.

    python d_stress.py --n 30 --types-model configs/LC_wheeled_vision_types.json --out <jsonl>   (needs rendering)

**Body.** The wheeled robot with IMU + rendered vision, the decision-theoretic mapping with VOI (DT_IMUVIS_VOI's
policy), and all §B corrections. Its configs are K's; its type model is the vision model of §C.2.

**Memories:**
- M0: none (prior);
- M1: the Bayesian change-aware type memory (λ 0.02, fixed clock);
- M2: a recency-weighted (leaky) evidence memory (α 0.2);
- M3: M1 keyed by (identity, zone);
- ORACLE: acts on the true current type (reference).

**Seeds.** Every system meets the same entity draws: per-encounter uniforms are drawn once per lifetime.

Scenarios are in `SCENARIOS`. "Risky"/"safe" are the scenario's type pair (default pushy/calm).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
K = REPO / "experiments" / "K_recovery"
N_ENC, SWITCH = 40, 16
CAM_OFFSET, DIST_EDGES = 24, (0.3, 0.5)
ZONES = {"L": (1.0, 0.45), "R": (1.0, -0.45)}
POSITIONS = ((0.9, 0.0), (1.2, 0.5), (0.8, -0.6), (1.5, -0.3))

SCENARIOS: dict[str, dict] = {
    "nominal": {},
    "overlap": {"types": ("forceful", "gentle")},          # risky/safe types the model was never fitted on
    "drift": {"drift": (SWITCH, 32)},                      # gradual change over encounters 16-32
    "erratic": {"flip": 0.3},                              # the type flips at each B meeting with p 0.3
    "positions": {"positions": True},                      # the entity stands at one of 4 places
    "occlusion": {"occlusion": 0.4},                       # a screen hides most of the entity in 40% of encounters
    "identities": {"identities": 6},                       # 6 identities; B2 changes the opposite way
    "tracking": {"swap": 0.15},                            # identity confused with another in 15% of encounters
    "speed_fast": {"speed": 1.6},
    "speed_slow": {"speed": 0.5},
    "camera_poor": {"vision": {"width": 32, "height": 24, "depth_sigma": 0.03, "dropout": 0.10, "hz": 5.0}},
    "context": {"context": True},                          # behaviour depends on the zone (see `true_type`)
    "goal_inspect": {"task": "inspect"},
    "goal_separation": {"task": "separation"},
    "goal_waypoint": {"task": "waypoint"},
}
HISTORIES = ("change", "late")
SYSTEMS = ("M0_prior", "M1_bayes", "M2_leaky", "M3_context", "ORACLE")


def _seeds():
    spec = importlib.util.spec_from_file_location("l_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def entities(scn: dict) -> tuple[list[tuple[str, str]], list[float]]:
    if scn.get("identities") == 6:
        ents = [("A1", "A"), ("A2", "A"), ("B1", "B"), ("B2", "B"), ("C1", "C"), ("C2", "C")]
        return ents, [0.125, 0.125, 0.25, 0.25, 0.125, 0.125]
    return [("A1", "A"), ("B1", "B"), ("C1", "C")], [0.25, 0.5, 0.25]


def true_type(scn: dict, history: str, ident: str, k: int, zone: str, u: float, state: dict) -> str:
    """The entity's physical type in encounter k (B identities change; A and C are always the safe type)."""
    risky, safe = scn.get("types", ("pushy", "calm"))
    if not ident.startswith("B"):
        return safe
    h = history if ident == "B1" else ("late" if history == "change" else "change")
    first, second = (risky, safe) if h == "change" else (safe, risky)
    if scn.get("context"):  # change: risky everywhere, then safe in R only; late: safe, then risky in L only
        if k < SWITCH:
            return first
        return (safe if zone == "R" else risky) if h == "change" else (risky if zone == "L" else safe)
    if "drift" in scn:
        a, b = scn["drift"]
        p_first = 1.0 if k < a else 0.0 if k >= b else 1.0 - (k - a) / (b - a)
        return first if u < p_first else second
    if "flip" in scn:
        cur = state.setdefault(ident, first)
        if k >= 1 and u < scn["flip"]:
            cur = safe if cur == risky else risky
        state[ident] = cur
        return cur
    return first if k < SWITCH else second


def lifetime(args: tuple) -> dict:
    scn_name, history, index, system, cfg = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import apm_transfer.nerva_world as NW
    from apm_revise.belief import TypeModel
    from apm_transfer.expression_map import GRID, TimeModel
    from apm_transfer.infoseek import Emissions, Task, choose
    from apm_transfer.nerva_memory import ContextKeyedMemory, LeakyEvidenceMemory, PriorOnlyMemory, TypeStateModel
    from apm_transfer.nerva_world import MIN_QUIET_FRAMES
    from apm_transfer.vision import VisionCamera
    from apm_transfer.wheeled_world import closed_loop_encounter

    scn = SCENARIOS[scn_name]
    if "speed" in scn:  # faster or slower lunges for every type (impulses stay capped)
        for t in list(NW.TYPES):
            if "_x" not in t:
                lo, hi = NW.TYPES[t]["speed"]
                NW.TYPES[f"{t}_x{scn['speed']}"] = {**NW.TYPES[t], "speed": (lo * scn["speed"], hi * scn["speed"])}
    S = _seeds()
    base = int(S.world_seed("LD", "X", index).generate_state(1)[0])
    rng = np.random.default_rng([base, list(SCENARIOS).index(scn_name), HISTORIES.index(history)])
    ents, p_ent = entities(scn)
    order = [ents[int(i)] for i in rng.choice(len(ents), N_ENC, p=p_ent)]
    enc_seeds = rng.integers(0, 2**31, N_ENC)
    u_type, u_occ, u_swap = rng.random(N_ENC), rng.random(N_ENC), rng.random(N_ENC)
    swap_to = rng.integers(0, len(ents), N_ENC)
    zones = rng.choice(list(ZONES), N_ENC)
    pos = rng.integers(0, len(POSITIONS), N_ENC)
    decide_rng = np.random.default_rng(base + 3)

    mcfg = json.loads(Path(cfg["types_model"]).read_text())
    model = TypeModel.from_dict(mcfg["model"])
    em = Emissions.from_model(model.E, camera_block=True)
    edges = json.loads((K / "configs" / "K_wheeled_edges.json").read_text())["acc8"]
    tm = TimeModel(json.loads((K / "configs" / "K_wheeled_time.json").read_text())["time_coef"])
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())
    task_name = scn.get("task", "handoff")
    task = Task(task_name, **{k: v for k, v in tasks["tasks"][task_name].items() if k != "why"})
    voi = tasks["voi"]
    risky_t, safe_t = scn.get("types", ("pushy", "calm"))
    if system == "M0_prior":
        mem = PriorOnlyMemory(model)
    elif system == "M2_leaky":
        mem = LeakyEvidenceMemory(model, alpha=0.2)
    elif system == "M3_context":
        mem = ContextKeyedMemory(TypeStateModel(model, 0.02, transition_each_meeting=True))
    else:
        mem = TypeStateModel(model, 0.02, transition_each_meeting=True)
    cam0 = VisionCamera(**scn.get("vision", {}))
    state: dict = {}
    rows = []
    t0 = time.time()
    for k, (ident, kind) in enumerate(order):
        zone = str(zones[k])
        etype = true_type(scn, history, ident, k, zone, float(u_type[k]), state)
        risky = etype == risky_t
        perceived = ident
        if "swap" in scn and u_swap[k] < scn["swap"]:
            others = [e[0] for e in ents if e[0] != ident]
            perceived = others[int(swap_to[k]) % len(others)]
        if scn.get("context"):
            target = ZONES[zone]
        elif scn.get("positions"):
            target = POSITIONS[int(pos[k])]
        else:
            target = None
        if isinstance(mem, ContextKeyedMemory):
            mem.context = zone if scn.get("context") else (str(int(pos[k])) if scn.get("positions") else "")
        b = mem.belief(perceived, kind)
        if system == "ORACLE":
            bb = np.asarray(mcfg["oracle"]["pushy" if risky else "calm"])
            gi, w, diag = choose("greedy", bb, model.pi, tm, Emissions(em.imu, None), task, rng=decide_rng)
        else:
            gi, w, diag = choose("voi", b, model.pi, tm, em, task, lam=0.02, horizon=voi["horizon"],
                                 gap=voi["gap"], u_trigger=voi["u_trigger"], rng=decide_rng)
        intent = GRID[2] if gi < 0 else GRID[gi]
        standoff = diag.get("standoff", 0.0)
        cam = replace(cam0, occluded=bool("occlusion" in scn and u_occ[k] < scn["occlusion"]))
        phys = f"{etype}_x{scn['speed']}" if "speed" in scn else etype
        ev, out, truth = closed_loop_encounter(int(enc_seeds[k]), phys, intent, None, windows=w, camera=cam,
                                               standoff=standoff, target=target)
        d = out["final_distance"]
        band = int(np.searchsorted(DIST_EDGES, d))
        imu = [] if standoff > 0 else [int(np.searchsorted(edges, e[1], side="right")) for e in out["windows"]
                                       if e[10] >= MIN_QUIET_FRAMES]
        cams = list(out.get("camera", []))
        mem.observe_many(perceived, kind, [band * 8 + c for c in imu] + [CAM_OFFSET + c for c in cams], 0.0)
        rows.append({"k": k, "entity": ident, "perceived": perceived, "type": etype, "risky": risky, "zone": zone,
                     "target": target, "occluded": cam.occluded, "b_before": [round(float(v), 5) for v in b],
                     "grid_i": gi, "windows_planned": w, "windows_done": len(out["windows"]), "d_stop": d,
                     "band": band, "imu_cats": imu, "cam_codes": cams, "contacts": truth["contacts"],
                     "max_push": truth["max_push"], "arrived": out["arrived"], "t_arrive": out["t_arrive"],
                     "hold_completed": task.hold_completed(out["arrived"], out.get("hold_max_dist")),
                     "standoff": standoff, "no_safe_action": bool(diag.get("no_safe_action", False)),
                     "trips": out["trips"], "fell": out["fell"]})
        if out["fell"]:
            break
    return {"scenario": scn_name, "history": history, "index": index, "system": system, "task": task_name,
            "rows": rows, "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--types-model", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--addendum", default="", help="confirmatory run: committed protocol (clean tree required)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from apm_transfer.provenance import confirmatory_lock, environment, write_json
    prov = confirmatory_lock(HERE / a.addendum) if a.addendum else environment()
    cfg = {"types_model": a.types_model}
    jobs = [(s, h, a.first + i, sy, cfg) for i in range(a.n) for s in a.scenarios.split(",") for h in HISTORIES
            for sy in a.systems.split(",")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out.with_suffix(".meta.json"), {"provenance": prov, "args": {k: str(v) for k, v in vars(a).items()}})
    t0 = time.time()
    with Pool(a.workers) as p, open(a.out, "w", encoding="utf-8") as f:
        for r in p.imap_unordered(lifetime, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
    print(f"done {len(jobs)} lifetimes in {time.time() - t0:.1f} s -> {a.out}")


if __name__ == "__main__":
    main()
