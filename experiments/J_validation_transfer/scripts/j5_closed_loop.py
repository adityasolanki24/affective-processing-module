"""J5: closed-loop expressive development on NERVA (MASTER_PREREGISTRATION §4.3). Needs NERVA's venv.

    python j5_closed_loop.py --partition P --n 4 --types-model <J5_types_distance.json> --edges <J4_edges.json>
        --lam L --systems CL,FROZEN,... --histories risky_B,change_B,late_B,safe --out <jsonl>

**A lifetime** is `N_ENC` encounters with three tagged entities (A1, B1, C1), met in a seeded random order. Each
entity's physical type follows the history:
- `risky_B`: B1 is pushy throughout;
- `change_B`: pushy, then calm from encounter `SWITCH`;
- `late_B`: calm, then pushy;
- `safe`: everyone calm.

Seeds are shared across systems (paired). Every encounter is a full NERVA closed-loop approach
(`nerva_world.closed_loop_encounter`) under the intent the system chooses.

**Systems** (all with the stopping-distance-conditioned NERVA type model, K by BIC, and H's L0 lookup):

| system | memory | intent |
|---|---|---|
| `CL` | updated from every encounter | L0(readout) |
| `FROZEN` | updated the same way | always neutral (expression frozen) |
| `NOMEM` | never updated | L0(prior readout) |
| `NOFEEDBACK` | updated only for the first `FEEDBACK_UNTIL` encounters | L0 |
| `RANDHIST` | each encounter's evidence credited to a random entity (history decoupled from identity) | L0 |
| `H_MEM` | H's `EntityMemory` on the robot-observable detector | L0 |

Per encounter, logged:
- the entity, its type (truth) and the readout before the encounter;
- the intent's caution and proximity;
- stopping distance;
- evidence (acc8 category, detected, peak acceleration);
- contacts and maximum push (truth);
- arrival time, trips and falls.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
N_ENC, SWITCH, FEEDBACK_UNTIL = 30, 15, 10
ENTITIES = (("A1", "A"), ("B1", "B"), ("C1", "C"))
DIST_EDGES = (0.3, 0.5)  # stopping-distance bands 0.2 / 0.4 / 0.6 m


def _seeds():
    spec = importlib.util.spec_from_file_location("j_seeds", HERE / "seeds.py")
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
    partition, index, history, system, model_path, edges_path, lam, channels = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from apm_express.states import EntityMemory, Probe, caution, lookup_intent
    from apm_express.styles import NEUTRAL
    from apm_revise.belief import TypeModel
    from apm_transfer.nerva_memory import ObsX, TypeStateModel
    from apm_transfer.nerva_world import closed_loop_encounter

    S = _seeds()
    base = int(S.world_seed("J5", partition, index).generate_state(1)[0])
    rng = np.random.default_rng(base)
    order = [ENTITIES[int(i)] for i in rng.integers(0, 3, N_ENC)]
    enc_seeds = rng.integers(0, 2**31, N_ENC)
    model = TypeModel.from_dict(json.loads(Path(model_path).read_text()))
    edges = json.loads(Path(edges_path).read_text())["acc8"]
    mem = EntityMemory() if system == "H_MEM" else TypeStateModel(model, lam, kind_prior=False)
    prior = mem.readout(Probe("Z9", "Z", "plain"))
    shuffle = np.random.default_rng(base + 1)
    rows = []
    t0 = time.time()
    for k, (ident, kind) in enumerate(order):
        etype = btype(history, k) if ident == "B1" else "calm"
        rd = mem.readout(Probe(ident, kind, "plain"))
        if system == "FROZEN":
            intent = NEUTRAL
        elif system == "NOMEM":
            intent, rd = lookup_intent(prior), prior
        else:
            intent = lookup_intent(rd)
        ev, out, truth = closed_loop_encounter(int(enc_seeds[k]), etype, intent, channels)
        e = ev.as_list()
        d = out["final_distance"]
        band = int(np.searchsorted(DIST_EDGES, d))
        cat = int(np.searchsorted(edges, e[1], side="right"))
        logp = None
        if isinstance(mem, TypeStateModel):  # self-supervised predictive log-probability of this evidence
            b = mem._at(ident) if ident in mem.b else mem._prior(kind)
            logp = round(float(np.log(max(1e-12, float(b @ model.E[:, band * 8 + cat])))), 5)
        learn = system not in ("NOMEM",) and not (system == "NOFEEDBACK" and k >= FEEDBACK_UNTIL)
        if learn:
            who, wkind = (ENTITIES[int(shuffle.integers(3))] if system == "RANDHIST" else (ident, kind))
            category = cat if system == "H_MEM" else band * 8 + cat
            mem.observe(ObsX(who, wkind, "plain", float(bool(e[9])), float(e[1]), category))
        rows.append({"k": k, "entity": ident, "type": etype, "p_before": round(rd.p, 5), "u_before": round(rd.u, 4),
                     "caution": round(caution(rd), 4), "proximity": round(intent.proximity, 3),
                     "hesitation": round(intent.hesitation, 3), "d_stop": d, "band": band, "category": cat,
                     "detected": bool(e[9]), "peak_acc": e[1], "logp": logp, "contacts": truth["contacts"],
                     "contacts_dwell": truth["contacts_dwell"], "max_push": truth["max_push"],
                     "arrived": out["arrived"], "t_arrive": out["t_arrive"], "trips": out["trips"],
                     "fell": out["fell"]})
    return {"partition": partition, "index": index, "history": history, "system": system, "rows": rows,
            "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--types-model", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--systems", default="CL,FROZEN,NOMEM,NOFEEDBACK,RANDHIST,H_MEM")
    ap.add_argument("--histories", default="risky_B,change_B,late_B,safe")
    ap.add_argument("--channels", default=str(HERE.parent / "H_expressive_intelligence" / "results" / "H0" / "X"
                                              / "admitted_interface.json"))
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    channels = json.loads(Path(a.channels).read_text())["interface"]
    jobs = [(a.partition, a.first + i, h, s, a.types_model, a.edges, a.lam, channels)
            for i in range(a.n) for h in a.histories.split(",") for s in a.systems.split(",")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with Pool(a.workers) as pool, open(a.out, "w", encoding="utf-8") as fh:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
    print("done", len(jobs), "lifetimes in", round(time.time() - t, 1), "s ->", a.out)


if __name__ == "__main__":
    main()
