"""L protocol §F: executable demo of the integrated runtime on a body.

    python f_runtime_demo.py --body wheeled --out results/F_runtime_wheeled.json   (NERVA needs its sim venv)

One lifetime: 24 encounters with entities A1, B1 and C1. B1 is pushy until encounter 12, then calm.
- **Runtime R1** drives the body. Each decision goes through the body's own expression adapter and safety monitor;
  the body's sensing adapter codes what it sensed into an `Observation`; R1 learns from it.
- **Runtime R2** receives exactly R1's inputs. At encounter 10 it is saved, deleted and rebuilt from the saved file
  alone.
- **AffectLink.** Identities arrive as AffectLink `EntityObservation` messages. The advisory outputs are AffectLink
  messages, each validated and round-tripped through the codec.

Checks:
- F1: the same runtime code runs on this body;
- F2: R2's decisions are identical to R1's (restart persistence);
- F3: `replay(R1's log)` is identical;
- F4: every AffectLink message validates and round-trips;
- F5: no executed approach has predicted risk above the cap, and the supervisor's veto count is reported;
- F6: every intent is within bounds;
- reported: falls and trips.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parents[1]
K = REPO / "experiments" / "K_recovery"
J = REPO / "experiments" / "J_validation_transfer"
N_ENC, SWITCH, RESTART = 24, 12, 10


def _seeds():
    spec = importlib.util.spec_from_file_location("l_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def setup(body_name: str):
    """(body adapter, type model cfg, time coefficients) for a body; IMU + rendered vision where a vision model
    exists (NERVA, wheeled), else IMU + the abstract camera (humanoid)."""
    from apm_transfer.nerva_world import Camera
    from apm_transfer.runtime import HumanoidBody, NervaBody, WheeledBody
    from apm_transfer.vision import VisionCamera
    if body_name == "nerva":
        edges = json.loads((J / "configs" / "J4_edges.json").read_text())["acc8"]
        channels = json.loads((J / "configs" / "H_admitted_interface.json").read_text())["interface"]
        model = json.loads((HERE / "configs" / "LC_nerva_vision_types.json").read_text())
        time_coef = json.loads((J / "configs" / "J6_model.json").read_text())["time_coef"]
        return NervaBody(edges, channels, VisionCamera()), model, time_coef, "IMU + rendered vision"
    if body_name == "wheeled":
        edges = json.loads((K / "configs" / "K_wheeled_edges.json").read_text())["acc8"]
        model = json.loads((HERE / "configs" / "LC_wheeled_vision_types.json").read_text())
        time_coef = json.loads((K / "configs" / "K_wheeled_time.json").read_text())["time_coef"]
        return WheeledBody(edges, VisionCamera()), model, time_coef, "IMU + rendered vision"
    edges = json.loads((K / "configs" / "K_humanoid_edges.json").read_text())["acc8"]
    model = json.loads((K / "configs" / "K_humanoid_types.json").read_text())
    time_coef = json.loads((K / "configs" / "K_humanoid_time.json").read_text())["time_coef"]
    return HumanoidBody(edges, Camera()), model, time_coef, "IMU + abstract camera (no humanoid vision model)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", choices=("nerva", "wheeled", "humanoid"), required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from apm.protocol import codec
    from apm.protocol.messages import EntityObservation, Header
    from apm_transfer.infoseek import Task
    from apm_transfer.runtime import Runtime, replay

    body, model, time_coef, sensing = setup(a.body)
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())
    task = Task("handoff", **{k: v for k, v in tasks["tasks"]["handoff"].items() if k != "why"})
    base = int(_seeds().world_seed("LF", "D", a.index).generate_state(1)[0])
    rng = np.random.default_rng(base)
    ents = [("A1", "A"), ("B1", "B"), ("C1", "C")]
    order = [ents[int(i)] for i in rng.choice(3, N_ENC, p=(0.25, 0.5, 0.25))]
    enc_seeds = rng.integers(0, 2**31, N_ENC)
    tmp = Path(tempfile.mkdtemp())

    def make(log: Path | None = None) -> Runtime:
        return Runtime(model, time_coef, task, tasks["voi"], seed=base, log=log)
    r1, r2 = make(tmp / "r1.jsonl"), make(tmp / "r2.jsonl")
    state = tmp / "r2_state.json"
    rows, msgs_ok, msgs_n, restart_equal = [], 0, 0, True
    t0 = time.time()
    for k, (ident, kind) in enumerate(order):
        etype = ("pushy" if k < SWITCH else "calm") if ident == "B1" else "calm"
        obs_msg = EntityObservation(header=Header(source=a.body, stream="entities", seq=k + 1, stamp_s=float(k)),
                                    track_id=f"track-{k}", identity=ident, identity_confidence=0.95,
                                    entity_type=kind)
        for rt in (r1, r2):
            rt.ingest(obs_msg)
        if k == RESTART:  # R2: save, delete, rebuild from the file alone
            r2.save(state)
            del r2
            r2 = make(tmp / "r2.jsonl")
            r2.load(state)
        i1, k1 = r1.identity_of(f"track-{k}")
        d1 = r1.decide(i1, k1)
        d2 = r2.decide(*r2.identity_of(f"track-{k}"))
        restart_equal &= asdict(d1) == asdict(d2)
        for m in r1.affectlink(d1, float(k), k + 1, source="apm-l"):
            msgs_n += 1
            msgs_ok += int(codec.decode(codec.encode(m)) == m)
        obs, truth = body.encounter(int(enc_seeds[k]), etype, d1)
        r1.observe(obs)
        r2.observe(obs)
        rows.append({"k": k, "identity": ident, "true_type": etype, "action": d1.action, "reason": d1.reason,
                     "windows": d1.windows, "standoff": d1.standoff, "proximity": d1.proximity,
                     "p_risk": round(d1.p_risk, 4), "predicted_risk": round(d1.predicted_risk, 4),
                     "d_stop": obs.report["d_stop"], "imu": list(obs.imu), "camera": list(obs.camera),
                     "contacts": truth["contacts"], "fell": obs.report["fell"], "trips": obs.report["trips"]})
        print(f"{k:2} {ident} {etype:5} action {d1.action:3} {d1.reason:28} p {d1.p_risk:.3f} "
              f"d {obs.report['d_stop']:.2f} contacts {truth['contacts']}", flush=True)
    rep = replay(tmp / "r1.jsonl", make)
    intents_bounded = all(0 <= r["proximity"] <= 1 and r["windows"] in (1, 2, 3) for r in rows)
    res = {"body": a.body, "sensing": sensing, "encounters": N_ENC, "wall_s": round(time.time() - t0, 1),
           "F1_runs_on_body": True, "F2_restart_identical": restart_equal, "F3_replay": rep,
           "F4_affectlink": {"messages": msgs_n, "valid_roundtrip": msgs_ok, "ok": msgs_ok == msgs_n},
           "F5_cap": {"max_predicted_risk": max(r["predicted_risk"] for r in rows), "cap": task.risk_cap,
                      "ok": all(r["predicted_risk"] <= task.risk_cap + 1e-9 for r in rows),
                      "supervisor_vetoes": r1.safety.vetoes},
           "F6_intents_bounded": intents_bounded,
           "falls": sum(r["fell"] for r in rows), "trips": sum(r["trips"] for r in rows),
           "B1_stops_before_after": [[r["d_stop"] for r in rows if r["identity"] == "B1" and r["k"] < SWITCH],
                                     [r["d_stop"] for r in rows if r["identity"] == "B1" and r["k"] >= SWITCH]],
           "rows": rows}
    res["all_checks_pass"] = bool(restart_equal and rep["identical"] and res["F4_affectlink"]["ok"]
                                  and res["F5_cap"]["ok"] and intents_bounded)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows",)}, indent=1))


if __name__ == "__main__":
    main()
