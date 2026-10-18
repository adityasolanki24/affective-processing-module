"""M1: the end-to-end integration check of the APM runtime on a physics body (MuJoCo wheeled robot).

    python experiments/M_release/scripts/m1_integration.py [--camera vision|abstract] [--n 12]

It exits non-zero if any check fails. Each check is printed with its evidence:
1. **init:** the runtime builds from the frozen type, time and task models;
2. **sensors:** each encounter yields an `Observation` from the body's IMU (and camera);
3. **memory:** beliefs change with observations;
4. **decisions:** each decision is a grid intent, OBSERVE or HOLD;
5. **bounded commands:** intents within channel ranges, windows in {1, 2, 3}, the predicted risk within the cap;
6. **safety:**
   - (a) the risk supervisor vetoes a forced over-cap approach, and the body executes a HOLD;
   - (b) the body's physical supervisor trips on a hard shove, independently of APM;
7. **motion:** the body arrives, and its stopping distance matches the commanded one;
8. **persist/restart:** a runtime rebuilt from its saved file decides identically;
9. **replay:** the log reproduces every decision;
10. **AffectLink:** every output message validates and survives the codec round trip.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
K = REPO / "experiments" / "K_recovery"
L = REPO / "experiments" / "L_audit_perception"


def build(camera: str):
    from apm_transfer.infoseek import Task
    from apm_transfer.nerva_world import Camera
    from apm_transfer.runtime import WheeledBody
    edges = json.loads((K / "configs" / "K_wheeled_edges.json").read_text())["acc8"]
    if camera == "vision":
        from apm_transfer.vision import VisionCamera
        model = json.loads((L / "configs" / "LC_wheeled_vision_types.json").read_text())
        cam = VisionCamera()
    else:
        model = json.loads((K / "configs" / "K_wheeled_types.json").read_text())
        cam = Camera()
    time_coef = json.loads((K / "configs" / "K_wheeled_time.json").read_text())["time_coef"]
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())
    task = Task("handoff", **{k: v for k, v in tasks["tasks"]["handoff"].items() if k != "why"})
    return WheeledBody(edges, cam), model, time_coef, task, tasks["voi"]


def check(name: str, ok: bool, evidence) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}", flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", choices=("vision", "abstract"), default="vision")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args(argv)
    from apm.protocol import codec
    from apm_express.intent import RANGES
    from apm_transfer import infoseek
    from apm_transfer.infoseek import HOLD
    from apm_transfer.runtime import Runtime, replay
    from apm_transfer.wheeled_world import WheeledRobot, closed_loop_encounter

    body, model, time_coef, task, voi = build(a.camera)
    tmp = Path(tempfile.mkdtemp())

    def make(log=None, seed=7):
        return Runtime(model, time_coef, task, voi, seed=seed, log=log)
    ok = True
    rt = make(tmp / "r.jsonl")
    ok &= check("1 init", rt.model.E.shape[0] >= 2, f"K={rt.model.E.shape[0]} types, model {rt.model_hash}")
    rng = np.random.default_rng(3)
    seq = [("B1", "B", "pushy")] * (a.n // 2) + [("B1", "B", "calm")] * (a.n - a.n // 2)
    beliefs, obs_n, cam_n, decisions, msgs_ok, msgs = [], 0, 0, [], 0, 0
    for k, (ident, kind, etype) in enumerate(seq):
        d = rt.decide(ident, kind)
        decisions.append(d)
        for m in rt.affectlink(d, float(k), k + 1):
            msgs += 1
            msgs_ok += int(codec.decode(codec.encode(m)) == m)
        obs, truth = body.encounter(int(rng.integers(2**31)), etype, d)
        obs_n += int(len(obs.imu) > 0 or d.standoff > 0)
        cam_n += int(len(obs.camera) > 0)
        rt.observe(obs)
        beliefs.append(rt.memory.belief(ident, kind).tolist())
        if k == 0:
            arrived, d_stop, d_cmd = obs.report["arrived"], obs.report["d_stop"], 0.6 - 0.4 * d.proximity + d.standoff
    ok &= check("2 sensors", obs_n == len(seq) and cam_n == len(seq), f"IMU-or-standoff {obs_n}/{len(seq)}, "
                f"camera codes {cam_n}/{len(seq)} ({a.camera})")
    ok &= check("3 memory", np.ptp(np.array(beliefs)[:, 1]) > 0.1,
                f"P(type 2) ranged {np.min(np.array(beliefs)[:, 1]):.3f}..{np.max(np.array(beliefs)[:, 1]):.3f}")
    ok &= check("4 decisions", all(d.action >= -2 for d in decisions),
                f"actions {[d.action for d in decisions]}")
    bounded = all(0 <= d.proximity <= 1 and 0 <= d.hesitation <= 1 and d.windows in (1, 2, 3)
                  and d.predicted_risk <= task.risk_cap + 1e-12 for d in decisions)
    it_ok = all(RANGES[c][0] <= getattr(d.intent(), c) <= RANGES[c][1] for d in decisions for c in RANGES)
    ok &= check("5 bounded commands", bounded and it_ok,
                f"max predicted risk {max(d.predicted_risk for d in decisions):.3f}")
    # 6a: a faulty decision layer proposes the closest, fastest approach while the belief is pushy
    sup = make(seed=11)
    for _ in range(6):
        sup.observe(type(obs)("B9", "B", 0, (7, 7), (6,), True))
    real = infoseek.choose
    infoseek.choose = lambda *x, **kw: (19, 1, {"cost": 0.0})  # fault injection
    import apm_transfer.runtime as R
    R.choose = infoseek.choose
    try:
        dv = sup.decide("B9", "B")
    finally:
        infoseek.choose = R.choose = real
    o_hold, _ = body.encounter(123, "pushy", dv)
    ok &= check("6a risk supervisor veto", dv.vetoed and dv.action == HOLD and o_hold.report["d_stop"] > 0.8,
                f"vetoed={dv.vetoed}, executed {dv.reason}, stopped {o_hold.report['d_stop']:.2f} m")

    # 6b: the physical supervisor trips on a hard shove (1.0 m/s, 10x the entity cap), with no APM involved
    class Shoved(WheeledRobot):
        def advance(self):
            if abs(self.t - 3.0) < 1e-9 + self.dt / 2:
                self.push(-1.0, 0.0)
            return super().advance()
    _, o, _ = closed_loop_encounter(5, "calm", decisions[0].intent(), None, windows=1, robot_factory=Shoved)
    ok &= check("6b physical safety trip", o["trips"] > 0 and not o["fell"], f"trips {o['trips']}, fell {o['fell']}")
    ok &= check("7 motion", arrived and abs(d_stop - d_cmd) < 0.08,
                f"arrived={arrived}, stopped {d_stop:.3f} m vs commanded {d_cmd:.3f} m")
    # 8: restart — save, rebuild from the file, compare the next decisions with an uninterrupted twin
    rt.save(tmp / "state.json")
    twin = make()
    twin.load(tmp / "state.json")
    same = all(asdict(rt.decide("B1", "B")) == asdict(twin.decide("B1", "B")) for _ in range(3))
    ok &= check("8 persist/restart", same, "3 post-restart decisions identical")
    rep = replay(tmp / "r.jsonl", make)
    ok &= check("9 replay", rep["identical"], rep)
    ok &= check("10 AffectLink", msgs_ok == msgs and msgs > 0, f"{msgs_ok}/{msgs} messages valid and round-trip")
    print("INTEGRATION", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
