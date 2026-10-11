"""One bounded live NERVA simulation with APM attached in OBSERVE-ONLY mode.

Runs NERVA's validated neutral locomotion candidate in its own simulator (Open Duck Mini v2, MuJoCo, backlash scene,
training observation noise), walking forward for 20 s with one horizontal push at t = 8 s. Every control step's
telemetry goes through the NERVA adapter into APM; APM's outputs are logged and never fed back: the policy, its
actions, rewards, checkpoints and NERVA's safety logic are untouched.

Requires a Python environment with NERVA's simulator dependencies (mujoco, onnxruntime, Open Duck Playground) and
the NERVA repository; APM is put on the path from this checkout:

    PYTHONPATH=<APM>/src python examples/nerva/live_rollout.py --nerva <NERVA> --policy <candidate.onnx> \
        [--push 1.0] [--out runs/nerva_live]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from pathlib import Path

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.nerva import NervaAdapter
from apm.protocol import to_dict
from apm.protocol.messages import OutcomeHypothesis, OutcomeSignal

DT, SECONDS, PUSH_S, COMMAND = 0.02, 20.0, 8.0, (0.074, 0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerva", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--push", type=float, default=1.0, help="push speed, m/s (NERVA's training uses 0.1–1.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/nerva_live"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.nerva.resolve()))
    from nerva.analysis.motor_eval import deployment_metadata
    from nerva.interfaces import BehaviourCommand
    from nerva.sim.open_duck import SCENE_BACKLASH, OpenDuckSim

    with contextlib.redirect_stdout(io.StringIO()):
        sim = OpenDuckSim(policy_path=args.policy, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=True,
                          init_joint_noise=0.02, seed=args.seed)
    sim.set_neutral_motor_contract(deployment_metadata(hashlib.sha256(args.policy.read_bytes()).hexdigest()))
    sim.set_behaviour(BehaviourCommand(*COMMAND))
    adapter = NervaAdapter(Emitter("nerva-open-duck-live"))
    apm = APM(profile=adapter.profile())
    args.out.mkdir(parents=True, exist_ok=True)
    events = []
    with (args.out / "outputs.jsonl").open("w", encoding="utf-8") as log:
        for k in range(int(SECONDS / DT)):
            if abs(k * DT - PUSH_S) < 1e-9:  # one push, sideways (world +y), like NERVA's training pushes
                sim.push(0.0, args.push)
                events.append({"t": k * DT, "event": f"push {args.push} m/s (+y)"})
            step = sim.run(DT)[-1]
            for m in adapter.step(step.t, step.base_quat, step.base_linvel, step.base_angvel, step.tau_sq, COMMAND):
                if apm.ingest(m) is None and isinstance(m, (OutcomeSignal, OutcomeHypothesis)) and \
                        (isinstance(m, OutcomeSignal) or m.probability >= 0.5):
                    events.append({"t": step.t, "event": m.kind if isinstance(m, OutcomeSignal)
                                   else f"predicted {m.kind} p={m.probability:.2f}"})
            if k % 5 == 4:
                out = apm.step(step.t)
                log.write(json.dumps({"t": step.t, "outputs": [to_dict(x) for x in out.messages()]}) + "\n")
                if k % 50 == 49 or 7.9 < step.t < 10.5 and k % 10 == 9:
                    me = next(x for x in out.tendencies if x.target == "self")
                    margin = apm.context.margin("stability_margin", step.t)
                    print(f"t={step.t:5.2f} tilt {45 * (1 - margin) if margin is not None else math.nan:5.1f}deg  "
                          f"V {out.affect.valence:+.2f} A {out.affect.arousal:+.2f} D {out.affect.dominance:+.2f}  "
                          f"self: avoid {me.avoid:.2f} disengage {me.disengage:.2f} seek {me.seek_assistance:.2f}  "
                          f"health {out.diagnostics.health}")
    (args.out / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    apm.save(args.out / "apm_state.json")
    print("events:", events)


if __name__ == "__main__":
    main()
