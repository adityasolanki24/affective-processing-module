"""Standalone APM demonstration with a scripted, clearly SYNTHETIC robot (no robot, simulator or NERVA needed).

Story (seconds): 0–10 calm walk toward a goal; 10–14 an unfamiliar person ("track p1", identity "visitor-a")
approaches fast and the robot's own predictor raises a collision hypothesis; 13.5 a collision is observed;
20–40 calm; 40–50 the same identity returns (new track p2) and approaches slowly; 50 goal completed.
On the second visit memory expects adverse outcomes from visitor-a before anything happens.

    python examples/standalone/scenario.py [--log runs/standalone.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from apm import APM
from apm.adapters.base import Emitter
from apm.protocol import to_dict
from apm.protocol.messages import (
    EmbodimentProfile,
    EntityObservation,
    GoalState,
    OutcomeHypothesis,
    OutcomeSignal,
    Quantity,
    SelfState,
)

HZ = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("runs/standalone.jsonl"))
    args = ap.parse_args()
    e = Emitter("synthetic-bot")
    profile = EmbodimentProfile(header=e.header("profile", 0.0), robot_id="synthetic-bot", embodiment_class="synthetic",
                                self_fields=("stability_margin", "energy_margin"),
                                outcome_kinds=("collision", "benign_contact", "task_completed", "task_interrupted"),
                                goal_types=("reach",))
    apm = APM(profile=profile)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.log.open("w", encoding="utf-8") as log:
        for i in range(int(60 * HZ)):
            t = (i + 1) / HZ
            apm.ingest(SelfState(header=e.header("self_state", t),
                                 stability_margin=Quantity.measured(0.85 if not 13.5 <= t < 15 else 0.4, "fraction"),
                                 energy_margin=Quantity.measured(max(0.2, 0.9 - t / 200), "fraction"), mode="walk"))
            apm.ingest(GoalState(header=e.header("goal", t), goal_id="reach-dock", goal_type="reach", priority=0.7,
                                 progress=Quantity.measured(min(1.0, t / 50), "fraction"), active=t < 50))
            if 10 <= t < 14:
                d = 3.0 - (t - 10) * 0.7
                apm.ingest(EntityObservation(header=e.header("entities", t), track_id="p1", identity="visitor-a",
                                             identity_confidence=0.9, entity_type="person",
                                             distance=Quantity.measured(d, "m"),
                                             closing_speed=Quantity.measured(0.7, "m/s")))
                if t >= 11.5:
                    apm.ingest(OutcomeHypothesis(header=e.header("hypotheses", t), hypothesis_id="p1-collision",
                                                 kind="collision", target="p1", affected="self",
                                                 probability=min(0.9, (t - 11) / 3), horizon_s=1.5))
            if abs(t - 13.5) < 1e-9:
                apm.ingest(OutcomeSignal(header=e.header("outcomes", t), outcome_id="contact-1", kind="collision",
                                         target="p1", affected="self", magnitude=0.7))
            if 40 <= t < 50:
                apm.ingest(EntityObservation(header=e.header("entities", t), track_id="p2", identity="visitor-a",
                                             identity_confidence=0.9, entity_type="person",
                                             distance=Quantity.measured(3.0 - (t - 40) * 0.2, "m"),
                                             closing_speed=Quantity.measured(0.2, "m/s")))
            if abs(t - 50) < 1e-9:
                apm.ingest(OutcomeSignal(header=e.header("outcomes", t), outcome_id="goal-1", kind="task_completed",
                                         target=None, affected="reach-dock", magnitude=1.0))
            out = apm.step(t)
            log.write(json.dumps({"t": t, "outputs": [to_dict(m) for m in out.messages()]}) + "\n")
            if i % (2 * HZ) == 2 * HZ - 1 or abs(t - 13.6) < 1e-9 or abs(t - 40.5) < 1e-9:
                tend = {x.target: max(("avoid", x.avoid), ("approach", x.approach), ("orient", x.orient),
                                      key=lambda kv: kv[1]) for x in out.tendencies if x.target != "self"}
                exp = {x.target: round(x.p_adverse, 2) for x in out.expectations}
                rows.append((t, out.affect, tend, exp, out.diagnostics.health))
    print(" t(s)   V      A      D     health   strongest tendency per target   expected p_adverse")
    for t, a, tend, exp, health in rows:
        print(f"{t:5.1f} {a.valence:+.2f}  {a.arousal:+.2f}  {a.dominance:+.2f}  {health:8s} "
              f"{ {k: (n, round(v, 2)) for k, (n, v) in tend.items()} }  {exp}")
    print(f"log: {args.log} ({sum(1 for _ in args.log.open())} steps); episodes: {len(apm.episodes.episodes)}")


if __name__ == "__main__":
    main()
