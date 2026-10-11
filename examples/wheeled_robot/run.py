"""The same APM core on a second, materially different embodiment: a simulated differential-drive robot.

One continuous 10-minute life in the wheeled testbed (several arena episodes in sequence, one persistent APM), then a
summary of what the robot's memory learned about each pedestrian identity and its affect/tendency trajectory.

    python examples/wheeled_robot/run.py [--minutes 10] [--log runs/wheeled.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.wheeled import WheeledAdapter
from apm.protocol import to_dict
from apm.protocol.messages import OutcomeSignal
from apm_testbeds.wheeled import DT, WheeledWorld


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--log", type=Path, default=Path("runs/wheeled.jsonl"))
    args = ap.parse_args()
    emitter = Emitter("wheeled-demo")
    adapter = WheeledAdapter(emitter)
    apm = APM(profile=adapter.profile())
    counts: collections.Counter = collections.Counter()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    offset, world_seed, rows = 0.0, 0, []
    with args.log.open("w", encoding="utf-8") as log:
        while offset < args.minutes * 60:
            world = WheeledWorld(20_000 + world_seed)  # a new arena layout; pedestrians reappear by identity
            for k in range(int(60 / DT)):
                tel = world.step()
                tel.t += offset
                for m in adapter.step(tel):
                    if apm.ingest(m) is None and isinstance(m, OutcomeSignal):
                        counts[m.kind] += 1
                if k % 4 == 3:
                    out = apm.step(tel.t)
                    if k % 200 == 199:
                        log.write(json.dumps({"t": tel.t, "outputs": [to_dict(x) for x in out.messages()]}) + "\n")
                        avoid = max((x.avoid for x in out.tendencies if x.target != "self"), default=0.0)
                        rows.append((tel.t, out.affect.valence, out.affect.arousal, out.affect.dominance, avoid))
            offset += 60.0
            world_seed += 1
            adapter = WheeledAdapter(emitter)  # same robot, new arena: detectors re-arm
    print("outcomes:", dict(counts))
    print(" t(s)    V      A      D   max avoid")
    for t, v, a, d, avoid in rows[::3]:
        print(f"{t:6.0f} {v:+.2f}  {a:+.2f}  {d:+.2f}   {avoid:.2f}")
    print("learned per-encounter collision probability by pedestrian identity (memory):")
    for key, record in sorted(apm.entities.records.items()):
        if key.startswith("id:"):
            p, confidence = apm.entities.p("collision", key[3:], "person")
            print(f"  {key:10s} encounters {record.encounters:5.1f}  p(collision) {p:.2f}  confidence {confidence:.2f}")
    print(f"log: {args.log}")


if __name__ == "__main__":
    main()
