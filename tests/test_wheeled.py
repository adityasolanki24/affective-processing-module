"""Second embodiment: the wheeled testbed and its adapter run on the unmodified core."""
import numpy as np

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.wheeled import WheeledAdapter
from apm.protocol import validate
from apm.protocol.messages import EntityObservation, OutcomeHypothesis, OutcomeSignal, SelfState, Status
from apm_testbeds.wheeled import WheeledWorld


def run(seed=3, steps=1200):
    world, adapter = WheeledWorld(seed), WheeledAdapter(Emitter("w"))
    apm = APM(profile=adapter.profile())
    msgs, rejected = [], 0
    for k in range(steps):
        tel = world.step()
        for m in adapter.step(tel):
            validate(m)
            rejected += apm.ingest(m) is not None
            msgs.append(m)
        if k % 2:
            apm.step(tel.t)
    return apm, msgs, rejected


def test_deterministic_world():
    a, b = WheeledWorld(5), WheeledWorld(5)
    for _ in range(100):
        ta, tb = a.step(), b.step()
    assert (ta.x, ta.y, ta.energy) == (tb.x, tb.y, tb.energy)


def test_wheeled_messages_are_its_own_and_accepted_by_the_same_core():
    apm, msgs, rejected = run()
    assert rejected == 0
    s = next(m for m in msgs if isinstance(m, SelfState))
    assert s.thermal_margin.status is Status.UNSUPPORTED and s.thermal_margin.value is None
    assert apm.profile.embodiment_class == "wheeled" and "collision" in apm.profile.outcome_kinds
    assert any(isinstance(m, EntityObservation) and m.identity for m in msgs)
    assert any(isinstance(m, OutcomeHypothesis) and m.kind == "collision" for m in msgs)
    kinds = {m.kind for m in msgs if isinstance(m, OutcomeSignal)}
    assert {"task_completed"} <= kinds


def test_collisions_teach_entity_memory_on_the_wheeled_body():
    totals = {}
    for seed in range(8):
        apm, msgs, _ = run(seed, 1200)
        for key, record in apm.entities.records.items():
            totals.setdefault(key, []).append(record.counts.get("collision", 0.0) / max(1e-9, record.encounters))
    assert "type:person" in totals or "type:obstacle" in totals
    assert all(np.isfinite(v).all() for v in map(np.asarray, totals.values()))
