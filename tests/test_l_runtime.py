"""Programme L Phase F: the integrated runtime (no simulator needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apm_transfer.infoseek import HOLD, Task
from apm_transfer.runtime import Observation, Runtime, SafetySupervisor, replay

REPO = Path(__file__).resolve().parents[1]
MODEL = json.loads((REPO / "experiments" / "K_recovery" / "configs" / "K_types.json").read_text())
TIME = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" / "J6_model.json").read_text())
TASKS = json.loads((REPO / "experiments" / "K_recovery" / "configs" / "K_tasks.json").read_text())
TASK = Task("handoff", **{k: v for k, v in TASKS["tasks"]["handoff"].items() if k != "why"})


def _rt(log=None, seed=3):
    return Runtime(MODEL, TIME["time_coef"], TASK, TASKS["voi"], seed=seed, log=log)


def _obs(k: int, ident: str) -> Observation:
    pushy = ident == "B1" and k < 8
    return Observation(ident, ident[0], band=1, imu=(7,) if pushy else (1,), camera=(5,) if pushy else (0,),
                       detected=pushy)


def _lifetime(rt: Runtime, n: int = 16, restart_at: int | None = None, state: Path | None = None) -> list:
    out = []
    for k in range(n):
        ident = ("A1", "B1", "C1", "B1")[k % 4]
        if restart_at is not None and k == restart_at:
            assert state is not None
            rt.save(state)
            fresh = _rt(log=rt.log_path)
            fresh.load(state)
            rt = fresh
        d = rt.decide(ident, ident[0])
        out.append(d)
        rt.observe(_obs(k, ident))
    return out


def test_restart_and_replay_are_exact(tmp_path):
    ref = _lifetime(_rt(log=tmp_path / "a.jsonl"))
    again = _lifetime(_rt(log=tmp_path / "b.jsonl"), restart_at=7, state=tmp_path / "state.json")
    assert ref == again
    assert replay(tmp_path / "b.jsonl", _rt)["identical"]


def test_supervisor_vetoes_any_approach_above_the_cap():
    sup = SafetySupervisor(np.array(MODEL["model"]["E"]), TIME["time_coef"], TASK)
    b = np.array([0.0, 1.0])  # certainly pushy
    risky = [i for i in range(20) if sup.risk(i, 1, b) > TASK.risk_cap]
    assert risky
    for i in risky:
        assert sup.check(i, 1, b)[:2] == (HOLD, 1)


def test_decisions_are_bounded_and_affectlink_valid():
    from apm.protocol import codec
    from apm.protocol.messages import EntityObservation, Header
    rt = _rt()
    rt.ingest(EntityObservation(header=Header(source="robot", stream="entities", seq=1, stamp_s=0.0),
                                track_id="t7", identity="B1", identity_confidence=0.9, entity_type="B"))
    ident, kind = rt.identity_of("t7")
    for k in range(10):
        d = rt.decide(ident, kind)
        assert 0 <= d.proximity <= 1 and 0 <= d.hesitation <= 1 and d.windows in (1, 2, 3)
        assert d.predicted_risk <= TASK.risk_cap + 1e-12
        for m in rt.affectlink(d, float(k), k + 1):
            assert codec.decode(codec.encode(m)) == m
        rt.observe(_obs(k, ident))
    rt.ingest(EntityObservation(header=Header(source="robot", stream="entities", seq=2, stamp_s=1.0),
                                track_id="t8", identity="B1", identity_confidence=0.2, entity_type="B"))
    assert rt.identity_of("t8")[0] == "anon:t8"  # low identity confidence: no persistent identity
