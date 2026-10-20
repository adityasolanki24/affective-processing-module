"""Programme J wheeled runner: programme I's systems and families, extended passive logging, validity controls and
(J2) stress conditions.

Systems: every spec `i_common.build` accepts, plus
- `{"kind": "decay_only", "type_model": ..., "lam": λ, "freeze_after_s": T}`: `apm_transfer.controls.DecayOnly`;
- `{"kind": "revising_x", "type_model": ..., "lam": λ}`: decision A's rule with `RevisionMemoryX` (conditioned or
  unconditional codings);
- `{"kind": "observer", "driver": <spec>, "type_model": ..., "lam": λ}`: the driver acts; a `RevisionMemoryX` learns
  passively;
- `"stress"`-dependent wrappers (J2), applied by `run_job` from `job["condition"]`.

Encounter rows keep programme I's first 11 columns, so `i_metrics` works unchanged. They then add:
near frames, careful-action frames, action band, context band, and mean robot speed over observable frames.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "I_autonomous_revision" / "scripts"))

import i_common as I  # noqa: E402
from i_common import A0, OracleMemory, init, type_model  # noqa: E402, F401  (init re-exported for Pool)

from apm_research.harness import run_lifetime  # noqa: E402
from apm_revise.evidence import ENTITY_TTL_S, NEAR_M  # noqa: E402
from apm_revise.scenarios import SensorGaps, dispositions, make_lifetime  # noqa: E402
from apm_select.instrument import Instrumented, collision_split  # noqa: E402
from apm_select.regulators import make_regulator  # noqa: E402
from apm_transfer.controls import DecayOnly  # noqa: E402
from apm_transfer.encounter import EncounterX  # noqa: E402


def _seeds():
    spec = importlib.util.spec_from_file_location("j_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SEEDS = _seeds()


def build(spec: dict):
    if spec["kind"] == "b0":
        from g_common import make_policy
        return make_policy(spec)
    if spec["kind"] == "revising_x":
        from g_common import _STATE

        from apm_transfer.conditioned import RevisingX
        return RevisingX.make(_STATE["predictor"], A0["lam_c"] * spec.get("lam_c_scale", 1.0), A0["lam_s"],
                              make_regulator(A0["regulator"], A0["params"]), type_model(spec["type_model"]),
                              spec["lam"], name=spec.get("label", "IX"))
    if spec["kind"] == "observer":
        from apm_transfer.conditioned import Observer, RevisionMemoryX
        m = type_model(spec["type_model"])
        return Observer(build(spec["driver"]), lambda: RevisionMemoryX(m, spec["lam"]))
    if spec["kind"] == "decay_only":
        from g_common import _STATE
        return DecayOnly(_STATE["predictor"], A0["lam_c"], A0["lam_s"], make_regulator(A0["regulator"], A0["params"]),
                         type_model(spec["type_model"]), spec["lam"], None, name=spec.get("label", "decay"),
                         freeze_after_s=spec["freeze_after_s"])
    return I.build(spec)


class Tracker(I.Tracker):
    """Programme I's tracker with the extended encounter record (`apm_transfer.encounter.EncounterX`)."""

    def observe(self, obs, t: float) -> None:
        self.policy.observe(obs, t)
        people = [e for e in obs.entities if e.kind == "person"]
        near = sum(e.distance < NEAR_M for e in people)
        for e in people:
            enc = self.open.setdefault(e.track, EncounterX(e.track, start=t))
            if e.identity is not None and e.identity_confidence >= enc.confidence:
                enc.identity, enc.confidence = e.identity, e.identity_confidence
            enc.observe_x(e.distance, obs.v, obs.action, e.bearing, near - (e.distance < NEAR_M) > 0)
            self.last[e.track] = t
        for ev in obs.events:
            if ev[0] == "collision" and ev[1] in self.open:
                self.open[ev[1]].collision = True
        for track in [k for k, s in self.last.items() if t - s > ENTITY_TTL_S]:
            self._close(track)

    def _close(self, track: str) -> None:
        enc = self.open[track]
        super()._close(track)
        assert isinstance(enc, EncounterX)
        self.encounters[-1] += [enc.near_frames, enc.careful_frames, enc.action_band(), enc.context_band(),
                                round(enc.speed_sum / enc.frames, 4) if enc.frames else None]


def make(job: dict):
    """(lifetime, roles, policy-wrapper) for a job; J2 stress conditions hook in here (`apm_transfer.stress`)."""
    ss = SEEDS.world_seed(job["experiment"], job["partition"], job["index"])
    cond = job.get("condition", "nominal")
    if cond == "nominal":
        lt, roles = make_lifetime(job["scenario"], ss, job.get("episodes", 30))
        return lt, roles, (lambda p: p), None
    from apm_transfer.stress import make_stress
    return make_stress(cond, job["scenario"], ss, job.get("episodes", 30))


def run_job(job: dict) -> dict:
    spec, sc = job["spec"], job["scenario"]
    lt, roles, wrap, ctx = make(job)
    inner = build(spec)
    policy = Instrumented(wrap(SensorGaps(inner) if sc == "S9" else inner))
    tracker = Tracker(policy, lt, getattr(inner, "inner", inner))
    if ctx is not None:
        with ctx():
            r = run_lifetime(lt, tracker)
    else:
        r = run_lifetime(lt, tracker)
    mem = inner.memory
    return {"label": spec.get("label"), "scenario": sc, "condition": job.get("condition", "nominal"), "roles": roles,
            "index": job["index"], "partition": job["partition"], "experiment": job["experiment"],
            "totals": {k: float(r[k]) for k in I.RESULT_KEYS} | {k: float(v) for k, v in policy.stats.items()}
            | collision_split(r),
            "actions": r["actions"], "memory_counts": dict(mem.counts), "probes": getattr(inner, "probes", 0),
            "keepaways": getattr(inner, "keepaways", 0), "snapshots": tracker.snapshots, "truth": tracker.truth,
            "pred_logp": [x[7] for x in getattr(mem, "log", []) if len(x) > 7 and x[7] is not None],
            "p_log": [[x[0], x[1], x[5], x[3]] for x in getattr(mem, "log", [])],
            "logp_log": [[x[0], x[1], x[7]] for x in getattr(mem, "log", []) if len(x) > 7 and x[7] is not None],
            "encounters": tracker.encounters}


__all__ = ["run_job", "init", "dispositions", "OracleMemory"]
