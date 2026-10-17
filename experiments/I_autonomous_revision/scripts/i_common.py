"""Programme I runner: systems, lifetimes and per-lifetime records on G's wheeled testbed and scenario families.

System specs (JSON):
- `{"kind": "regulated", ...}`: any G system, built by G's `make_policy` (e.g. A0 with F's cumulative memory, or
  `A0_mem` = G2b's revisable memory + uncertainty probing).
- `{"kind": "revising", "type_model": <path>, "lam": λ, "explore": {...}, "lam_c_scale": s}`: decision A's rule
  (A0's tuned regulator and weights) with `apm_revise.RevisionMemory`.
- `{"kind": "oracle"}`: evaluation-only reference. Memory answers with the true disposition's rate in force (G2's
  Oracle), on A0's rule.

`lam_c_scale` (default 1) multiplies λc for any system: the *cautious robot* variant used to create real censoring.

Every lifetime also runs a passive evidence logger, identical for every system. It records each person encounter
(robot-observable fields, both outcome codings) and attaches the true disposition for evaluation only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
G = REPO / "experiments" / "G_architecture_selection"
sys.path.insert(0, str(G / "scripts"))

from g_common import RESULT_KEYS, init, make_policy  # noqa: E402, F401  (init re-exported for Pool initialisers)

from apm_research.harness import run_lifetime  # noqa: E402
from apm_revise.belief import TypeModel  # noqa: E402
from apm_revise.evidence import ENTITY_TTL_S, Encounter  # noqa: E402
from apm_revise.policy import KeepAway, Revising  # noqa: E402
from apm_revise.scenarios import EPISODES, SensorGaps, dispositions, make_lifetime  # noqa: E402
from apm_select.instrument import Instrumented, collision_split  # noqa: E402
from apm_select.memory import BeliefMemory  # noqa: E402
from apm_select.regulators import make_regulator  # noqa: E402

ORACLE_RATES = json.loads((REPO / "experiments" / "F_behavioral_learning" / "configs" / "oracle_rates.json")
                          .read_text())["rates_by_disposition"]
A0 = json.loads((G / "configs" / "G6_systems.json").read_text())["rules"]["A0"]
NAMES = [f"ped-{i}" for i in range(8)]


def _seeds():
    spec = importlib.util.spec_from_file_location("i_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SEEDS = _seeds()
_MODELS: dict[str, TypeModel] = {}


def type_model(path: str) -> TypeModel:
    if path not in _MODELS:
        _MODELS[path] = TypeModel.from_dict(json.loads((REPO / path).read_text()))
    return _MODELS[path]


class OracleMemory(BeliefMemory):
    """Evaluation reference: p = converged rate of the identity's true disposition in the current episode."""

    def set_truth(self, truth: dict[str, str]) -> None:
        self.truth = truth

    def p_collision(self, identity, kind):
        if kind == "person" and identity in getattr(self, "truth", {}):
            return ORACLE_RATES[self.truth[identity]], 1.0
        return super().p_collision(identity, kind)


def build(spec: dict):
    """The policy for a spec; `keep_away` wraps it in `apm_revise.policy.KeepAway`."""
    ka = spec.get("keep_away")
    inner = _build(spec)
    if not ka:
        return inner
    return KeepAway(inner, ka.get("keep_m", 2.0), ka.get("kappa", 1.5), ka.get("explore"),
                    ka.get("approach_only", True))


def _build(spec: dict):
    scale = spec.get("lam_c_scale", 1.0)
    if spec["kind"] == "revising":
        from g_common import _STATE
        return Revising(_STATE["predictor"], A0["lam_c"] * scale, A0["lam_s"],
                        make_regulator(A0["regulator"], A0["params"]), type_model(spec["type_model"]),
                        spec.get("lam", 0.0), spec.get("explore"), name=spec.get("label", "I"))
    if spec["kind"] == "oracle":
        pol = make_policy({**A0, "lam_c": A0["lam_c"] * scale, "memory": {"model": "M0", "evidence": "all"}})
        base_begin = pol.begin_lifetime

        def begin_with_oracle():
            base_begin()
            pol.memory = OracleMemory("M0", {}, "all")
        pol.begin_lifetime = begin_with_oracle
        return pol
    spec = {k: v for k, v in spec.items() if k not in ("label", "lam_c_scale", "keep_away")}
    return make_policy({**spec, "lam_c": spec["lam_c"] * scale})


class Tracker:
    """Outer wrapper: per-episode truth (Oracle only, and evaluation), end-of-episode memory snapshots, and a
    passive evidence log identical for every system."""

    def __init__(self, policy, lifetime, inner):
        self.policy, self.lt, self.inner = policy, lifetime, inner
        self.snapshots: list[dict] = []
        self.truth: list[dict] = []
        self.episode = 0
        self.open: dict[str, Encounter] = {}
        self.last: dict[str, float] = {}
        self.encounters: list[list] = []

    def __getattr__(self, item):
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.truth.append(dispositions(self.lt, self.episode))
        mem = getattr(self.inner, "memory", None)
        if isinstance(mem, OracleMemory):
            mem.set_truth(self.truth[-1])
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.policy.observe(obs, t)
        for e in obs.entities:
            if e.kind != "person":
                continue
            enc = self.open.setdefault(e.track, Encounter(e.track, start=t))
            if e.identity is not None and e.identity_confidence >= enc.confidence:
                enc.identity, enc.confidence = e.identity, e.identity_confidence
            enc.observe(e.distance, obs.v, obs.action)
            self.last[e.track] = t
        for ev in obs.events:
            if ev[0] == "collision" and ev[1] in self.open:
                self.open[ev[1]].collision = True
        for track in [k for k, s in self.last.items() if t - s > ENTITY_TTL_S]:
            self._close(track)

    def _close(self, track: str) -> None:
        del self.last[track]
        enc = self.open.pop(track)
        truth = self.truth[-1].get(enc.identity) if enc.identity else None
        gap = None if enc.min_gap == float("inf") else round(enc.min_gap, 4)
        self.encounters.append([self.episode, round(enc.start, 2), enc.identity, round(enc.confidence, 3),
                                enc.collision, enc.frames, gap, enc.g_class(), enc.category("dense"),
                                enc.category("sparse"), truth])

    def decide(self, t: float) -> int:
        return self.policy.decide(t)

    def end_episode(self, t: float) -> None:
        for track in list(self.open):
            self._close(track)
        self.policy.end_episode(t)
        mem = self.inner.memory
        read = getattr(mem, "posterior_p", None)
        self.snapshots.append({n: [round(v, 5) for v in (read(n) if read else mem.p_collision(n, "person"))]
                               for n in NAMES if n in self.truth[-1]})
        self.episode += 1


def run_job(job: dict) -> dict:
    spec, sc = job["spec"], job["scenario"]
    lt, roles = make_lifetime(sc, SEEDS.world_seed(job["experiment"], job["partition"], job["index"]),
                              job.get("episodes", EPISODES))
    inner = build(spec)
    policy = Instrumented(SensorGaps(inner) if sc == "S9" else inner)
    tracker = Tracker(policy, lt, getattr(inner, "inner", inner))
    r = run_lifetime(lt, tracker)
    voi = getattr(inner, "voi_log", [])
    return {"label": spec.get("label"), "scenario": sc, "roles": roles, "index": job["index"],
            "partition": job["partition"], "experiment": job["experiment"],
            "totals": {k: float(r[k]) for k in RESULT_KEYS} | {k: float(v) for k, v in policy.stats.items()}
            | collision_split(r),
            "actions": r["actions"], "memory_counts": dict(inner.memory.counts),
            "state_size": inner.memory.state_size(), "probes": getattr(inner, "probes", 0),
            "voi_eligible": len(voi), "keepaways": getattr(inner, "keepaways", 0),
            "snapshots": tracker.snapshots, "truth": tracker.truth,
            "encounters": tracker.encounters}
