"""Programme L Phase F: the minimal integrated APM runtime.

One object (`Runtime`) runs the whole loop for any body:

    obs = body.encounter(seed, entity_type, runtime.decide(identity, kind))   # the body acts and senses
    runtime.observe(obs)                                                    # memory learns from what was sensed

**Parts:**
- `Observation`: a robot-independent record of what one encounter showed. Adapters code the body's own sensors
  into it: the IMU category per sensing window and its distance band, the camera code per window, whether its own
  detector fired. It never carries the entity's true type.
- **Memory:** the Bayesian change-aware type memory (`TypeStateModel`, fixed clock). It persists through `save` /
  `load` (JSON) and survives a process restart exactly.
- **Intent:** the decision-theoretic mapping with value of information (`infoseek.choose`). Its outputs are
  bounded: a grid intent, OBSERVE or HOLD, sanitised to the expression channels' ranges. Each decision draws from
  its own seeded stream `[seed, decision index]`, so a decision does not depend on how the process got there.
- `SafetySupervisor`: independent of the decision layer. With its own copy of the risk model, it recomputes the
  predicted risk of the proposed approach from the current belief, and replaces anything above the cap with HOLD.
  Physical safety stays with each body's own monitor and validated controller. The runtime is advisory.
- **Body adapters** (`NervaBody`, `WheeledBody`, `HumanoidBody`): execute a decision through the body's own
  expression adapter and safety monitor, and code what its sensors saw into an `Observation`.
- **Log and replay:** every `decide` and `observe` is appended to a JSONL log. `replay(log)` rebuilds a fresh
  runtime from the logged observations alone and checks that every decision is identical.
- **AffectLink:** `affectlink(decision)` emits validated AffectLink v1 advisory messages for the decision's
  target: `MemoryExpectation`, `ActionTendency` and `ExpressionIntent`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apm_revise.belief import TypeModel
from apm_transfer.experience_state import ExperienceState
from apm_transfer.expression_map import GRID, TimeModel
from apm_transfer.infoseek import HOLD, OBSERVE, STANDOFF_M, Emissions, Task, choose, cost
from apm_transfer.nerva_memory import TypeStateModel
from apm_transfer.sensor_memory import SensorExpectation

DIST_EDGES = (0.3, 0.5)
CUSTOM = -3  # a continuous intent from a non-grid controller (lookup, monotone rule, fixed style, neutral)
CAM_OFFSET = 24
STATE_VERSION = 1


@dataclass(frozen=True)
class Observation:
    identity: str
    kind: str
    band: int
    imu: tuple[int, ...] = ()
    camera: tuple[int, ...] = ()
    detected: bool = False
    context: str = ""
    report: dict = field(default_factory=dict)  # robot-side outcome summary for logs (arrived, d_stop, trips)

    def categories(self, camera_block: bool) -> list[int]:
        return [self.band * 8 + c for c in self.imu] + ([CAM_OFFSET + c for c in self.camera] if camera_block else [])


@dataclass(frozen=True)
class Decision:
    n: int
    identity: str
    kind: str
    action: int            # grid index, OBSERVE (-1) or HOLD (-2)
    windows: int
    standoff: float
    proximity: float
    tempo: float
    hesitation: float
    p_risk: float          # expected perturbation readout (b · r)
    uncertainty: float     # normalised belief entropy
    predicted_risk: float  # expected strong perturbations of the executed action
    voi: float
    vetoed: bool
    reason: str

    def intent(self):
        if self.action == CUSTOM:
            from apm_express.intent import ExpressionIntent
            from apm_express.styles import NEUTRAL
            return ExpressionIntent(**{**NEUTRAL.as_dict(), "proximity": self.proximity, "tempo": self.tempo,
                                       "hesitation": self.hesitation}).sanitized()
        return GRID[2] if self.action < 0 else GRID[self.action]


class SafetySupervisor:
    """Independent check of each proposed approach: predicted risk from the current belief ≤ the task's cap, or
    HOLD. It has its own copy of the risk model and never trusts the decision layer's diagnostics."""

    def __init__(self, E: np.ndarray, time_coef: list[float], task: Task):
        self.em = Emissions.from_model(np.array(E, float), camera_block=np.shape(E)[1] > 24)
        self.tm = TimeModel(list(time_coef))
        self.task = task
        self.vetoes = 0

    def risk(self, action: int, windows: int, b: np.ndarray) -> float:
        return 0.0 if action < 0 else cost(action, windows, b, self.tm, self.em, self.task)[1]

    def check(self, action: int, windows: int, b: np.ndarray) -> tuple[int, int, bool]:
        if action >= 0 and self.risk(action, windows, b) > self.task.risk_cap + 1e-12:
            self.vetoes += 1
            return HOLD, 1, True
        return action, windows, False


class Runtime:
    def __init__(self, model_cfg: dict, time_coef: list[float], task: Task, voi: dict, lam: float = 0.02,
                 seed: int = 0, camera: bool = True, observe_imu: bool = True, log: Path | None = None,
                 policy: str = "voi", supervise: bool = True):
        self.model_cfg, self.time_coef, self.task, self.voi = model_cfg, list(time_coef), task, voi
        self.lam, self.seed, self.camera, self.observe_imu, self.policy = lam, seed, camera, observe_imu, policy
        # APM's predicted-risk supervisor. Conventional baselines and exploration have no learned risk model, so
        # they run without it; every body's physical safety layer applies regardless.
        self.supervise = supervise
        self.model = TypeModel.from_dict(model_cfg["model"])
        em = Emissions.from_model(self.model.E, camera_block=self.model.E.shape[1] > 24)
        self.camera_block = em.cam is not None and camera
        self.em = Emissions(em.imu, em.cam if camera else None, observe_imu=observe_imu)
        self.tm = TimeModel(self.time_coef)
        self.memory = TypeStateModel(self.model, lam, transition_each_meeting=True)
        self.safety = SafetySupervisor(self.model.E, self.time_coef, task)
        self.n = 0
        self.model_hash = hashlib.sha256(json.dumps(model_cfg["model"], sort_keys=True).encode()).hexdigest()[:16]
        self.log_path = log
        self.tracks: dict[str, tuple[str, str]] = {}  # AffectLink track id -> (identity, kind)
        self.sensor_expectation: SensorExpectation | None = None  # optional pilot, independent of navigation/risk
        self.experience: ExperienceState | None = None  # optional: drives the expressive program (designed map)

    # ── the loop ─────────────────────────────────────────────────────────────
    def decide(self, identity: str, kind: str, context: str = "") -> Decision:
        b = self.memory.belief(identity, kind)
        p = float(b @ self.model.r)
        u = float(-np.sum(b * np.log(np.clip(b, 1e-12, 1.0))) / math.log(len(b))) if len(b) > 1 else 0.0
        rng = np.random.default_rng([self.seed, self.n])
        custom = None
        if self.policy in ("greedy", "voi", "uncertain"):
            a, w, diag = choose(self.policy, b, self.model.pi, self.tm, self.em, self.task, lam=self.lam,
                                horizon=self.voi["horizon"], gap=self.voi["gap"], u_trigger=self.voi["u_trigger"],
                                rng=rng)
        else:
            custom = self._baseline_intent(p, u, rng)
            a, w, diag = CUSTOM, custom[3], {}
        check_a = a if custom is None else self._nearest_grid(custom[0], custom[1])
        a2, w, vetoed = self.safety.check(check_a, w, b) if self.supervise else (check_a, w, False)
        if vetoed:
            a, custom = a2, None
        standoff = STANDOFF_M if a in (OBSERVE, HOLD) else 0.0
        if custom is None:
            it = GRID[2] if a < 0 else GRID[a]
            prox, tempo, hes = it.proximity, it.tempo, it.hesitation
        else:
            prox, tempo, hes = custom[0], custom[1], custom[2]
        reason = ("vetoed by the safety supervisor" if vetoed else "no safe action: hold" if a == HOLD
                  else "observe from beyond reach" if a == OBSERVE else f"approach ({self.policy})")
        risk = self.safety.risk(check_a if a == CUSTOM else a, w, b)
        d = Decision(self.n, identity, kind, int(a), int(w), standoff, float(prox), float(tempo), float(hes), p, u,
                     risk, float(diag.get("voi", 0.0)), vetoed, reason)
        self.n += 1
        self._log({"t": "decide", "identity": identity, "kind": kind, "context": context,
                   "belief": [round(float(v), 12) for v in b], "decision": asdict(d)})
        return d

    def observe(self, obs: Observation) -> None:
        if self.sensor_expectation is not None:
            self.sensor_expectation.observe(obs.identity, obs.report.get("camera_reaches"))
        if self.experience is not None:
            self.experience.observe(obs.identity, obs.report.get("self_disp_peaks"), obs.report.get("camera_reaches"))
        cats = obs.categories(self.camera_block) if self.observe_imu else (
            [CAM_OFFSET + c for c in obs.camera] if self.camera_block else [])
        self.memory.observe_many(obs.identity, obs.kind, cats, float(obs.detected))
        self._log({"t": "observe", "obs": asdict(obs)})

    # ── baselines that share the runtime (memory, logging, supervisor) ───────
    def _baseline_intent(self, p: float, u: float, rng: np.random.Generator) -> tuple[float, float, float, int]:
        """(proximity, tempo, hesitation, windows) for non-decision-theoretic policies.

        Policy strings:
        - "lookup[:c=..,a=..]": H's lookup (E0) or the retuned lookup (E1);
        - "monotone:lo=..,width=..": the tuned monotonic rule (E2);
        - "fixed:proximity=..,tempo=..,hesitation=..": a fixed style, the same every time;
        - "neutral": neutral motion at the hand-over distance;
        - "random": a random grid intent (exploration);
        - "reactive": memory-free; neutral approach, 3 windows, leaves when disturbed (the body aborts on its own
          detector).
        """
        from apm_express.states import Readout
        from apm_transfer.controllers import lookup_tuned, monotone
        name, _, args = self.policy.partition(":")
        kw = {k: float(v) for k, v in (x.split("=") for x in args.split("/") if x)}
        rd = Readout(p, u, self.memory.g)
        if name == "lookup":
            it = lookup_tuned(rd, **kw)
            return it.proximity, it.tempo, it.hesitation, 1
        if name == "monotone":
            it = monotone(rd, **kw)
            return it.proximity, it.tempo, it.hesitation, 1
        if name == "fixed":
            return kw.get("proximity", 0.5), kw.get("tempo", 1.0), kw.get("hesitation", 0.0), 1
        if name == "neutral":
            return 0.75, 1.0, 0.0, 1
        if name == "reactive":
            return 0.75, 0.7, 0.0, 3
        if name == "random":
            g = GRID[int(rng.integers(len(GRID)))]
            return g.proximity, g.tempo, g.hesitation, int(rng.integers(1, 4))
        raise ValueError(f"unknown policy {self.policy}")

    @staticmethod
    def _nearest_grid(proximity: float, tempo: float) -> int:
        return int(np.argmin([abs(g.proximity - proximity) + 0.1 * abs(g.tempo - tempo) for g in GRID]))

    # ── persistence, logging, replay ────────────────────────────────────────
    def state(self) -> dict:
        state = {"version": STATE_VERSION, "model_hash": self.model_hash, "n": self.n, "seed": self.seed,
                "memory": self.memory.state(), "vetoes": self.safety.vetoes, "tracks": self.tracks}
        if self.sensor_expectation is not None:
            state["sensor_expectation"] = self.sensor_expectation.state()
        if self.experience is not None:
            state["experience"] = self.experience.state()
        return state

    def save(self, path: Path) -> None:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state()), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash leaves the previous state

    def load(self, path: Path) -> None:
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        if s["version"] != STATE_VERSION or s["model_hash"] != self.model_hash:
            raise ValueError("saved state belongs to another runtime version or type model")
        self.n, self.seed, self.safety.vetoes = int(s["n"]), int(s["seed"]), int(s["vetoes"])
        self.memory.load_state(s["memory"])
        self.tracks = {k: (str(v[0]), str(v[1])) for k, v in s.get("tracks", {}).items()}
        if "sensor_expectation" in s:
            self.sensor_expectation = SensorExpectation()
            self.sensor_expectation.load(s["sensor_expectation"])
        else:
            self.sensor_expectation = None
        if "experience" in s:
            self.experience = ExperienceState()
            self.experience.load(s["experience"])
        else:
            self.experience = None

    def _log(self, rec: dict) -> None:
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    # ── AffectLink v1 ────────────────────────────────────────────────────────
    def ingest(self, message: Any) -> None:
        """AffectLink input: an `EntityObservation` binds a track to a persistent identity when its identity
        confidence is ≥ 0.5; otherwise the track stays anonymous (kind prior only)."""
        from apm.protocol.messages import EntityObservation
        from apm.protocol.validation import validate
        validate(message)
        if isinstance(message, EntityObservation):
            ident = message.identity if (message.identity and message.identity_confidence >= 0.5) else (
                f"anon:{message.track_id}")
            self.tracks[message.track_id] = (ident, message.entity_type)

    def identity_of(self, track_id: str) -> tuple[str, str]:
        return self.tracks[track_id]

    def affectlink(self, d: Decision, stamp_s: float, seq: int, source: str = "apm-l") -> list[Any]:
        from apm.protocol.messages import ActionTendency, ExpressionIntent, Header, MemoryExpectation
        from apm.protocol.validation import validate

        def h(stream: str) -> Header:
            return Header(source=source, stream=stream, seq=seq, stamp_s=stamp_s)
        lo, hi = float(self.model.r.min()), float(self.model.r.max())
        p_adv = float(np.clip((d.p_risk - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
        approach = 0.0 if d.action < 0 else float(d.proximity)
        msgs = [MemoryExpectation(header=h("expectation"), target=d.identity, context=d.kind, p_adverse=p_adv,
                                  p_benign=1.0 - p_adv, familiarity=float(min(1.0, self.memory.k.get(d.identity, 0)
                                                                              / max(self.n, 1))),
                                  predictability=1.0 - d.uncertainty, confidence=1.0 - d.uncertainty),
                ActionTendency(header=h("tendency"), target=d.identity, approach=approach,
                               avoid=float(1.0 - approach) if d.action >= 0 else 1.0,
                               orient=1.0 if d.action == OBSERVE else 0.0,
                               explore=float(np.clip(d.voi / 10.0, 0.0, 1.0))),
                # L's intent channels in AffectLink's terms: energy = tempo, openness = closeness without hesitation,
                # posture = proximity, activation = mean(energy, 1 - uncertainty); no urgency is ever signalled
                ExpressionIntent(header=h("expression"),
                                 activation=float(np.clip(0.5 * d.tempo + 0.5 * (1.0 - d.uncertainty), 0.0, 1.0)),
                                 posture=float(np.clip(d.proximity, 0.0, 1.0)),
                                 movement_energy=float(np.clip(d.tempo, 0.0, 1.0)),
                                 openness=float(np.clip(d.proximity * (1.0 - d.hesitation), 0.0, 1.0)),
                                 urgency=0.0)]
        for m in msgs:
            validate(m)
        return msgs


def _same(a: dict, b: dict, tol: float) -> bool:
    if tol == 0.0:
        return a == b
    for k, v in a.items():
        w = b.get(k)
        if isinstance(v, float) and isinstance(w, (int, float)) and not isinstance(w, bool):
            if abs(v - float(w)) > tol:
                return False
        elif v != w:
            return False
    return set(a) == set(b)


def replay(log_path: Path, factory, tol: float = 0.0) -> dict:
    """Rebuild a runtime from `factory()` and feed it the logged observations in order. Every decision must be
    identical to the logged one. Restarts within the log are transparent, because decisions depend only on the
    memory state and the decision index.

    `tol` > 0 compares float fields within tol and everything else (actions, windows, stand-off, veto) exactly:
    for replays on another platform, where floating-point rounding can differ in the last digit."""
    rt = factory()
    n_dec = mismatches = exact = 0
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["t"] == "decide":
            d = asdict(rt.decide(rec["identity"], rec["kind"], rec.get("context", "")))
            n_dec += 1
            mismatches += int(not _same(d, rec["decision"], tol))
            exact += int(d == rec["decision"])
        else:
            o = rec["obs"]
            rt.observe(Observation(o["identity"], o["kind"], o["band"], tuple(o["imu"]), tuple(o["camera"]),
                                   o["detected"], o.get("context", ""), o.get("report", {})))
    return {"decisions": n_dec, "mismatches": mismatches, "identical": mismatches == 0, "bit_exact": exact}


# ── body adapters: execute a decision through the body's own adapter and monitor; code what it sensed ─────────────

class _Body:
    name = "body"
    dist_edges = DIST_EDGES

    def __init__(self, edges: list[float], camera=None):
        self.edges, self.camera = edges, camera

    def _observation(self, identity: str, kind: str, out: dict, standoff: float) -> Observation:
        from apm_transfer.nerva_world import MIN_QUIET_FRAMES
        d = out["final_distance"]
        band = int(np.searchsorted(self.dist_edges, d))
        imu = () if standoff > 0 else tuple(int(np.searchsorted(self.edges, e[1], side="right"))
                                            for e in out["windows"] if e[10] >= MIN_QUIET_FRAMES)
        detected = any(bool(e[9]) for e in out["windows"] if e[10] >= MIN_QUIET_FRAMES)
        return Observation(identity, kind, band, imu, tuple(out.get("camera", [])), detected,
                           report={"arrived": out["arrived"], "d_stop": d, "trips": out["trips"],
                                   "fell": out["fell"], "windows": len(out["windows"])})

    def _run(self, seed: int, etype: str, d: Decision) -> tuple[dict, dict]:
        raise NotImplementedError

    def encounter(self, seed: int, etype: str, d: Decision) -> tuple[Observation, dict]:
        """Execute decision d against an entity of physical type etype (simulation only). Returns the robot's
        observation and the evaluation-only truth."""
        out, truth = self._run(seed, etype, d)
        return self._observation(d.identity, d.kind, out, d.standoff), truth


class NervaBody(_Body):
    """NERVA biped: H's admitted expression interface, its SafetyMonitor and the frozen neutral controller."""
    name = "nerva"

    def __init__(self, edges: list[float], channels: dict, camera=None):
        super().__init__(edges, camera)
        self.channels = channels

    def _run(self, seed: int, etype: str, d: Decision) -> tuple[dict, dict]:
        from apm_transfer.nerva_world import closed_loop_encounter
        _, out, truth = closed_loop_encounter(seed, etype, d.intent(), self.channels, windows=d.windows,
                                              camera=self.camera, standoff=d.standoff)
        return out, truth


class WheeledBody(_Body):
    """MuJoCo differential-drive robot: its own (v, ω) expression adapter and independent supervisor."""
    name = "wheeled"

    def _run(self, seed: int, etype: str, d: Decision) -> tuple[dict, dict]:
        from apm_transfer.wheeled_world import closed_loop_encounter
        _, out, truth = closed_loop_encounter(seed, etype, d.intent(), None, windows=d.windows, camera=self.camera,
                                              standoff=d.standoff)
        return out, truth


class HumanoidBody(_Body):
    """Booster T1 on its validated walking policy, same (v, ω) adapter and supervisor."""
    name = "humanoid"

    def _run(self, seed: int, etype: str, d: Decision) -> tuple[dict, dict]:
        from apm_transfer.humanoid_world import closed_loop_encounter
        _, out, truth = closed_loop_encounter(seed, etype, d.intent(), None, windows=d.windows, camera=self.camera,
                                              standoff=d.standoff)
        return out, truth
