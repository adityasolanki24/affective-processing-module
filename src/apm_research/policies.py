"""Decision systems B0, B1, B2 (+ ablations), exploration and Oracle (MASTER_PREREGISTRATION.md §1.2).

Every system sees the same `Obs` stream through `observe(obs, t)` and returns one of the same four actions from
`decide(t)`. None of them can reach the simulator, its ground truth or the safety supervisor. The harness
(`apm_research.harness`) falls back to GO whenever a system raises or returns an illegal action.
"""
from __future__ import annotations

import numpy as np

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.wheeled_ctrl import WheeledCtrlAdapter
from apm.core.runtime import Outputs
from apm.protocol.messages import OutcomeHypothesis
from apm_research.features import MEM_CONF, MEM_P, DecisionState
from apm_research.memory import ObsMemory
from apm_research.predictor import ActionPredictor
from apm_testbeds.wheeled_ctrl import DETOUR, GO, N_ACTIONS, YIELD


class Policy:
    name = "policy"

    def begin_lifetime(self) -> None: ...
    def begin_episode(self, t0: float) -> None: ...
    def observe(self, obs, t: float) -> None: ...
    def decide(self, t: float) -> int:
        return GO
    def end_episode(self, t: float) -> None: ...


class B0(Policy):
    """Ordinary controller: always GO (the shared safety supervisor still acts)."""
    name = "B0"


class FeaturePolicy(Policy):
    """Owns the explicit memory and the decision-time features shared by B1, B2 and exploration."""

    def begin_lifetime(self) -> None:
        self.memory = ObsMemory()
        self.state = DecisionState()
        self.last_features: np.ndarray | None = None

    def begin_episode(self, t0: float) -> None:
        self.memory.expire(t0)
        self.state = DecisionState()

    def observe(self, obs, t: float) -> None:
        self.memory.observe(obs, t)
        self.state.observe(obs, t)

    def features(self, t: float) -> np.ndarray:
        self.last_features = self.state.features(t, self.memory.p_collision)
        return self.last_features


class Explore(FeaturePolicy):
    """Randomised safe action variation for predictor training data: uniform action, repeated with prob. 0.5."""
    name = "explore"

    def __init__(self, seed: int | np.random.SeedSequence):
        self.rng = np.random.default_rng(seed)
        self.action = GO

    def decide(self, t: float) -> int:
        self.features(t)
        if self.rng.random() >= 0.5:
            self.action = int(self.rng.integers(N_ACTIONS))
        return self.action


class B1(FeaturePolicy):
    """Prediction + explicit memory + transparent rule: argmax_a progress(a) − λc·P_col(a) − λs·P_slip(a)."""
    name = "B1"

    def __init__(self, predictor: ActionPredictor, lam_c: float, lam_s: float, oracle_rates: dict | None = None):
        self.predictor, self.lam_c, self.lam_s = predictor, lam_c, lam_s
        self.oracle_rates = oracle_rates  # Oracle only: identity → true per-encounter rate (evaluation reference)
        self.last_utility: np.ndarray | None = None

    def features(self, t: float) -> np.ndarray:
        x = super().features(t)
        if self.oracle_rates is not None:
            obs = self.state.last_obs
            assert obs is not None
            people = sorted((e for e in obs.entities if e.kind == "person"), key=lambda e: e.distance)[:len(MEM_P)]
            for i, e in enumerate(people):
                rate = self.oracle_rates.get(e.identity)
                if rate is not None:
                    x[MEM_P[i]], x[MEM_CONF[i]] = rate, 1.0
        return x

    def utility(self, t: float) -> np.ndarray:
        p_col, p_slip, progress = self.predictor.predict_actions(self.features(t))
        return progress - self.lam_c * p_col - self.lam_s * p_slip

    def decide(self, t: float) -> int:
        self.last_utility = self.utility(t)
        return int(np.argmax(self.last_utility))


class AblatedAPM(APM):
    """The APM v0.1 core with optional ablations (no change to the core itself).

    tonic=False: predictions (hypotheses and memory's internal predictions) are not appraised at all.
    memory_frames=False: memory's internal predictions are not raised.
    """

    def __init__(self, *args, tonic: bool = True, memory_frames: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.tonic, self.use_memory_frames = tonic, memory_frames

    def ingest(self, message, now_s=None):
        if not self.tonic and isinstance(message, OutcomeHypothesis):
            return None
        return super().ingest(message, now_s)

    def _memory_frames(self, margins, now_s):
        if not (self.tonic and self.use_memory_frames):
            return []
        return super()._memory_frames(margins, now_s)


class B2(B1):
    """B1 plus the APM v0.1 core: λc' = λc·(1 + k_pad·(max(0, A) + max(0, −V))); DETOUR += k_t·avoid_max;
    YIELD += k_t·disengage_max (max over tracked targets). Ablations via flags; `trace` replays PAD/tendency
    readouts recorded elsewhere (shuffled-history negative control)."""
    name = "B2"

    def __init__(self, predictor: ActionPredictor, lam_c: float, lam_s: float, k_pad: float, k_t: float,
                 tonic: bool = True, memory_frames: bool = True, persistent: bool = True,
                 trace: list[tuple[float, float, float, float]] | None = None, name: str = "B2"):
        super().__init__(predictor, lam_c, lam_s)
        self.k_pad, self.k_t = k_pad, k_t
        self.tonic, self.memory_frames, self.persistent, self.trace = tonic, memory_frames, persistent, trace
        self.name = name
        self.readouts: list[tuple[float, float, float, float]] = []

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.adapter = WheeledCtrlAdapter(Emitter("wheeled-ctrl"))
        self.apm = AblatedAPM(profile=self.adapter.profile(), tonic=self.tonic, memory_frames=self.memory_frames)
        self.apm.ingest(self.adapter.profile(), now_s=0.0)
        self.outputs: Outputs | None = None
        self.readouts = []

    def begin_episode(self, t0: float) -> None:
        super().begin_episode(t0)
        self.outputs = self.apm.step(t0)

    def observe(self, obs, t: float) -> None:
        super().observe(obs, t)
        for m in self.adapter.step(obs, t):
            self.apm.ingest(m)
        self.outputs = self.apm.step(t)

    def readout(self) -> tuple[float, float, float, float]:
        out = self.outputs
        assert out is not None
        v, a = out.affect.valence, out.affect.arousal
        targets = [x for x in out.tendencies if x.target != "self"]
        avoid = max((x.avoid for x in targets), default=0.0)
        disengage = max((x.disengage for x in targets), default=0.0)
        return float(v), float(a), float(avoid), float(disengage)

    def decide(self, t: float) -> int:
        live = self.readout()
        self.readouts.append(live)
        v, a, avoid, disengage = self.trace[(len(self.readouts) - 1) % len(self.trace)] if self.trace else live
        p_col, p_slip, progress = self.predictor.predict_actions(self.features(t))
        lam_c = self.lam_c * (1.0 + self.k_pad * (max(0.0, a) + max(0.0, -v)))
        u = progress - lam_c * p_col - self.lam_s * p_slip
        u[DETOUR] += self.k_t * avoid
        u[YIELD] += self.k_t * disengage
        if not self.persistent:  # no persistent affect: state lives for one decision period only
            self.apm.affect.p[:] = self.apm.affect.baseline
            self.apm.affect.r[:] = 0.0
            self.apm.tendency_filter.state.clear()
        self.last_utility = u
        return int(np.argmax(u))
