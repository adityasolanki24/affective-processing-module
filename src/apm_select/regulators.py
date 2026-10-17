"""G1 regulators: persistent risk regulation on top of B1's predictor + memory + transparent rule.

Every system here chooses `argmax_a progress(a) − λc·m(t)·P_col(a) − λs·P_slip(a)` (+ tendency bonuses for the B2
family), with the *same* frozen predictor, the same explicit memory, the same features and the same `Obs` stream as
B1 and B2 (programme F). They differ only in how the collision-cost multiplier m(t) is produced:

  R0/R1  static          m ≡ 1                                       (B1; R1 = B1 with an equal tuning budget)
  R2     EMA             s ← s + (1 − e^(−dt/τ))(x − s); on a collision s ← s + c(1 − s); m = 1 + k·s
  R3     hysteresis      as R2 with τ_up when x > s and τ_down otherwise
  R4     Bayes hazard    Gamma–Poisson rate of collisions per minute of close (< 1 m) person exposure with
                         exponential forgetting τ_f; m = 1 + k·min(5, UCB_z / h0)
  R5     learned scalar  m = exp(clip(w·φ, −1.5, 2)), φ = [1, x, EMA_2s(x), EMA_30s(x), collision trace (10 s),
                         intervention trace (5 s), people within 1.5 m / 3, |v| / 0.8]; w trained by ES (scripts)
  R6     B2              APM v0.1 PAD: m = 1 + k_pad·(max(0, A) + max(0, −V)), tendency bonuses k_t
  R7     B2, PAD → appraisal   m = 1 + k·u_threat (the APM's instantaneous tonic appraisal), tendencies kept
  R8     B2, PAD → filter      m = 1 + k·s, s = R2-style low-pass of u_threat (τ, collision impulse c), tendencies kept

x(t) = noisy-OR of the robot-side TTC collision estimates of all tracked entities (the same estimates the APM adapter
turns into collision hypotheses). All filters are updated at every observation (20 Hz) and persist across episodes of
a lifetime, like B2's affect. `trace` replays m (and the tendency bonuses) recorded elsewhere: the shuffled-history
and time-shifted negative controls.
"""
from __future__ import annotations

import math

import numpy as np

from apm_research.policies import B1, B2
from apm_testbeds.wheeled_ctrl import DETOUR, YIELD

CLOSE_M, NEAR_M = 1.0, 1.5


def threat(obs) -> float:
    return 1.0 - math.prod(1.0 - min(1.0, max(0.0, e.ttc_p)) for e in obs.entities)


def _alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-max(0.0, dt) / max(1e-6, tau))


class Regulator:
    """Produces the collision-cost multiplier m(t) ≥ 0 from the observation stream only."""
    def reset(self) -> None:
        self.t: float | None = None

    def update(self, obs, t: float) -> None: ...

    def multiplier(self) -> float:
        return 1.0

    def _dt(self, t: float) -> float:
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        return dt


class Static(Regulator):
    pass


class EMA(Regulator):
    def __init__(self, tau: float, k: float, c: float):
        self.tau, self.k, self.c = tau, k, c
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.s = 0.0

    def _tau(self, x: float) -> float:
        return self.tau

    def update(self, obs, t: float) -> None:
        dt = self._dt(t)
        x = threat(obs)
        self.s += _alpha(dt, self._tau(x)) * (x - self.s)
        if any(ev[0] == "collision" for ev in obs.events):
            self.s += self.c * (1.0 - self.s)

    def multiplier(self) -> float:
        return 1.0 + self.k * self.s


class Hysteresis(EMA):
    def __init__(self, tau_up: float, tau_down: float, k: float, c: float):
        super().__init__(tau_up, k, c)
        self.tau_up, self.tau_down = tau_up, tau_down

    def _tau(self, x: float) -> float:
        return self.tau_up if x > self.s else self.tau_down


class BayesHazard(Regulator):
    """Collisions per minute of close person exposure, Gamma(a, b) posterior with forgetting; UCB = (a + z√a)/b."""
    A0, B0 = 0.1, 0.2   # prior: 0.1 events in 0.2 exposure-minutes (h0 = 0.5 / min)

    def __init__(self, tau_f: float, z: float, k: float):
        self.tau_f, self.z, self.k = tau_f, z, k
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.a, self.b = 0.0, 0.0  # evidence beyond the prior

    def update(self, obs, t: float) -> None:
        dt = self._dt(t)
        keep = math.exp(-dt / self.tau_f)
        self.a *= keep
        self.b *= keep
        if any(e.kind == "person" and e.distance < CLOSE_M for e in obs.entities):
            self.b += dt / 60.0
        self.a += sum(ev[0] == "collision" for ev in obs.events)

    def ucb(self) -> float:
        a, b = self.A0 + self.a, self.B0 + self.b
        return (a + self.z * math.sqrt(a)) / b

    def multiplier(self) -> float:
        return 1.0 + self.k * min(5.0, self.ucb() / (self.A0 / self.B0))


class LearnedScalar(Regulator):
    N_PHI = 8

    def __init__(self, w):
        self.w = np.asarray(w, float)
        assert self.w.shape == (self.N_PHI,)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.x = self.fast = self.slow = self.col = self.intv = 0.0
        self.near = self.speed = 0.0

    def update(self, obs, t: float) -> None:
        dt = self._dt(t)
        self.x = threat(obs)
        self.fast += _alpha(dt, 2.0) * (self.x - self.fast)
        self.slow += _alpha(dt, 30.0) * (self.x - self.slow)
        self.col *= math.exp(-dt / 10.0)
        self.intv *= math.exp(-dt / 5.0)
        for ev in obs.events:
            self.col += ev[0] == "collision"
            self.intv += ev[0] == "intervention"
        self.near = sum(e.kind == "person" and e.distance < NEAR_M for e in obs.entities) / 3.0
        self.speed = abs(obs.v) / 0.8

    def phi(self) -> np.ndarray:
        return np.array([1.0, self.x, self.fast, self.slow, min(3.0, self.col), min(3.0, self.intv), self.near,
                         self.speed])

    def multiplier(self) -> float:
        return float(math.exp(min(2.0, max(-1.5, float(self.w @ self.phi())))))


def make_regulator(kind: str, p: dict) -> Regulator:
    if kind == "static":
        return Static()
    if kind == "ema":
        return EMA(p["tau"], p["k"], p["c"])
    if kind == "hysteresis":
        return Hysteresis(p["tau_up"], p["tau_down"], p["k"], p["c"])
    if kind == "bayes":
        return BayesHazard(p["tau_f"], p["z"], p["k"])
    if kind == "learned":
        return LearnedScalar(p["w"])
    raise ValueError(kind)


class Regulated(B1):
    """B1 + a persistent regulator of the collision cost (R0–R5)."""

    def __init__(self, predictor, lam_c: float, lam_s: float, regulator: Regulator,
                 trace: list[tuple[float, float, float]] | None = None, name: str = "R"):
        super().__init__(predictor, lam_c, lam_s)
        self.regulator, self.trace, self.name = regulator, trace, name
        self.readouts: list[tuple[float, float, float]] = []

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.regulator.reset()
        self.readouts = []

    def observe(self, obs, t: float) -> None:
        super().observe(obs, t)
        self.regulator.update(obs, t)

    def modulation(self) -> tuple[float, float, float]:
        """(m, DETOUR bonus, YIELD bonus)."""
        return self.regulator.multiplier(), 0.0, 0.0

    def decide(self, t: float) -> int:
        live = self.modulation()
        self.readouts.append(live)
        m, b_detour, b_yield = self.trace[(len(self.readouts) - 1) % len(self.trace)] if self.trace else live
        p_col, p_slip, progress = self.predictor.predict_actions(self.features(t))
        u = progress - self.lam_c * m * p_col - self.lam_s * p_slip
        u[DETOUR] += b_detour
        u[YIELD] += b_yield
        self.last_utility = u
        return int(np.argmax(u))


class B2Variant(B2):
    """R6 (`mode="pad"`, the original B2 rule), R7 (`"appraisal"`) and R8 (`"filter"`). The APM v0.1 core runs
    unchanged in all three; only the term that scales the collision cost differs. Tendency bonuses are kept."""

    def __init__(self, predictor, lam_c: float, lam_s: float, k: float, k_t: float, mode: str = "pad",
                 tau: float = 1.0, c: float = 0.0, trace=None, name: str = "R6"):
        super().__init__(predictor, lam_c, lam_s, k_pad=k, k_t=k_t, name=name)
        if mode not in ("pad", "appraisal", "filter"):
            raise ValueError(mode)
        self.mode, self.tau, self.c, self.replay = mode, tau, c, trace

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.s = 0.0
        self.t_last: float | None = None

    def observe(self, obs, t: float) -> None:
        super().observe(obs, t)
        if self.mode == "filter":
            dt = 0.0 if self.t_last is None else t - self.t_last
            self.t_last = t
            self.s += _alpha(dt, self.tau) * (float(self.apm.affect.u[0]) - self.s)
            if any(ev[0] == "collision" for ev in obs.events):
                self.s += self.c * (1.0 - self.s)

    def modulation(self) -> tuple[float, float, float]:
        v, a, avoid, disengage = self.readout()
        if self.mode == "pad":
            m = 1.0 + self.k_pad * (max(0.0, a) + max(0.0, -v))
        elif self.mode == "appraisal":
            m = 1.0 + self.k_pad * float(self.apm.affect.u[0])
        else:
            m = 1.0 + self.k_pad * self.s
        return m, self.k_t * avoid, self.k_t * disengage

    def decide(self, t: float) -> int:
        live = self.modulation()
        self.readouts.append(live)
        m, b_detour, b_yield = self.replay[(len(self.readouts) - 1) % len(self.replay)] if self.replay else live
        p_col, p_slip, progress = self.predictor.predict_actions(self.features(t))
        u = progress - self.lam_c * m * p_col - self.lam_s * p_slip
        u[DETOUR] += b_detour
        u[YIELD] += b_yield
        self.last_utility = u
        return int(np.argmax(u))
