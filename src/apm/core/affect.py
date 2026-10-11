"""Persistent affect: a bounded attractor (tonic context) plus decaying phasic impulses (new events).

PAD (valence, arousal, dominance) is an *experimental, interpretable readout* of the module's internal control
state, not an established universal emotional state. It is kept because it is compact and comparable across
embodiments; Stage-2 experiments test whether it adds anything beyond outcome memory and prediction.

Dynamics (APM design, after the attractor proposal in the APM design report; not a published law):

    p*(t)  = tanh(W u(t) + b)                       context-dependent target from the *current* tonic appraisal u
    r(t)   = r(t0) · exp(−(t − t0)/τ_r) + impulses   phasic trace; each observed outcome adds one impulse, once
    dp/dt  = −Λ (p − p0) + K (p* − p) + B r

Integrated exactly over each interval with u held constant (piecewise-constant input), so the trajectory does not
depend on how often the robot sends messages: an unchanged scene drives p towards a fixed point and never
accumulates. The tonic part is bounded by construction (|p*| < 1, convex combination with p0); a burst of strong
phasic events can still reach the bounds, so p is clipped to [−1, 1] and the diagnostics report the fraction of time
spent near them.

Tonic appraisal vector u (from live hypothesis frames and body margins):
    threat  = 1 − Π(1 − relevance·max(−d, 0)·likelihood)     adverse predictions (noisy-OR)
    promise = 1 − Π(1 − relevance·max(d, 0)·likelihood)      beneficial predictions
    strain  = (1 − min body margin)² (0 when no margin is available; uncertainty rises instead)
    control = mean controllability of adverse frames, mapped to [−1, 1] (0 when there are none)
    unknown = mean uncertainty of frames (and missing-data uncertainty)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from apm.core.appraisal import Frame

MODEL_VERSION = "apm-affect-baseline-0.1"
U_NAMES = ("threat", "promise", "strain", "control", "unknown")
# rows: V, A, D; columns: threat, promise, strain, control, unknown  (APM design values)
W = np.array([[-1.6, 1.2, -0.8, 0.0, -0.3],
              [1.4, 0.4, 0.9, 0.0, 0.5],
              [-0.6, 0.2, -0.6, 0.9, -0.4]])
B_BIAS = np.zeros(3)
LAMBDA = np.array([1 / 120.0, 1 / 60.0, 1 / 120.0])  # 1/s, slow return to baseline
KAPPA = np.array([1 / 4.0, 1 / 2.0, 1 / 6.0])        # 1/s, approach to the context attractor
PHASIC_TAU_S = 3.0
PHASIC_GAIN = np.array([[0.3, 0.0], [0.0, 0.3], [0.0, -0.15]])  # (V, A, D) ← (valenced, arousing) impulse
NEAR_BOUND = 0.9


def noisy_or(values: list[float]) -> float:
    return 1.0 - math.prod(1.0 - min(1.0, max(0.0, v)) for v in values)


def tonic_input(frames: list[Frame], margins: dict[str, float], missing_margin: bool) -> np.ndarray:
    adverse = [f for f in frames if f.desirability < 0]
    threat = noisy_or([f.relevance * -f.desirability * f.likelihood for f in adverse])
    promise = noisy_or([f.relevance * f.desirability * f.likelihood for f in frames if f.desirability > 0])
    strain = (1.0 - min(margins.values())) ** 2 if margins else 0.0  # matters mainly when margins run low
    control = 2.0 * float(np.mean([f.controllability for f in adverse])) - 1.0 if adverse else 0.0
    unknown_terms = [f.uncertainty for f in frames] + ([0.5] if missing_margin else [])
    unknown = float(np.mean(unknown_terms)) if unknown_terms else 0.0
    return np.array([threat, promise, strain, control, unknown])


@dataclass
class AffectCore:
    baseline: np.ndarray = field(default_factory=lambda: np.zeros(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    r: np.ndarray = field(default_factory=lambda: np.zeros(2))   # phasic (valenced, arousing)
    u: np.ndarray = field(default_factory=lambda: np.zeros(len(U_NAMES)))
    t: float | None = None
    near_bound_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    elapsed_s: float = 0.0

    def target(self) -> np.ndarray:
        return np.tanh(W @ self.u + B_BIAS)

    def advance(self, now_s: float) -> None:
        """Integrate exactly from the last time to now with the current u and decaying phasic trace."""
        if self.t is None:
            self.t = now_s
            return
        dt = now_s - self.t
        if dt <= 0:
            return
        rate = LAMBDA + KAPPA
        eq = (LAMBDA * self.baseline + KAPPA * self.target()) / rate
        drive = PHASIC_GAIN @ self.r  # phasic forcing at the interval start, decaying with tau_r
        # p' = -rate (p - eq) + drive·exp(-s/τ): exact solution over [0, dt]
        decay = np.exp(-rate * dt)
        k = 1.0 / PHASIC_TAU_S
        phasic = drive * np.where(np.isclose(rate, k), dt * decay,
                                  (np.exp(-k * dt) - decay) / np.where(np.isclose(rate, k), 1.0, rate - k))
        self.p = np.clip(eq + (self.p - eq) * decay + phasic, -1.0, 1.0)
        self.r = self.r * math.exp(-dt / PHASIC_TAU_S)
        self.near_bound_s += dt * (np.abs(self.p) > NEAR_BOUND)
        self.elapsed_s += dt
        self.t = now_s

    def set_context(self, u: np.ndarray) -> None:
        self.u = np.asarray(u, dtype=float)

    def impulse(self, frame: Frame) -> None:
        """One observed outcome → one phasic impulse (valenced by desirability, arousing by unexpectedness)."""
        weight = frame.relevance * (1.0 - frame.uncertainty)
        self.r += weight * np.array([frame.desirability, abs(frame.desirability) * (1.5 - frame.expectedness)])
        self.r = np.clip(self.r, -3.0, 3.0)

    def saturation(self) -> dict[str, float]:
        if self.elapsed_s <= 0:
            return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        frac = self.near_bound_s / self.elapsed_s
        return {"valence": float(frac[0]), "arousal": float(frac[1]), "dominance": float(frac[2])}

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> AffectCore:
        core = cls()
        for k, v in data.items():
            setattr(core, k, np.asarray(v, dtype=float) if isinstance(v, list) else v)
        return core
