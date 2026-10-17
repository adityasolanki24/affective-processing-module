"""J2 conditioned evidence models, the memory that uses them, and a passive observer (MASTER_PREREGISTRATION §3.2).

**Conditioned type model.** Emissions p(c | band, k) for the dense outcome category c given a *band* that the robot
knows about the encounter:
- `dense_action`: the robot's own speed during observable frames (slow / fast; `EncounterX.action_band`);
- `dense_context`: the situation (frontal approach × crowded; `EncounterX.context_band`).

The model is stored as an ordinary `TypeModel`, with categories flattened to band·C + c and each band block normalised
separately. Programme I's `TypeBelief` update then multiplies by p(c | band, k) unchanged. The band itself is never
evidence about the person: conditioning removes what the robot's action or the situation explains.

Encounters with no observable frame have no band; a collision there is coded with band 0.

`fit_conditional` is EM on per-person counts n[i, band, c]. The M-step normalises within each (type, band).

`RevisionMemoryX`: programme I's memory using `EncounterX`, so it can compute bands.

`Observer`: drives the robot with one policy while another memory learns passively from the same observations. This
is for J2's robot-policy group: does the memory blame people for near-misses the robot itself causes?
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from apm_revise.belief import SMOOTH, TypeModel
from apm_revise.evidence import CODINGS, NEAR_M
from apm_revise.memory import RevisionMemory
from apm_transfer.encounter import EncounterX

C_DENSE = CODINGS["dense"]
BANDS = {"dense_action": 2, "dense_context": 4}


def category_x(enc: EncounterX, coding: str) -> int | None:
    if coding in ("dense", "sparse"):
        return enc.category(coding)
    c = enc.category("dense")
    if c is None:
        return None
    band = enc.action_band() if coding == "dense_action" else enc.context_band()
    return (band or 0) * C_DENSE + c


def fit_conditional(counts: np.ndarray, encounters: np.ndarray, collisions: np.ndarray, K: int, coding: str,
                    restarts: int = 10, iters: int = 300, seed: int = 0) -> TypeModel:
    """counts[i, band, c] → TypeModel with flattened conditional emissions (band blocks normalised)."""
    counts = np.asarray(counts, float)
    n, B, C = counts.shape
    flat = counts.reshape(n, B * C)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        pi = np.full(K, 1.0 / K)
        E = rng.dirichlet(np.ones(C), size=(K, B))           # [K, B, C]
        ll = -np.inf
        for _ in range(iters):
            L = flat @ np.log(E.reshape(K, B * C)).T + np.log(pi)
            m = L.max(1, keepdims=True)
            post = np.exp(L - m)
            z = post.sum(1, keepdims=True)
            gamma = post / z
            new_ll = float((m[:, 0] + np.log(z[:, 0])).sum())
            pi = np.clip(gamma.mean(0), 1e-6, None)
            pi /= pi.sum()
            E = np.einsum("ik,ibc->kbc", gamma, counts) + SMOOTH
            E /= E.sum(2, keepdims=True)
            if abs(new_ll - ll) < 1e-8:
                ll = new_ll
                break
            ll = new_ll
        if best is None or ll > best[0]:
            best = (ll, pi, E, gamma)
    assert best is not None
    ll, pi, E, gamma = best
    r = (gamma.T @ np.asarray(collisions, float)) / np.maximum(1e-9, gamma.T @ np.asarray(encounters, float))
    order = np.argsort(r)
    return TypeModel(coding, pi[order], E[order].reshape(K, B * C), r[order], ll, n)


def bic_conditional(m: TypeModel, bands: int) -> float:
    C = m.E.shape[1] // bands
    n_par = (m.K - 1) + m.K * bands * (C - 1)
    return -2.0 * m.loglik + n_par * math.log(max(1, m.n_units))


@dataclass
class RevisionMemoryX(RevisionMemory):
    def observe(self, obs, t: float) -> None:
        self.now = t
        people = [e for e in obs.entities if e.kind == "person"]
        near = sum(e.distance < NEAR_M for e in people)
        for e in people:
            enc = self.open.get(e.track)
            if enc is None:
                enc = self.open[e.track] = EncounterX(e.track, start=t)
                enc.p_start = self.posterior_p(e.identity)[0] if e.identity else None
            if e.identity is not None and e.identity_confidence >= enc.confidence:
                enc.identity, enc.confidence = e.identity, e.identity_confidence
            assert isinstance(enc, EncounterX)
            enc.observe_x(e.distance, obs.v, obs.action, e.bearing, near - (e.distance < NEAR_M) > 0)
            self.last_seen[e.track] = t
        for ev in obs.events:
            if ev[0] == "collision" and ev[1] in self.open:
                self.open[ev[1]].collision = True
        self.expire(t)

    def _end(self, enc, t: float) -> None:
        coding = self.model.coding
        if coding in ("dense", "sparse"):
            super()._end(enc, t)
            return
        # as RevisionMemory._end, with the band-flattened category
        klass = enc.g_class()
        cat = category_x(enc, coding)
        self.counts[klass] += 1
        if not enc.reliable:
            self.counts["unreliable"] += 1
        if cat is None:
            self.counts["censored"] += 1
        logp = None
        if cat is not None and enc.reliable:
            logp = round(math.log(max(1e-12, float(self.posterior(enc.identity) @ self.model.E[:, cat]))), 5)
        self.log.append((round(t, 2), enc.identity, round(enc.confidence, 3), klass, enc.reliable, enc.p_start, cat,
                         logp))
        if cat is None or not enc.reliable:
            return
        from apm_revise.belief import TypeBelief
        b = self.beliefs.get(enc.identity)
        if b is None:
            b = self.beliefs[enc.identity] = TypeBelief(self.model, self.lam, t)
        b.update(cat, enc.confidence, t)


class RevisingX:
    """Factory helper: programme I's `Revising` policy with `RevisionMemoryX` (conditioned codings)."""

    @staticmethod
    def make(predictor, lam_c: float, lam_s: float, regulator, model: TypeModel, lam: float, name: str = "IX"):
        from apm_revise.policy import Revising

        class _P(Revising):
            def begin_lifetime(self) -> None:
                super().begin_lifetime()
                self.memory = RevisionMemoryX(self.type_model, self.lam)  # type: ignore[assignment]
        return _P(predictor, lam_c, lam_s, regulator, model, lam, None, name=name)


class Observer:
    """Drives the robot with `driver`; a fresh memory from `memory_factory` learns passively from the same
    observations in each lifetime (J2 policy group)."""

    def __init__(self, driver, memory_factory):
        self.driver, self.memory_factory = driver, memory_factory
        self.memory = memory_factory()
        self.name = getattr(driver, "name", "observer")

    def __getattr__(self, item):
        return getattr(self.driver, item)

    def begin_lifetime(self) -> None:
        self.memory = self.memory_factory()
        self.driver.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.memory.expire(t0)
        self.driver.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        self.memory.observe(obs, t)
        self.driver.observe(obs, t)

    def decide(self, t: float) -> int:
        return self.driver.decide(t)

    def end_episode(self, t: float) -> None:
        self.driver.end_episode(t)
