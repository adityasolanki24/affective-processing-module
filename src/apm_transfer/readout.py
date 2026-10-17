"""Decision read-out: what decision A's frozen rule would do, in a fixed bank of real situations, given a belief.

The bank holds decision-time feature vectors recorded from A0 runs whenever a reliably identified person was the
nearest person within 1.5 m, together with the regulator multiplier in force at that moment.

For a belief (p, confidence) about the nearest person, the read-out:
1. writes the belief into the nearest-person memory features of every bank vector;
2. evaluates the rule's utilities with the frozen predictor;
3. returns the share of vectors where the rule picks a careful action (SLOW/YIELD/DETOUR).

Everything else in the situation is held fixed. A change in the read-out is therefore a change in decisions caused
by the belief alone, isolated from how the other person happened to behave.
"""
from __future__ import annotations

import numpy as np

from apm_research.features import MEM_CONF, MEM_P
from apm_testbeds.wheeled_ctrl import DETOUR, SLOW, YIELD


class DecisionReadout:
    def __init__(self, predictor, lam_c: float, lam_s: float, bank: np.ndarray, multipliers: np.ndarray):
        self.predictor, self.lam_c, self.lam_s = predictor, lam_c, lam_s
        self.bank = np.asarray(bank, dtype=np.float32)
        self.m = np.asarray(multipliers, dtype=np.float32)

    def careful_share(self, p: float, conf: float) -> float:
        return self.careful_shares([(p, conf)])[0]

    def careful_shares(self, beliefs: list[tuple[float, float]], chunk: int = 64) -> list[float]:
        """Batched read-out for many beliefs (one predictor call per `chunk` beliefs)."""
        out: list[float] = []
        nb, a = len(self.bank), 4
        for i in range(0, len(beliefs), chunk):
            part = beliefs[i:i + chunk]
            x = np.tile(self.bank, (len(part), 1))
            x[:, MEM_P[0]] = np.repeat([b[0] for b in part], nb)
            x[:, MEM_CONF[0]] = np.repeat([b[1] for b in part], nb)
            X = np.repeat(x, a, axis=0)
            A = np.tile(np.arange(a), len(x))
            p_col, p_slip, prog = self.predictor.predict(X, A)
            m = np.tile(self.m, len(part))[:, None]
            u = prog.reshape(-1, a) - self.lam_c * m * p_col.reshape(-1, a) - self.lam_s * p_slip.reshape(-1, a)
            careful = np.isin(u.argmax(1), (SLOW, YIELD, DETOUR)).reshape(len(part), nb).mean(1)
            out += [float(v) for v in careful]
        return out
