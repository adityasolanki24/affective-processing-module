"""G3 stateful predictors: the online closed-loop state reproduces the offline training inputs exactly (for every
kind and control), and states are causal (future decisions do not change past predictions)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from apm_select.stateful import KINDS, StatefulPredictor, StatefulRule  # noqa: E402
from apm_testbeds.wheeled_ctrl import N_ACTIONS  # noqa: E402


def _lifetimes(n, T, width, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        Z = rng.normal(size=(T, width)).astype(np.float32)
        A = rng.integers(0, N_ACTIONS, T)
        y = (rng.random(T) < 0.2).astype(np.float32)
        out.append({"Z": Z, "A": A, "y_col": y, "y_slip": y[::-1].copy(),
                    "y_prog": rng.normal(size=T).astype(np.float32), "mask": np.ones(T, bool)})
    return out


class _Replay(StatefulRule):
    def __init__(self, models, Z, control=None, shuffle=None):
        super().__init__(models, 1.0, 1.0, control=control, shuffle_history=shuffle)
        self.Z, self.outputs = Z, []

    def features(self, t):
        return self.Z[self.k][:-3] if self.kind == "pad" else self.Z[self.k]

    def apm_readout(self):
        return self.Z[self.k][-3:]

    def decide(self, t):
        a = super().decide(t)
        return a


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("control", [None, "reset", "shuffle"])
def test_online_state_matches_offline_inputs(kind, control):
    width = 9
    data = _lifetimes(4, 40, width, 0)
    model = StatefulPredictor(kind, width, seed=1).fit(data[:3], data[3:], epochs=1)
    Z = data[0]["Z"]
    other = data[1]["Z"]
    shuffle = other if control == "shuffle" else None
    ctrl = control if control != "shuffle" else None
    offline = {a: model.predict_lifetime(Z, np.full(len(Z), a), control=ctrl, shuffle_from=shuffle)
               for a in range(N_ACTIONS)}
    pol = _Replay([model], Z, control=ctrl, shuffle=shuffle)
    pol.begin_lifetime()
    if kind == "pad":
        pol.apm_reset = lambda: None
    for k in range(len(Z)):
        pol.decide(k * 0.5)
        u = pol.last_utility
        exp = np.array([offline[a][k, 2] - offline[a][k, 0] - offline[a][k, 1] for a in range(N_ACTIONS)])
        assert np.allclose(u, exp, atol=1e-5), (kind, control, k)


@pytest.mark.parametrize("kind", ["ema", "gru", "fastslow", "lds"])
def test_states_are_causal(kind):
    width = 7
    data = _lifetimes(3, 30, width, 2)
    model = StatefulPredictor(kind, width, seed=0).fit(data[:2], data[2:], epochs=1)
    Z = data[0]["Z"]
    A = data[0]["A"]
    full = model.predict_lifetime(Z, A)
    for cut in (5, 17, 29):
        part = model.predict_lifetime(Z[:cut + 1], A[:cut + 1])
        assert np.allclose(part[cut], full[cut], atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    data = _lifetimes(3, 25, 8, 4)
    for kind in ("ema", "gru", "fastslow", "lds", "none"):
        m = StatefulPredictor(kind, 8, seed=0).fit(data[:2], data[2:], epochs=1)
        torch.save({"state": m.net.state_dict(), "mean": m.mean, "scale": m.scale, "kind": kind, "width": 8,
                    "seed": 0}, tmp_path / "m.pt")
        m2 = StatefulPredictor.load(torch.load(tmp_path / "m.pt", weights_only=False))
        Z, A = data[0]["Z"], data[0]["A"]
        assert np.allclose(m.predict_lifetime(Z, A), m2.predict_lifetime(Z, A))
