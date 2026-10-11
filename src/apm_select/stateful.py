"""G3: action-conditioned consequence predictors with and without persistent internal state.

Every model predicts, for the action actually taken at a decision, P(collision within 2 s), P(slip within 2 s) and
goal progress within 2 s (programme F's predictor targets and labels), from the same decision features z_t (F's 41
features, optionally ⊕ APM readouts) and an internal state built only from past decisions of the same lifetime:

  none      MLP(z_t)                                                        (F's predictor architecture)
  ema       MLP(z_t ⊕ EMA_2s(z) ⊕ EMA_30s(z))                                fixed exponential smoothing
  lds       MLP(z_t ⊕ s_t), s_t = σ(d) ⊙ s_{t−1} + (1 − σ(d)) ⊙ (W z_t), learned diagonal decay, 16 dims
  gru       MLP(z_t ⊕ GRU_32(z_{t−W+1..t})), W = 20 decisions (10 s)        learned recurrent state
  fastslow  MLP(z_t ⊕ GRU_32(window) ⊕ EMA_120s(z))       fast learned + slow lifetime-persistent state
  pad       MLP(z_t ⊕ PAD (V, A, D) ⊕ tonic appraisal u (5))                 programme F's affect state as input

Controls (evaluation only): `reset` (window of length 1 / zero EMAs: no history) and `shuffle` (history taken from
another lifetime at the same position). States use only decisions at or before t (tested). Training: torch, Adam,
BCE + BCE + MSE, early stopping on validation lifetimes; serving with torch.no_grad on one CPU thread.
"""
from __future__ import annotations

import copy

import numpy as np

from apm_testbeds.wheeled_ctrl import N_ACTIONS

KINDS = ("none", "ema", "lds", "gru", "fastslow", "pad")
WINDOW, DT_DECISION = 20, 0.5
EMA_TAUS = {"ema": (2.0, 30.0), "fastslow": (120.0,)}
HIDDEN = 64


def ema_stream(Z: np.ndarray, taus: tuple[float, ...], reset: bool = False) -> np.ndarray:
    """Causal EMAs of a lifetime's decision stream Z [T, D] → [T, D·len(taus)] (value after including step t)."""
    out = np.zeros((len(Z), Z.shape[1] * len(taus)), np.float32)
    if reset:
        return out
    for j, tau in enumerate(taus):
        a = 1.0 - np.exp(-DT_DECISION / tau)
        s = np.zeros(Z.shape[1], np.float32)
        for t in range(len(Z)):
            s = s + a * (Z[t] - s)
            out[t, j * Z.shape[1]:(j + 1) * Z.shape[1]] = s
    return out


def windows(Z: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """[T, w, D] windows ending at t (zero-padded before the lifetime start)."""
    pad = np.concatenate([np.zeros((w - 1, Z.shape[1]), Z.dtype), Z])
    return np.stack([pad[t:t + w] for t in range(len(Z))])


def _onehot(a: np.ndarray) -> np.ndarray:
    o = np.zeros((len(a), N_ACTIONS), np.float32)
    o[np.arange(len(a)), a] = 1.0
    return o


class StatefulPredictor:
    """One model of a given kind. `fit` takes per-lifetime arrays; `predict_lifetime` gives per-decision outputs."""

    def __init__(self, kind: str, width: int, seed: int = 0):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.width, self.seed = kind, width, seed
        self.mean = np.zeros(width, np.float32)
        self.scale = np.ones(width, np.float32)
        self.net = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ inputs
    def _static_extra(self, Z: np.ndarray, control: str | None) -> np.ndarray:
        if self.kind in ("ema", "fastslow"):
            return ema_stream(Z, EMA_TAUS[self.kind], reset=control == "reset")
        return np.zeros((len(Z), 0), np.float32)

    def inputs(self, Z_raw: np.ndarray, control: str | None = None, shuffle_from: np.ndarray | None = None):
        """Per-lifetime model inputs from raw decision features [T, width] (z-scored with training statistics)."""
        Z = ((Z_raw - self.mean) / self.scale).astype(np.float32)
        H = Z if shuffle_from is None else ((shuffle_from - self.mean) / self.scale).astype(np.float32)
        if shuffle_from is not None and len(H) < len(Z):
            H = np.concatenate([H, np.zeros((len(Z) - len(H), Z.shape[1]), np.float32)])
        H = H[:len(Z)]
        extra = self._static_extra(H, control)
        seq = None
        if self.kind in ("lds", "gru", "fastslow"):
            seq = windows(H, 1 if control == "reset" else WINDOW)
            if shuffle_from is not None:  # the current step itself is always the real one
                seq[:, -1] = Z
        return Z, extra, seq

    # ------------------------------------------------------------------ model
    def _build(self, extra_dim: int):
        import torch
        from torch import nn

        kind, width = self.kind, self.width

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                rep = width + extra_dim
                if kind == "gru" or kind == "fastslow":
                    self.rnn = nn.GRU(width, 32, batch_first=True)
                    rep += 32
                if kind == "lds":
                    self.proj = nn.Linear(width, 16)
                    self.decay = nn.Parameter(torch.zeros(16))
                    rep += 16
                self.mlp = nn.Sequential(nn.Linear(rep + N_ACTIONS, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN),
                                         nn.ReLU(), nn.Linear(HIDDEN, 3))

            def forward(self, z, extra, seq, a):
                parts = [z, extra]
                if kind in ("gru", "fastslow"):
                    _, h = self.rnn(seq)
                    parts.append(h[-1])
                if kind == "lds":
                    lam = torch.sigmoid(self.decay)
                    s = torch.zeros(z.shape[0], 16)
                    for t in range(seq.shape[1]):
                        s = lam * s + (1 - lam) * self.proj(seq[:, t])
                    parts.append(s)
                parts.append(a)
                return self.mlp(torch.cat(parts, 1))
        return Net()

    def fit(self, lifetimes: list[dict], val: list[dict], epochs: int = 40, lr: float = 1e-3, batch: int = 512,
            patience: int = 5):
        """lifetimes: dicts with Z [T, width], A [T], y_col, y_slip, y_prog [T] and `mask` [T] (labelled decisions)."""
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        rng = np.random.default_rng(self.seed)
        allZ = np.concatenate([lt["Z"][lt["mask"]] for lt in lifetimes])
        self.mean, self.scale = allZ.mean(0).astype(np.float32), (allZ.std(0) + 1e-3).astype(np.float32)

        def tensors(lts):
            parts = [self.inputs(lt["Z"]) for lt in lts]
            m = [lt["mask"] for lt in lts]
            cat = lambda xs: np.concatenate(xs) if xs and xs[0] is not None else None  # noqa: E731
            z = np.concatenate([p[0][k] for p, k in zip(parts, m, strict=True)])
            e = np.concatenate([p[1][k] for p, k in zip(parts, m, strict=True)])
            s = cat([p[2][k] for p, k in zip(parts, m, strict=True)]) if parts[0][2] is not None else None
            a = np.concatenate([_onehot(lt["A"][k]) for lt, k in zip(lts, m, strict=True)])
            ys = [np.concatenate([lt[y][k] for lt, k in zip(lts, m, strict=True)]) for y in ("y_col", "y_slip",
                                                                                           "y_prog")]
            T = lambda v: None if v is None else torch.as_tensor(v, dtype=torch.float32)  # noqa: E731
            return [T(v) for v in (z, e, s, a, *ys)]
        tr, va = tensors(lifetimes), tensors(val)
        self.net = net = self._build(tr[1].shape[1])
        bce, mse = torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        def loss(t, idx=None):
            z, e, s, a, c, sl, p = (None if v is None else (v if idx is None else v[idx]) for v in t)
            out = net(z, e, s if s is not None else z[:, None], a)
            return bce(out[:, 0], c) + bce(out[:, 1], sl) + mse(out[:, 2], p)
        best, best_state, stale = np.inf, copy.deepcopy(net.state_dict()), 0
        n = len(tr[0])
        for epoch in range(epochs):
            net.train()
            order = torch.as_tensor(rng.permutation(n))
            for i in range(0, n, batch):
                opt.zero_grad()
                loss(tr, order[i:i + batch]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss(va))
            self.history.append({"epoch": epoch, "val_loss": v})
            if v < best - 1e-5:
                best, best_state, stale = v, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return self

    def predict(self, z, extra, seq, actions) -> np.ndarray:
        import torch
        with torch.no_grad():
            T = lambda v: torch.as_tensor(v, dtype=torch.float32)  # noqa: E731
            out = self.net(T(z), T(extra), T(seq) if seq is not None else T(z)[:, None], T(_onehot(actions)))
            out = out.numpy()
        p = 1.0 / (1.0 + np.exp(-np.clip(out[:, :2], -30, 30)))
        return np.column_stack([p, out[:, 2]])

    def predict_lifetime(self, Z_raw: np.ndarray, A: np.ndarray, control: str | None = None,
                         shuffle_from: np.ndarray | None = None) -> np.ndarray:
        z, e, s = self.inputs(Z_raw, control, shuffle_from)
        return self.predict(z, e, s, A)

    @property
    def parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    @classmethod
    def load(cls, saved: dict) -> StatefulPredictor:
        """From a `torch.save` dict written by scripts/g3_train.py."""
        m = cls(saved["kind"], saved["width"], saved["seed"])
        m.mean, m.scale = np.asarray(saved["mean"], np.float32), np.asarray(saved["scale"], np.float32)
        extra = saved["width"] * len(EMA_TAUS.get(saved["kind"], ()))
        m.net = m._build(extra)
        m.net.load_state_dict(saved["state"])
        m.net.eval()
        return m


# ---------------------------------------------------------------------- data collection and closed-loop policy
from collections import deque  # noqa: E402

from apm.adapters.base import Emitter  # noqa: E402
from apm.adapters.wheeled_ctrl import WheeledCtrlAdapter  # noqa: E402
from apm_research.policies import AblatedAPM, Explore, FeaturePolicy  # noqa: E402
from apm_research.predictor import HORIZON_DECISIONS  # noqa: E402

N_APM = 8  # V, A, D, u (5)


class _APMReadout:
    """The APM v0.1 core run observe-only on the policy's observation stream (as in B2); readout = (V, A, D, u)."""

    def apm_reset(self) -> None:
        self.adapter = WheeledCtrlAdapter(Emitter("wheeled-ctrl"))
        self.apm = AblatedAPM(profile=self.adapter.profile())
        self.apm.ingest(self.adapter.profile(), now_s=0.0)

    def apm_observe(self, obs, t: float) -> None:
        for m in self.adapter.step(obs, t):
            self.apm.ingest(m)
        self.apm.step(t)

    def apm_readout(self) -> np.ndarray:
        a = self.apm.affect
        return np.concatenate([a.p, a.u]).astype(np.float32)


class ExploreRecorder(Explore, _APMReadout):
    """Exploration policy (F's randomised safe action variation) that records, per decision, the features, the APM
    readout and the action, plus the outcome counters needed for 2 s labels (ground truth only for labels)."""

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.apm_reset()
        self.rows: list[dict] = []

    def observe(self, obs, t: float) -> None:
        super().observe(obs, t)
        self.apm_observe(obs, t)

    def decide(self, t: float) -> int:
        a = super().decide(t)
        self.rows.append({"x": self.last_features.copy(), "apm": self.apm_readout(), "a": a, "t": t})
        return a


class LabelRecorder:
    """Harness recorder: outcome counters at each decision; labels = what happened in the next 2 s."""

    def __init__(self) -> None:
        self.counters: list[tuple] = []
        self.episode_of: list[int] = []
        self.ends: list[int] = []
        self.episode = 0

    def decision(self, world, policy, action, t) -> None:
        self.counters.append((len(world.log.collisions), len(world.log.slips), world.log.progress))
        self.episode_of.append(self.episode)

    def end_episode(self, world) -> None:
        self.ends.append(len(self.counters))
        self.episode += 1


def lifetime_arrays(policy: ExploreRecorder, rec: LabelRecorder) -> dict:
    T = len(policy.rows)
    c = np.array(rec.counters, float)
    y = np.zeros((T, 3), np.float32)
    mask = np.zeros(T, bool)
    start = 0
    for end in rec.ends:
        for i in range(start, end - HORIZON_DECISIONS):
            j = i + HORIZON_DECISIONS
            y[i] = (float(c[j, 0] > c[i, 0]), float(c[j, 1] > c[i, 1]), c[j, 2] - c[i, 2])
            mask[i] = True
        start = end
    return {"X": np.stack([r["x"] for r in policy.rows]).astype(np.float32),
            "APM": np.stack([r["apm"] for r in policy.rows]).astype(np.float32),
            "A": np.array([r["a"] for r in policy.rows]), "y_col": y[:, 0], "y_slip": y[:, 1], "y_prog": y[:, 2],
            "mask": mask, "episode": np.array(rec.episode_of)}


class StatefulRule(FeaturePolicy, _APMReadout):
    """Closed loop: argmax_a progress(a) − λc·P_col(a) − λs·P_slip(a) with a G3 stateful predictor (ensemble of
    seeds). The state is built online from this lifetime's decisions only; `control` reproduces the evaluation
    controls (reset / shuffle with a recorded history from another lifetime)."""

    def __init__(self, models: list[StatefulPredictor], lam_c: float, lam_s: float, control: str | None = None,
                 shuffle_history: np.ndarray | None = None, name: str = "S"):
        self.models, self.lam_c, self.lam_s, self.control, self.shuffle = models, lam_c, lam_s, control, \
            shuffle_history
        self.kind = models[0].kind
        self.name = name

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.k = 0
        self.z_log: list[np.ndarray] = []
        if self.kind == "pad":
            self.apm_reset()
        self.taus = EMA_TAUS.get(self.kind, ())
        self.ema = [[np.zeros(m.width, np.float32) for _ in self.taus] for m in self.models]
        self.win: deque = deque(maxlen=WINDOW)  # raw history (real or shuffled), excluding the current step

    def observe(self, obs, t: float) -> None:
        super().observe(obs, t)
        if self.kind == "pad":
            self.apm_observe(obs, t)

    def decide(self, t: float) -> int:
        x = self.features(t)
        z_raw = np.concatenate([x, self.apm_readout()]) if self.kind == "pad" else x
        h_raw = z_raw if self.shuffle is None else self.shuffle[min(self.k, len(self.shuffle) - 1)]
        self.z_log.append(z_raw)
        self.k += 1
        outs = []
        for i, m in enumerate(self.models):
            z = ((z_raw - m.mean) / m.scale).astype(np.float32)
            h = ((h_raw - m.mean) / m.scale).astype(np.float32)
            extra = np.zeros(0, np.float32)
            if self.taus:
                if self.control == "reset":
                    extra = np.zeros(m.width * len(self.taus), np.float32)
                else:
                    for j, tau in enumerate(self.taus):
                        self.ema[i][j] = self.ema[i][j] + (1 - np.exp(-DT_DECISION / tau)) * (h - self.ema[i][j])
                    extra = np.concatenate(self.ema[i])
            seq = None
            if self.kind in ("lds", "gru", "fastslow"):
                if self.control == "reset":
                    seq = z[None]
                else:
                    past = [((r - m.mean) / m.scale).astype(np.float32) for r in self.win][-(WINDOW - 1):]
                    seq = np.zeros((WINDOW, m.width), np.float32)
                    rows = past + [z]
                    seq[-len(rows):] = np.stack(rows)
            Z = np.repeat(z[None], N_ACTIONS, 0)
            E = np.repeat(extra[None], N_ACTIONS, 0)
            S = None if seq is None else np.repeat(seq[None], N_ACTIONS, 0)
            outs.append(m.predict(Z, E, S, np.arange(N_ACTIONS)))
        self.win.append(h_raw)
        o = np.mean(outs, axis=0)
        u = o[:, 2] - self.lam_c * o[:, 0] - self.lam_s * o[:, 1]
        self.last_utility = u
        return int(np.argmax(u))
