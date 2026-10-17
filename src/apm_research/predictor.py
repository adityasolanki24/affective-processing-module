"""Action-conditioned consequence predictor for B1/B2 (MASTER_PREREGISTRATION.md §4.2).

Input: decision features ⊕ one-hot(action). Outputs: P(collision within 2 s), P(slip within 2 s), expected goal
progress within 2 s (m). Labels come only from what actually happened after the action that was actually taken
(exploration data with randomised actions); nothing is labelled for an action not taken. Trained with torch,
served with a NumPy forward pass (no torch needed at decision time).
"""
from __future__ import annotations

import copy

import numpy as np

from apm_testbeds.wheeled_ctrl import N_ACTIONS

HORIZON_DECISIONS = 4  # 2 s at 0.5 s per decision


def _onehot(actions: np.ndarray) -> np.ndarray:
    out = np.zeros((len(actions), N_ACTIONS), dtype=np.float32)
    out[np.arange(len(actions)), actions] = 1.0
    return out


class ActionPredictor:
    def __init__(self, inputs: int, hidden: int = 64, seed: int = 0):
        self.inputs, self.hidden, self.seed = inputs, hidden, seed
        self.weights: list[tuple[np.ndarray, np.ndarray]] = []
        self.mean: np.ndarray = np.zeros(inputs, dtype=np.float32)
        self.scale: np.ndarray = np.ones(inputs, dtype=np.float32)
        self.history: list[dict] = []

    @property
    def parameters(self) -> int:
        return int(sum(w.size + b.size for w, b in self.weights))

    def fit(self, X, A, y_col, y_slip, y_prog, Xv, Av, yv_col, yv_slip, yv_prog, epochs: int = 60, lr: float = 1e-3,
            batch: int = 512, patience: int = 6):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mean = X.mean(0).astype(np.float32)
        self.scale = (X.std(0) + 1e-3).astype(np.float32)
        net = nn.Sequential(nn.Linear(self.inputs + N_ACTIONS, self.hidden), nn.ReLU(),
                            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))

        def tensors(X, A, c, s, p):
            Z = np.concatenate([(X - self.mean) / self.scale, _onehot(A)], 1)
            return [torch.as_tensor(v, dtype=torch.float32) for v in (Z, c, s, p)]

        Z, c, s, p = tensors(X, A, y_col, y_slip, y_prog)
        Zv, cv, sv, pv = tensors(Xv, Av, yv_col, yv_slip, yv_prog)
        bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        def loss(Z, c, s, p):
            out = net(Z)
            return bce(out[:, 0], c) + bce(out[:, 1], s) + mse(out[:, 2], p)

        best, best_state, stale = np.inf, copy.deepcopy(net.state_dict()), 0
        for epoch in range(epochs):
            net.train()
            order = rng.permutation(len(Z))
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                opt.zero_grad()
                loss(Z[idx], c[idx], s[idx], p[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss(Zv, cv, sv, pv))
            self.history.append({"epoch": epoch, "val_loss": v})
            if v < best - 1e-5:
                best, best_state, stale = v, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        net.load_state_dict(best_state)
        layers = [m for m in net if isinstance(m, nn.Linear)]
        self.weights = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in layers]
        return self

    def raw(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        h = np.concatenate([(X - self.mean) / self.scale, _onehot(A)], 1)
        for i, (w, b) in enumerate(self.weights):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return h

    def predict(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.raw(np.asarray(X, dtype=np.float32), np.asarray(A))
        sig = 1.0 / (1.0 + np.exp(-np.clip(out[:, :2], -30, 30)))
        return sig[:, 0], sig[:, 1], out[:, 2]

    def predict_actions(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For one feature vector: (P_col, P_slip, progress) for each of the actions."""
        X = np.repeat(np.asarray(x, dtype=np.float32)[None], N_ACTIONS, 0)
        return self.predict(X, np.arange(N_ACTIONS))

    def to_dict(self) -> dict:
        return {"inputs": self.inputs, "hidden": self.hidden, "seed": self.seed, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "weights": [(w.tolist(), b.tolist()) for w, b in self.weights]}

    @classmethod
    def from_dict(cls, d: dict) -> ActionPredictor:
        m = cls(d["inputs"], d["hidden"], d["seed"])
        m.mean, m.scale = np.asarray(d["mean"], np.float32), np.asarray(d["scale"], np.float32)
        m.weights = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in d["weights"]]
        return m


class EnsemblePredictor:
    """Mean of several seeds' predictions (the B1/B2 predictor; MASTER_PREREGISTRATION.md §4.2 trains 5 seeds)."""

    def __init__(self, members: list[ActionPredictor]):
        self.members = members

    def predict(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outs = [m.predict(X, A) for m in self.members]
        pc, ps, pr = (np.mean([o[i] for o in outs], axis=0) for i in range(3))
        return pc, ps, pr

    def predict_actions(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.repeat(np.asarray(x, dtype=np.float32)[None], N_ACTIONS, 0)
        return self.predict(X, np.arange(N_ACTIONS))

    def to_dict(self) -> dict:
        return {"members": [m.to_dict() for m in self.members]}

    @classmethod
    def from_dict(cls, d: dict) -> EnsemblePredictor:
        return cls([ActionPredictor.from_dict(m) for m in d["members"]])
