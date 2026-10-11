"""Compact recurrent outcome predictor (GRU + MLP head), the first learned model behind the stable AffectLink contract.

Any model exposing fit(X, y, X_val, y_val) / predict(X) and a `model_version` can replace it (a Transformer, an
active-inference model, …) without changing robots or AffectLink. Requires the optional `torch` dependency.
"""
from __future__ import annotations

import copy
import io
import time

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover
    raise ImportError("apm.learning.predictor needs the optional dependency: pip install -e '.[learning]'") from error


class _Net(nn.Module):
    def __init__(self, inputs: int, hidden: int, head: int):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


class GRUPredictor:
    model_version = "apm-gru-0.1"

    def __init__(self, inputs: int, hidden: int = 64, head: int = 32, lr: float = 2e-3, epochs: int = 30,
                 batch: int = 256, patience: int = 5, seed: int = 0, pos_weight: float | None = None):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = _Net(inputs, hidden, head)
        self.lr, self.epochs, self.batch, self.patience, self.pos_weight = lr, epochs, batch, patience, pos_weight
        self.history: list[dict] = []

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def predict(self, X) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 4096):
                out.append(torch.sigmoid(self.net(torch.as_tensor(X[i:i + 4096], dtype=torch.float32))).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, X, y, X_val=None, y_val=None, epochs: int | None = None, early_stopping: bool = True):
        epochs = epochs or self.epochs
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, stale = np.inf, copy.deepcopy(self.net.state_dict()), 0
        Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        for epoch in range(epochs):
            self.net.train()
            order = self.rng.permutation(len(X))
            start, total = time.monotonic(), 0.0
            for i in range(0, len(order), self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += float(loss) * len(idx)
            row = {"epoch": epoch, "train_bce": total / len(X), "seconds": time.monotonic() - start}
            if X_val is not None:
                p = self.predict(X_val)
                row["val_brier"] = float(np.mean((p - y_val) ** 2))
                if row["val_brier"] < best - 1e-6:
                    best, best_state, stale = row["val_brier"], copy.deepcopy(self.net.state_dict()), 0
                else:
                    stale += 1
            self.history.append(row)
            if early_stopping and X_val is not None and stale >= self.patience:
                break
        if early_stopping and X_val is not None:
            self.net.load_state_dict(best_state)
        return self

    def clone(self) -> GRUPredictor:
        return copy.deepcopy(self)

    def state_bytes(self) -> int:
        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)
        return buffer.tell()
