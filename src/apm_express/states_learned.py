"""Learned recurrent internal states for H3 (S5–S7). Needs torch (optional dependency `learning`).

Training objective: **world learning only**. Predict the observed outcome y_k of encounter k from the history before
it and the query (identity, kind, zone) of encounter k, with binary cross-entropy. No expression labels exist or are
used; expression is the fixed lookup L0 applied to the prediction (`apm_express.states.lookup_intent`).

**Identity encoding.** Each identity gets a random ±1 code (8 dims), drawn per lifetime. The network must therefore
bind outcomes to identities *within* a lifetime, and a never-seen identity simply arrives with a new code. Kinds
(A, B, C, D) and zones are one-hot.

| model | structure |
|---|---|
| S5 `gru` | GRU(16 → H) over events; readout MLP([h, query]) |
| S6 `fastslow` | two GRUs (H/2 each), the slow one leaky-integrated (λ = 0.1 per event); readout from both + query |
| S7 `gru_mem` | S5 + memory features for the query: identity (p, confidence), kind-pool p |

An ensemble of seeds gives p̂ as the ensemble mean and u = clip(4 · ensemble SD); g = BASE (no global term).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from apm_express.states import BASE, EntityMemory, Obs, Probe, Readout, StateModel

KINDS = ("A", "B", "C", "D")
ZONES = ("plain", "marked")
CODE = 8
X_DIM = CODE + len(KINDS) + len(ZONES) + 2  # + y + magnitude
Q_DIM = CODE + len(KINDS) + len(ZONES)
MEM_DIM = 3


def query_vec(code: np.ndarray, kind: str, zone: str) -> np.ndarray:
    v = np.zeros(Q_DIM, dtype=np.float32)
    v[:CODE] = code
    v[CODE + KINDS.index(kind)] = 1.0
    v[CODE + len(KINDS) + ZONES.index(zone)] = 1.0
    return v


def event_vec(code: np.ndarray, ob: Obs) -> np.ndarray:
    return np.concatenate([query_vec(code, ob.kind, ob.zone), [ob.y, min(ob.magnitude, 3.0)]]).astype(np.float32)


class Net(nn.Module):
    def __init__(self, kind: str, hidden: int = 32):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        if kind == "fastslow":
            self.fast = nn.GRUCell(X_DIM, hidden // 2)
            self.slow = nn.GRUCell(X_DIM, hidden // 2)
        else:
            self.cell = nn.GRUCell(X_DIM, hidden)
        extra = MEM_DIM if kind == "gru_mem" else 0
        self.head = nn.Sequential(nn.Linear(hidden + Q_DIM + extra, 32), nn.Tanh(), nn.Linear(32, 1))
        self.lam = 0.1

    def init(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.hidden)

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fastslow":
            half = self.hidden // 2
            hf, hs = h[:, :half], h[:, half:]
            hf = self.fast(x, hf)
            hs = (1 - self.lam) * hs + self.lam * self.slow(x, hs)
            return torch.cat([hf, hs], dim=1)
        return self.cell(x, h)

    def logit(self, h: torch.Tensor, q: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        parts = [h, q] + ([mem] if self.kind == "gru_mem" and mem is not None else [])
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def lifetime_tensors(events: list[Obs], rng: np.random.Generator) -> dict:
    """Per-lifetime training tensors: X[k] event k, Q[k] query of event k, Y[k], M[k] memory features before k."""
    codes: dict[str, np.ndarray] = {}
    mem = EntityMemory()
    X, Q, Y, M = [], [], [], []
    for ob in events:
        code = codes.setdefault(ob.identity, rng.choice([-1.0, 1.0], CODE).astype(np.float32))
        r = mem.readout(Probe(ob.identity, ob.kind, ob.zone))
        pool = mem.pools.get(ob.kind)
        M.append([r.p, 1.0 - r.u, pool.p(BASE)[0] if pool else BASE])
        Q.append(query_vec(code, ob.kind, ob.zone))
        X.append(event_vec(code, ob))
        Y.append(ob.y)
        mem.observe(ob)
    f = np.float32
    return {"X": np.array(X, f), "Q": np.array(Q, f), "Y": np.array(Y, f), "M": np.array(M, f)}


def batch_loss(net: Net, batch: list[dict]) -> torch.Tensor:
    X = torch.tensor(np.stack([b["X"] for b in batch]))
    Q = torch.tensor(np.stack([b["Q"] for b in batch]))
    Y = torch.tensor(np.stack([b["Y"] for b in batch]))
    M = torch.tensor(np.stack([b["M"] for b in batch]))
    h = net.init(len(batch))
    losses = []
    for k in range(X.shape[1]):
        logit = net.logit(h, Q[:, k], M[:, k])
        losses.append(nn.functional.binary_cross_entropy_with_logits(logit, Y[:, k], reduction="none"))
        h = net.step(h, X[:, k])
    return torch.stack(losses, dim=1).mean()


def train(kind: str, train_sets: list[dict], val_sets: list[dict], seed: int, hidden: int = 32, lr: float = 3e-3,
          epochs: int = 60, batch: int = 16) -> tuple[Net, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(kind, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, curve = float("inf"), None, []
    for _ in range(epochs):
        order = rng.permutation(len(train_sets))
        net.train()
        for i in range(0, len(order), batch):
            loss = batch_loss(net, [train_sets[j] for j in order[i:i + batch]])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(batch_loss(net, val_sets))
        curve.append(v)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"val_loss": best, "curve": curve, "params": n_params(net)}


class LearnedState(StateModel):
    """Online inference with an ensemble of trained nets (one per seed)."""

    def __init__(self, nets: list[Net], name: str, code_seed: int = 0):
        self.nets, self.name, self.code_seed = nets, name, code_seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.code_seed)
        self.codes: dict[str, np.ndarray] = {}
        self.mem = EntityMemory()
        self.h = [n.init(1) for n in self.nets]

    def _code(self, identity: str) -> np.ndarray:
        return self.codes.setdefault(identity, self.rng.choice([-1.0, 1.0], CODE).astype(np.float32))

    def observe(self, ob: Obs) -> None:
        x = torch.tensor(event_vec(self._code(ob.identity), ob))[None]
        with torch.no_grad():
            self.h = [n.step(h, x) for n, h in zip(self.nets, self.h, strict=True)]
        self.mem.observe(ob)

    def readout(self, probe: Probe) -> Readout:
        q = torch.tensor(query_vec(self._code(probe.identity), probe.kind, probe.zone))[None]
        r = self.mem.readout(probe)
        pool = self.mem.pools.get(probe.kind)
        m = torch.tensor([[r.p, 1.0 - r.u, pool.p(BASE)[0] if pool else BASE]], dtype=torch.float32)
        with torch.no_grad():
            ps = np.array([float(torch.sigmoid(n.logit(h, q, m))) for n, h in zip(self.nets, self.h, strict=True)])
        return Readout(float(ps.mean()), float(np.clip(4.0 * ps.std(), 0.0, 1.0)), BASE)
