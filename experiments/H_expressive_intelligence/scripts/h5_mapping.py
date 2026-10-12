"""H5: can the expression mapping itself be learned? (prereg/H5_ADDENDUM.md)

Supervision: **reference styles from privileged simulator truth**, i.e. reference intent =
blend(relaxed, cautious, true hidden risk of the probe). It exists only in simulation and was not validated by
people. Systems:

| system | inputs | mapping |
|---|---|---|
| L0 | `M_id` (A0 memory) readout | fixed lookup (no learning) |
| L1 | memory readout (p, conf, g, kind-pool p, kind/zone one-hot, novelty) | MLP (11 → 32 → 8) |
| L2 | event history only | GRU (H) over events, head([h, query]) → 8-D intent |
| L3 | as L2 + memory-readout features of the query | GRU + MLP head |

Outputs are squashed into every channel's range (sigmoid). Loss: MSE to the reference intent, normalised by channel
range. Equal budget per learned family:
- 8 configurations on seed 0, selected by V loss;
- then a 5-seed ensemble (mean output).

L4 (fast/slow) runs only if H3 found S6 better than S5 (`--with-l4`). L5 (learned motion latent) is not run, for the
reasons in H1/X3.

    python scripts/h5_mapping.py --root results/H34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h34_states import PROBES, load_streams  # noqa: E402

from apm_express.develop import N_ENCOUNTERS, jolt_probability  # noqa: E402
from apm_express.intent import CHANNELS, RANGES, ExpressionIntent, blend  # noqa: E402
from apm_express.states import BASE, CAUTIOUS, RELAXED, EntityMemory, Probe, lookup_intent  # noqa: E402
from apm_express.states_learned import CODE, KINDS, Q_DIM, X_DIM, ZONES, event_vec, query_vec  # noqa: E402

LO = np.array([RANGES[c][0] for c in CHANNELS], np.float32)
HI = np.array([RANGES[c][1] for c in CHANNELS], np.float32)
F_DIM = 4 + len(KINDS) + len(ZONES) + 1
GRID = [{"hidden": h, "lr": lr, "epochs": e} for h in (16, 32) for lr in (1e-3, 3e-3) for e in (40, 80)]
THRESHOLD = 0.04


def reference(truth: float) -> np.ndarray:
    return blend(RELAXED, CAUTIOUS, truth).as_array().astype(np.float32)


def mem_features(mem: EntityMemory, probe: Probe, counts: dict) -> np.ndarray:
    r = mem.readout(probe)
    pool = mem.pools.get(probe.kind)
    f = np.zeros(F_DIM, np.float32)
    f[:4] = [r.p, 1.0 - r.u, r.g, pool.p(BASE)[0] if pool else BASE]
    f[4 + KINDS.index(probe.kind)] = 1.0
    f[4 + len(KINDS) + ZONES.index(probe.zone)] = 1.0
    f[-1] = 1.0 / (1.0 + counts.get(probe.identity, 0))
    return f


def lifetime_data(stream: dict, rng: np.random.Generator) -> dict:
    """Per lifetime: event vectors, and for every step k and probe: query, memory features, reference intent."""
    codes: dict = {}
    mem, counts = EntityMemory(), {}
    X, Q, F, Y, T = [], [], [], [], []
    names = list(PROBES)

    def code(i):
        return codes.setdefault(i, rng.choice([-1.0, 1.0], CODE).astype(np.float32))

    for k, ob in enumerate(stream["obs"]):
        mem.observe(ob)
        counts[ob.identity] = counts.get(ob.identity, 0) + 1
        X.append(event_vec(code(ob.identity), ob))
        q, f, y, tr = [], [], [], []
        for n in names:
            pr = PROBES[n]
            truth = (
                jolt_probability(stream["history"], pr.identity, pr.zone, min(k + 1, N_ENCOUNTERS - 1))
                if pr.kind != "D"
                else BASE
            )
            q.append(query_vec(code(pr.identity), pr.kind, pr.zone))
            f.append(mem_features(mem, pr, counts))
            y.append(reference(truth))
            tr.append(truth)
        Q.append(q)
        F.append(f)
        Y.append(y)
        T.append(tr)
    return {k: np.array(v, np.float32) for k, v in (("X", X), ("Q", Q), ("F", F), ("Y", Y), ("T", T))}


class Mapper(nn.Module):
    def __init__(self, kind: str, hidden: int):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        if kind != "L1":
            self.cell = nn.GRUCell(X_DIM, hidden)
        inp = {"L1": F_DIM, "L2": hidden + Q_DIM, "L3": hidden + Q_DIM + F_DIM}[kind]
        self.head = nn.Sequential(nn.Linear(inp, 32), nn.Tanh(), nn.Linear(32, len(CHANNELS)))

    def forward(self, d: dict) -> torch.Tensor:
        """Batched lifetimes -> normalised intents [B, K, P, C] in [0, 1]."""
        X, Q, F = (torch.tensor(d[k]) for k in ("X", "Q", "F"))
        B, K, P = Q.shape[0], Q.shape[1], Q.shape[2]
        if self.kind == "L1":
            return torch.sigmoid(self.head(F))
        h = torch.zeros(B, self.hidden)
        outs = []
        for k in range(K):
            h = self.cell(X[:, k], h)  # state after observing event k (the probe is read after it)
            hp = h[:, None, :].expand(B, P, self.hidden)
            parts = [hp, Q[:, k]] + ([F[:, k]] if self.kind == "L3" else [])
            outs.append(torch.sigmoid(self.head(torch.cat(parts, dim=-1))))
        return torch.stack(outs, dim=1)


def _batch(data: list[dict], idx) -> dict:
    return {k: np.stack([data[i][k] for i in idx]) for k in ("X", "Q", "F", "Y", "T")}


def loss_of(model, b) -> torch.Tensor:
    target = (torch.tensor(b["Y"]) - torch.tensor(LO)) / torch.tensor(HI - LO)
    return ((model(b) - target) ** 2).mean()


def train(kind, tr, va, seed, hidden, lr, epochs, batch=16):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Mapper(kind, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, state = float("inf"), None
    vb = _batch(va, range(len(va)))
    for _ in range(epochs):
        order = rng.permutation(len(tr))
        for i in range(0, len(order), batch):
            loss = loss_of(m, _batch(tr, order[i : i + batch]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            v = float(loss_of(m, vb))
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    if state is not None:
        m.load_state_dict(state)
    return m, best, int(sum(p.numel() for p in m.parameters()))


def eval_rows(name, models, xs, streams) -> list[dict]:
    rows = []
    for d, s in zip(xs, streams, strict=True):
        b = {k: v[None] for k, v in d.items()}
        with torch.no_grad():
            out = np.mean([m(b).numpy()[0] for m in models], axis=0) if models else None
        course = []
        for k in range(len(s["obs"])):
            if (k + 1) % 4 and k + 1 != len(s["obs"]):
                continue
            snap = {}
            for j, pn in enumerate(PROBES):
                if out is None:  # L0
                    mem = EntityMemory()
                    for ob in s["obs"][: k + 1]:
                        mem.observe(ob)
                    it = lookup_intent(mem.readout(PROBES[pn]))
                else:
                    it = ExpressionIntent.from_array(LO + (HI - LO) * out[k, j])
                truth = float(d["T"][k, j]) if PROBES[pn].kind != "D" else None
                snap[pn] = {"intent": it.as_dict(), "truth": truth}
            course.append({"k": k + 1, "probes": snap})
        rows.append(
            {
                "system": name,
                "control": "none",
                "history": s["history"],
                "index": s["index"],
                "course": course,
                "reset_intents": None,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--with-l4", action="store_true")
    a = ap.parse_args()
    from apm_express.provenance import confirmatory_lock

    lock = confirmatory_lock(HERE / "prereg" / "H5_ADDENDUM.md")
    rng = np.random.default_rng(0)
    T = load_streams(a.root / "T" / "experience.json", THRESHOLD)
    V = load_streams(a.root / "V" / "experience.json", THRESHOLD)
    X = load_streams(a.root / "X" / "experience.json", THRESHOLD)
    td, vd, xd = ([lifetime_data(s, rng) for s in S] for S in (T, V, X))
    summary, rows = {}, eval_rows("L0_fixed", [], xd, X)
    for kind in ("L1", "L2", "L3"):
        trials = []
        for cfg in GRID:
            _, v, n = train(kind, td, vd, 0, **cfg)
            trials.append({"cfg": cfg, "val": v, "params": n})
        best = min(trials, key=lambda r: r["val"])
        models = [train(kind, td, vd, s, **best["cfg"])[0] for s in range(a.seeds)]
        summary[kind] = {"trials": trials, "selected": best}
        rows += eval_rows(kind, models, xd, X)
        print(kind, best, flush=True)
    out = a.root / "X" / "h5_eval.json"
    out.write_text(json.dumps({"lock": lock, "training": summary, "rows": rows}), encoding="utf-8")


if __name__ == "__main__":
    main()
