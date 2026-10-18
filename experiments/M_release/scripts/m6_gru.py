"""M6: closes the deferred question from L D. Does a small learned sequence model (GRU) predict encounter outcomes
better than APM's explicit Bayesian type memory?

    python experiments/M_release/scripts/m6_gru.py --robot nova_carter --runs <T dirs> --out results/m6/gru_<robot>.json

Data: the robot's partition-T exploration lifetimes (random intents, so outcomes are not shaped by a policy).
Lifetimes from the warehouse train and lifetimes from the home scene test (a scene shift).

Task: for each encounter with an identity, predict whether the person's lunge reaches the robot (contacts > 0), from
- that identity's past observations (IMU categories, camera codes, detector, band), and
- the current intent (proximity, tempo, hesitation; known before the encounter).
Simulator truth is never an input.

Predictors (all trained on warehouse lifetimes only):
- **MEM:** APM's fitted type memory (`M5_<robot>_types.json`, λ 0.02). The belief before the encounter, b · r and
  the intent go into a logistic regression.
- **GRU:** a 1-layer GRU (16 units) over the identity's observation sequence (one-hot categories as counts,
  detector, band), with the intent concatenated at the output. Adam for 200 epochs; 5 seeds, averaged.
- **BASE:** a logistic regression on the intent alone (no experience).

Metric: test log loss and AUC on home lifetimes, with a bootstrap CI over lifetimes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
CFG = REPO / "experiments" / "M_release" / "configs"
CAM_OFFSET = 24


def load(paths: list[Path], robot: str) -> list[dict]:
    out = []
    for p in paths:
        for f in sorted(p.rglob("part_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["robot"] == robot and r["partition"] == "T":
                    out.append(r)
    return out


def examples(lts: list[dict], edges: list[float], model_cfg: dict):
    """Per encounter: (sequence of past obs feature vectors of this identity, intent, belief feats, label, lt id)."""
    from apm_revise.belief import TypeModel
    from apm_transfer.nerva_memory import TypeStateModel
    model = TypeModel.from_dict(model_cfg["model"])
    camera_block = model.E.shape[1] > 24
    ex = []
    for li, lt in enumerate(lts):
        mem = TypeStateModel(model, 0.02, transition_each_meeting=True)
        hist: dict[str, list[np.ndarray]] = {}
        for r in lt["rows"]:
            ident, kind = r["entity"], r["entity"][0]
            b = mem.belief(ident, kind)
            intent = np.array([r["proximity"], r["tempo"], r["hesitation"], float(r["standoff"] > 0)])
            seq = hist.get(ident, [])
            ex.append({"seq": list(seq), "intent": intent, "mem": np.concatenate([[float(b @ model.r)], b]),
                       "y": float(r["contacts"] > 0), "lt": li})
            imu = [int(np.searchsorted(edges, v, side="right")) for v in r["imu_peaks"]] if r["standoff"] == 0 else []
            cats = [r["band"] * 8 + c for c in imu] + ([CAM_OFFSET + c for c in r["cam_codes"]] if camera_block else [])
            mem.observe_many(ident, kind, cats, float(r["detected"]))
            f = np.zeros(32 + 2)
            for c in cats:
                f[c] += 1
            f[32], f[33] = float(r["detected"]), r["band"] / 2.0
            hist.setdefault(ident, []).append(f)
    return ex


def logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 2000):
    w = np.zeros(X.shape[1] + 1)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= 0.5 * g
    return lambda Z: 1 / (1 + np.exp(-np.hstack([Z, np.ones((len(Z), 1))]) @ w))


def gru_predict(train, test, seed: int) -> np.ndarray:
    import torch
    torch.manual_seed(seed)
    D = len(train[0]["intent"])

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.g = torch.nn.GRU(34, 16, batch_first=True)
            self.o = torch.nn.Linear(16 + D, 1)

        def forward(self, seqs, intent):
            h = torch.zeros(len(seqs), 16)
            for i, s in enumerate(seqs):
                if len(s):
                    _, hn = self.g(torch.tensor(np.array(s), dtype=torch.float32)[None])
                    h[i] = hn[0, 0]
            return self.o(torch.cat([h, intent], 1)).squeeze(1)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    seqs = [e["seq"] for e in train]
    it = torch.tensor(np.array([e["intent"] for e in train]), dtype=torch.float32)
    y = torch.tensor([e["y"] for e in train], dtype=torch.float32)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(seqs, it), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net([e["seq"] for e in test],
                              torch.tensor(np.array([e["intent"] for e in test]), dtype=torch.float32)))
    return p.numpy()


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lts = load(a.runs, a.robot)
    mcfg = json.loads((CFG / f"M5_{a.robot}_types.json").read_text())
    edges = json.loads((CFG / f"M5_{a.robot}_edges.json").read_text())["acc8"]
    tr_lt = [lt for lt in lts if lt["scene"] == "warehouse"]
    te_lt = [lt for lt in lts if lt["scene"] == "home"]
    tr, te = examples(tr_lt, edges, mcfg), examples(te_lt, edges, mcfg)
    ytr, yte = np.array([e["y"] for e in tr]), np.array([e["y"] for e in te])
    base = logreg(np.array([e["intent"] for e in tr]), ytr)(np.array([e["intent"] for e in te]))
    memf = lambda E: np.array([np.r_[e["mem"], e["intent"]] for e in E])  # noqa: E731
    mem = logreg(memf(tr), ytr)(memf(te))
    gru = np.mean([gru_predict(tr, te, s) for s in range(5)], axis=0)
    lt_ids = np.array([e["lt"] for e in te])
    rng = np.random.default_rng(20261019)
    uniq = np.unique(lt_ids)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([np.flatnonzero(lt_ids == u) for u in pick])
        diffs.append(ll(gru[idx], yte[idx]) - ll(mem[idx], yte[idx]))
    res = {"robot": a.robot, "train_lifetimes": len(tr_lt), "test_lifetimes": len(te_lt),
           "train_encounters": len(tr), "test_encounters": len(te), "test_contact_rate": float(yte.mean()),
           "log_loss": {"BASE": ll(base, yte), "MEM": ll(mem, yte), "GRU": ll(gru, yte)},
           "auc": {"BASE": auc(base, yte), "MEM": auc(mem, yte), "GRU": auc(gru, yte)},
           "gru_minus_mem_logloss_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
