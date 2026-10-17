"""J4 recurrent baseline: train `nerva_memory.GruMemory` self-supervised on V lifetimes (bank rows 0–299).

    python j4_gru.py --runs <V lifetimes.jsonl> --out configs/J4_gru.json

For each V lifetime and identity, the sequence is that identity's own meetings in order (from the lifetime's
observation log: acc8 category and detected; the entity's type is never read). At each meeting the GRU has seen the
earlier meetings and predicts this meeting's detection (binary cross-entropy). The first prediction uses h = 0, so the
output bias is the learned prior for an unmet identity.

Hidden 8; Adam 3e-3; 40 epochs; lifetimes split 80/20 (seed 0); the epoch with the best validation log loss is kept.
Weights go out in `expression_map.GruMapping`'s layout (out_w has one row).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm_transfer.expression_map import gru_input  # noqa: E402


def sequences(path: Path) -> tuple[list, list]:
    rng = np.random.default_rng(0)
    tr, va = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        lt = json.loads(line)
        per: dict = defaultdict(list)
        for ident, _zone, _etype, y, cat in lt["obs"]:
            per[ident].append((cat % 8, y))
        dest = va if rng.random() < 0.2 else tr
        for meets in per.values():
            x = np.stack([np.zeros(13)] + [gru_input(c, 1, bool(y)) for c, y in meets[:-1]])
            dest.append((x, np.array([y for _, y in meets], float)))
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tr, va = sequences(a.runs)
    import torch
    torch.manual_seed(0)
    H = 8
    cell, out = torch.nn.GRUCell(13, H), torch.nn.Linear(H, 1)
    opt = torch.optim.Adam(list(cell.parameters()) + list(out.parameters()), lr=3e-3)

    def loss_of(x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        h = torch.zeros(1, H)
        logits = []
        for t in range(len(y)):
            if t > 0:  # x[0] is the zero vector: the first prediction is from h = 0
                h = cell(torch.tensor(x[t:t + 1], dtype=torch.float32), h)
            logits.append(out(h)[0, 0])
        target = torch.tensor(y, dtype=torch.float32)
        return torch.nn.functional.binary_cross_entropy_with_logits(torch.stack(logits), target, reduction="sum")
    rng = np.random.default_rng(0)
    n_va = sum(len(y) for _, y in va)
    best: tuple[float, dict | None] = (np.inf, None)
    for _ep in range(40):
        for i in rng.permutation(len(tr)):
            opt.zero_grad()
            loss_of(*tr[i]).backward()
            opt.step()
        with torch.no_grad():
            v = float(sum(loss_of(x, y).item() for x, y in va)) / n_va
        if v < best[0]:
            best = (v, {"weight_ih": cell.weight_ih.detach().numpy().tolist(),
                        "weight_hh": cell.weight_hh.detach().numpy().tolist(),
                        "bias_ih": cell.bias_ih.detach().numpy().tolist(),
                        "bias_hh": cell.bias_hh.detach().numpy().tolist(),
                        "out_w": out.weight.detach().numpy().tolist(), "out_b": out.bias.detach().numpy().tolist()})
    assert best[1] is not None
    rate = float(np.mean(np.concatenate([y for _, y in tr])))
    base = float(-np.mean([y * np.log(rate) + (1 - y) * np.log(1 - rate) for _, ys in va for y in ys]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({**best[1], "val_logloss": best[0], "constant_rate_logloss": base, "hidden": H,
                                 "train_sequences": len(tr), "val_sequences": len(va)}), encoding="utf-8")
    print("val log loss", round(best[0], 4), "constant-rate baseline", round(base, 4))


if __name__ == "__main__":
    main()
