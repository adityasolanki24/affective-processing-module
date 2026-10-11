"""Train M6 (learned recurrent per-identity memory) on G2-T encounter logs (prereg/G2_ADDENDUM.md).

Data: `g2_run.py --partition T` logs (logging policy: the G1-selected regulator with M0 + exposed evidence, all
scenarios). Per lifetime, the observable trials (collision / exposed encounters with a reliable identity) are
replayed in time order exactly as the memory sees them: the pool prior is the running exposed-evidence pool mean
(BeliefMemory's rule), each identity has its own GRU state (input [y, w]), and the loss is the binary cross-entropy of
the prediction made *before* each trial. 80/20 split of lifetimes (seeded) for early stopping; hidden size 8.
Weights are exported in BeliefMemory's `Neural` layout (torch GRUCell gate order r, z, n).

    python scripts/g2_m6_train.py --logs runs/G2/T/lifetimes.jsonl --out results/G2/m6_weights.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
HIDDEN, EPOCHS, LR, SEED, PATIENCE = 8, 60, 3e-3, 0, 8
BASE, STRENGTH = 0.05, 2.0


def sequences(path: Path) -> list[list[tuple[str, float, float]]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        seq = [(ident, 1.0 if klass == "collision" else 0.0, float(conf))
               for (_, ident, conf, klass, reliable, _, _, _) in r["encounters"]
               if klass in ("collision", "exposed") and reliable]
        if seq:
            out.append(seq)
    return out


class M6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = torch.nn.GRUCell(2, HIDDEN)
        self.out = torch.nn.Linear(HIDDEN, 1)

    def lifetime_loss(self, seq) -> tuple[torch.Tensor, int]:
        h: dict[str, torch.Tensor] = {}
        n_pool = k_pool = 0.0
        losses = []
        for ident, y, w in seq:
            prior = (k_pool + STRENGTH * BASE) / (n_pool + STRENGTH)
            hi = h.get(ident, torch.zeros(HIDDEN))
            logit = self.out(hi)[0] + math.log(prior / (1 - prior))
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y)))
            h[ident] = self.cell(torch.tensor([[y, w]], dtype=torch.float32), hi[None])[0]
            n_pool, k_pool = n_pool + 1.0, k_pool + y
        return torch.stack(losses).sum(), len(losses)


def export(model: M6) -> dict:
    W_ih, W_hh = model.cell.weight_ih.detach().numpy(), model.cell.weight_hh.detach().numpy()
    b_ih, b_hh = model.cell.bias_ih.detach().numpy(), model.cell.bias_hh.detach().numpy()
    H = HIDDEN
    w = {}
    for i, g in enumerate("rzn"):
        w[f"W_i{g}"], w[f"W_h{g}"] = W_ih[i * H:(i + 1) * H].tolist(), W_hh[i * H:(i + 1) * H].tolist()
        w[f"b_i{g}"], w[f"b_h{g}"] = b_ih[i * H:(i + 1) * H].tolist(), b_hh[i * H:(i + 1) * H].tolist()
    w["w_out"] = model.out.weight.detach().numpy()[0].tolist()
    w["b_out"] = model.out.bias.detach().numpy().tolist()
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "G2" / "m6_weights.json")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    seqs = sequences(args.logs)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(seqs))
    cut = int(0.8 * len(seqs))
    train, val = [seqs[i] for i in order[:cut]], [seqs[i] for i in order[cut:]]
    model = M6()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def evaluate(m: M6, data) -> float:
        with torch.no_grad():
            tot, n = 0.0, 0
            for s in data:
                loss, k = m.lifetime_loss(s)
                tot, n = tot + float(loss), n + k
        return tot / max(1, n)
    best, wait, curve = evaluate(model, val), 0, []
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(EPOCHS):
        for i in rng.permutation(len(train)):
            opt.zero_grad()
            loss, k = model.lifetime_loss(train[i])
            (loss / k).backward()
            opt.step()
        v = evaluate(model, val)
        curve.append({"epoch": epoch, "val_bce": v})
        if v < best - 1e-5:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    prior_only = M6()  # all-zero weights: predicts the pool prior for everyone (reference)
    with torch.no_grad():
        for p in prior_only.parameters():
            p.zero_()
    out = {"weights": export(model), "hidden": HIDDEN, "train_lifetimes": len(train), "val_lifetimes": len(val),
           "trials": int(sum(len(s) for s in seqs)), "val_bce": evaluate(model, val),
           "val_bce_pool_prior_only": evaluate(prior_only, val), "curve": curve}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("trials", "val_bce", "val_bce_pool_prior_only", "train_lifetimes")}))


if __name__ == "__main__":
    main()
