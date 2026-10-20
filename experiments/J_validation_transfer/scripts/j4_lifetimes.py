"""J4: paired NERVA lifetimes from a bank of physically simulated standing encounters (MASTER_PREREGISTRATION §4.2).

    python j4_lifetimes.py --bank <X_bank.jsonl> --rows 300:1000 --types-model <json> --edges <json> --lam L
        --n 200 --out <jsonl>

**Pairing.** Lifetime i uses the same seed in every history and for every memory. The identity and zone sequence and
the bank draws are therefore shared; only the entities' physical types (the history) differ.

**Evidence per encounter.** A bank row of the entity's current type, standing at the neutral 0.4 m, measured by
NERVA's own sensors:
- y = `detected` (self-calibrated detector);
- magnitude = peak acceleration deviation;
- category = acc8 code (edges frozen from J3-T).

**Memories:**
- `H_A0`: H's `EntityMemory` (kind pools, Beta on y);
- `count`: a per-identity Beta on y, without pools (simple statistical baseline);
- `type`: `TypeStateModel`, NERVA-fitted, no pooling;
- `type_kind`: as `type`, with kind pooling;
- `gru` (with `--gru`): the recurrent baseline `nerva_memory.GruMemory`, trained on V by `j4_gru.py`.

**Probes** every 6 encounters: identities B1 and A1 (plain zone), B1 (marked zone), and the novel B3 (plain). Each
gives (p, u, g) and the L0 caution.

**Restart.** `type_kind` is also run with its state serialised to JSON and restored at encounter 24
(`type_kind_restart`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from apm_express.states import BASE, EntityMemory, Probe, StateModel, caution  # noqa: E402
from apm_revise.belief import TypeModel  # noqa: E402
from apm_select.memory import Cumulative  # noqa: E402
from apm_transfer.nerva_memory import (  # noqa: E402
    HISTORIES,
    N_ENCOUNTERS,
    GruMemory,
    ObsX,
    TypeStateModel,
    lifetime_types,
)

PROBES = (("B1", "B", "plain"), ("A1", "A", "plain"), ("B1", "B", "marked"), ("B3", "B", "plain"))
EVERY = 6
RESTART_AT = 24


class CountMemory(StateModel):
    name = "count"

    def reset(self) -> None:
        self.ids: dict = {}
        self.g = BASE

    def __init__(self):
        self.reset()

    def observe(self, ob) -> None:
        self.ids.setdefault(ob.identity, Cumulative()).update(ob.y, 1.0, BASE)
        self.g += 0.15 * (ob.y - self.g)

    def readout(self, probe):
        from apm_express.states import Readout
        b = self.ids.get(probe.identity)
        if b is None:
            return Readout(BASE, 1.0, self.g)
        p, conf = b.p(BASE)
        return Readout(float(p), float(1 - conf), self.g)


def load_bank(path: Path, rows: tuple[int, int]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.setdefault(r["type"], []).append(r)
    for t in out:
        out[t] = sorted(out[t], key=lambda r: r["index"])[rows[0]:rows[1]]
    return out


def code(ev: list, edges: list[float]) -> int:
    return int(np.searchsorted(edges, ev[1], side="right"))


def run_lifetime(i: int, history: str, bank: dict, edges: list[float], model: TypeModel, lam: float,
                 seed: int, gru: dict | None = None) -> dict:
    rng = np.random.default_rng([seed, i])
    seq = lifetime_types(history, np.random.default_rng([seed, i, 1]))
    mems: dict[str, StateModel] = {"H_A0": EntityMemory(), "count": CountMemory(),
                                   "type": TypeStateModel(model, lam), "type_kind": TypeStateModel(model, lam, True),
                                   "type_kind_restart": TypeStateModel(model, lam, True)}
    if gru is not None:
        mems["gru"] = GruMemory(gru)
    draws = rng.random(N_ENCOUNTERS)
    probes: dict[str, list] = {m: [] for m in mems}
    obs_log = []
    for k, (ident, zone, etype) in enumerate(seq):
        rows = bank[etype]
        r = rows[int(draws[k] * len(rows))]
        ev = r["evidence"]
        ob = ObsX(ident, ident[0], zone, float(bool(ev[9])), float(ev[1]), code(ev, edges))
        obs_log.append([ident, zone, etype, int(ob.y), ob.category])
        if k == RESTART_AT:
            m = mems["type_kind_restart"]
            assert isinstance(m, TypeStateModel)
            fresh = TypeStateModel(model, lam, True)
            fresh.load_state(json.loads(json.dumps(m.state())))
            mems["type_kind_restart"] = fresh
        for m in mems.values():
            m.observe(ob)
        if (k + 1) % EVERY == 0:
            for name, m in mems.items():
                snap = []
                for pid, kind, pz in PROBES:
                    rd = m.readout(Probe(pid, kind, pz))
                    snap.append([round(rd.p, 5), round(rd.u, 4), round(rd.g, 4), round(caution(rd), 5)])
                probes[name].append(snap)
    return {"history": history, "index": i, "probes": probes, "obs": obs_log}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--rows", default="300:1000")
    ap.add_argument("--types-model", type=Path, required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--histories", default=",".join(HISTORIES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gru", type=Path, default=None, help="recurrent baseline weights (j4_gru.py)")
    ap.add_argument("--addendum", default="", help="confirmatory run: committed addendum (clean tree required)")
    a = ap.parse_args()
    if a.addendum:
        from apm_transfer.provenance import confirmatory_lock, write_json
        write_json(a.out.with_suffix(".meta.json"), {"provenance": confirmatory_lock(HERE / "prereg" / a.addendum),
                                                     "args": {k: str(v) for k, v in vars(a).items()}})
    lo, hi = (int(x) for x in a.rows.split(":"))
    bank = load_bank(a.bank, (lo, hi))
    model = TypeModel.from_dict(json.loads(a.types_model.read_text()))
    edges = json.loads(a.edges.read_text())["acc8"]
    gru = json.loads(a.gru.read_text()) if a.gru else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for h in a.histories.split(","):
            for i in range(a.first, a.first + a.n):
                fh.write(json.dumps(run_lifetime(i, h, bank, edges, model, a.lam, a.seed, gru)) + "\n")
    print("done", a.out)


if __name__ == "__main__":
    main()
