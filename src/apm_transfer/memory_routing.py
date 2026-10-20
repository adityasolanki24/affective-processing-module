"""Memory-routing controls for the identity-specific memory test (remediation protocol, "Corrected comparisons").

Every mode uses the same runtime, observations and supervisor; only which memory slot is read and written changes:

| mode | decide reads | observe writes | tests |
|---|---|---|---|
| learn | the person's own memory | the person's own memory | APM (DT) |
| pooled | one shared slot | one shared slot | population-wide caution, no identity |
| swapped | another person's memory (A1 <-> B1) | the person's own memory | right memory vs wrong memory |
| shuffled | the person's own memory | a random person's memory | programme M's A2 control |

`swapped` must differ between reading and writing: a consistent swap would just rename the slots and behave
exactly like `learn`.
"""
from __future__ import annotations

import numpy as np

POOLED = "ANY"
SWAP = {"A1": "B1", "B1": "A1", "C1": "C1"}


def decide_identity(mode: str, identity: str) -> str:
    if mode == "pooled":
        return POOLED
    if mode == "swapped":
        return SWAP.get(identity, identity)
    return identity


def observe_identity(mode: str, identity: str, entities, rng: np.random.Generator | None) -> str:
    if mode == "pooled":
        return POOLED
    if mode == "shuffled":
        if rng is None:
            raise ValueError("shuffled routing needs its random stream")
        return str(entities[int(rng.integers(len(entities)))][0])
    return identity
