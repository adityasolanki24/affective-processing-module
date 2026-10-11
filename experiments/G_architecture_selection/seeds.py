"""Frozen programme-G seed partitions (MASTER_PREREGISTRATION.md §3). The only source of G world seeds.

    world_seed(experiment, partition, index) -> numpy SeedSequence

A new programme entropy, so every G partition is independent of every F partition (F used 0xA9_0002_2026_1011);
in particular F's confirmatory partition X is never reproduced. Partitions are disjoint by construction.
"""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0003_2026_1012
EXPERIMENTS = {f"G{i}": i for i in range(9)}
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "S": 5}  # pilot, train, validation/tuning, confirmatory, shift
CONFIRMATORY = frozenset({"X", "S"})
F_PROGRAMME_ENTROPY = 0xA9_0002_2026_1011


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))


def fingerprint(ss: np.random.SeedSequence) -> str:
    return f"{int(ss.generate_state(1, np.uint64)[0]):016x}"
