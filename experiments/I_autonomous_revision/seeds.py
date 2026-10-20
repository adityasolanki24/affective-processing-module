"""Frozen programme-I seed partitions. The only source of I world seeds.

    world_seed(experiment, partition, index) -> numpy SeedSequence

New programme entropy, so every I partition is independent of every F, G and H partition. Partitions are disjoint by
construction (different spawn keys).
"""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0005_2026_1017
EXPERIMENTS = {f"I{i}": i for i in range(10)}
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "S": 5}  # pilot, train, validation/tuning, locked test, shift
CONFIRMATORY = frozenset({"X", "S"})
EARLIER_PROGRAMME_ENTROPIES = (0xA9_0002_2026_1011, 0xA9_0003_2026_1012, 0xA9_0004_2026_1012)


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))
