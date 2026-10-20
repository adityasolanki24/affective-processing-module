"""Frozen programme-J seed partitions. The only source of J world seeds.

    world_seed(experiment, partition, index) -> numpy SeedSequence

New programme entropy, so every J partition is independent of every F, G, H and I partition.
"""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0006_2026_1017
EXPERIMENTS = {f"J{i}": i for i in range(10)}
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "S": 5}  # pilot, train, validation/calibration, locked test, shift
CONFIRMATORY = frozenset({"X", "S"})
EARLIER_PROGRAMME_ENTROPIES = (0xA9_0002_2026_1011, 0xA9_0003_2026_1012, 0xA9_0004_2026_1012, 0xA9_0005_2026_1017)


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))
