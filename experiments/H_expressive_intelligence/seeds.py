"""Frozen programme-H seed partitions (MASTER_PREREGISTRATION.md §4). The only source of H world seeds.

    world_seed(experiment, partition, index) -> numpy SeedSequence
    sim_seed(experiment, partition, index)   -> int (a 31-bit seed for NERVA's OpenDuckSim, derived from the above)

A new programme entropy, so every H partition is independent of every F and G partition (F 0xA9_0002_2026_1011,
G 0xA9_0003_2026_1012). Partitions are disjoint by construction (different spawn keys).
"""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0004_2026_1012
EXPERIMENTS = {f"H{i}": i for i in range(9)}
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "S": 5}  # pilot, train, validation/tuning, locked test, shift
CONFIRMATORY = frozenset({"X", "S"})
EARLIER_PROGRAMME_ENTROPIES = (0xA9_0002_2026_1011, 0xA9_0003_2026_1012)


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))


def sim_seed(experiment: str, partition: str, index: int) -> int:
    """Integer seed for simulators that take an int (NERVA's OpenDuckSim)."""
    return int(world_seed(experiment, partition, index).generate_state(1, np.uint32)[0] & 0x7FFF_FFFF)


def fingerprint(ss: np.random.SeedSequence) -> str:
    return f"{int(ss.generate_state(1, np.uint64)[0]):016x}"
