"""Frozen v0.2 seed partitions (MASTER_PREREGISTRATION.md §3). The only source of world seeds for E0–E7.

    world_seed(experiment, partition, index) -> numpy SeedSequence

Partitions are disjoint by construction (distinct spawn keys under one programme entropy). v0.1 integer seeds feed
``default_rng(int)`` in the old testbed only, a different generator path, and are never produced here.
"""
from __future__ import annotations

import numpy as np

PROGRAMME_ENTROPY = 0xA9_0002_2026_1011
EXPERIMENTS = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5, "E6": 6, "E7": 7}
PARTITIONS = {"P": 1, "T": 2, "V": 3, "X": 4, "S": 5}  # pilot, train, validation, confirmatory, shift
CONFIRMATORY = frozenset({"X", "S"})
# Seeds already used by v0.1 work (AUDIT.md); listed so tests can prove they are never reproduced.
V01_WHEELED_INT_SEEDS = (*range(0, 12), *range(10_000, 10_400), 10_999, *range(50_000, 50_020), *range(70_000, 70_004))
NERVA_V02_MASTER_SEED, NERVA_V02_SIM_SEED_OFFSET = 2027, 100_000


def world_seed(experiment: str, partition: str, index: int) -> np.random.SeedSequence:
    if experiment not in EXPERIMENTS or partition not in PARTITIONS or index < 0:
        raise ValueError(f"unknown seed request {experiment}/{partition}/{index}")
    return np.random.SeedSequence(PROGRAMME_ENTROPY, spawn_key=(EXPERIMENTS[experiment], PARTITIONS[partition], index))


def seeds(experiment: str, partition: str, n: int, start: int = 0) -> list[np.random.SeedSequence]:
    return [world_seed(experiment, partition, i) for i in range(start, start + n)]


def fingerprint(ss: np.random.SeedSequence) -> str:
    """Stable short id of a seed (first state word), for manifests."""
    return f"{int(ss.generate_state(1, np.uint64)[0]):016x}"
