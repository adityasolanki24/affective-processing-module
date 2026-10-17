"""Two-level (lifetimes × training seeds) paired bootstrap used by E4."""
import numpy as np

from apm_research.stats import paired_lifetimes, paired_lifetimes_two_level


def _system(rng, n, rate):
    minutes = np.full(n, 5.0)
    return {"collisions": rng.poisson(rate * minutes).astype(float), "minutes": minutes,
            "goals_completed": rng.integers(20, 30, n).astype(float), "goals_issued": np.full(n, 30.0),
            "idle": rng.integers(0, 10, n).astype(float), "periods": np.full(n, 600.0)}


def test_one_seed_point_estimates_match_single_level():
    rng = np.random.default_rng(1)
    a, b = _system(rng, 50, 0.5), _system(rng, 50, 0.6)
    one = paired_lifetimes(a, b, n_boot=500)
    two = paired_lifetimes_two_level({k: v[None] for k, v in a.items()}, {k: v[None] for k, v in b.items()},
                                     n_boot=500)
    for k in ("collision_rate_relative", "completion_difference", "idle_difference"):
        assert np.isclose(one[k]["estimate"], two[k]["estimate"])
        assert np.allclose(one[k]["ci95"], two[k]["ci95"])  # same lifetime weights from the same seed


def test_seed_variance_widens_interval():
    rng = np.random.default_rng(2)
    b = _system(rng, 80, 0.6)
    base = _system(rng, 80, 0.5)
    seeds = {k: np.stack([v * (s if k == "collisions" else 1.0) for s in (0.5, 1.0, 1.5)]) for k, v in base.items()}
    same = {k: np.stack([v] * 3) for k, v in base.items()}
    wide = paired_lifetimes_two_level(seeds, {k: v[None] for k, v in b.items()}, n_boot=2000)
    narrow = paired_lifetimes_two_level(same, {k: v[None] for k, v in b.items()}, n_boot=2000)
    assert wide["collision_rate_relative"]["se"] > narrow["collision_rate_relative"]["se"]
