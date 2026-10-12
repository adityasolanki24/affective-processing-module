# H5 report: can the expression mapping itself be learned? (2026-10-13)

**Setup:**
- Governed by `prereg/H5_ADDENDUM.md`;
- X streams of H3/H4 (210 lifetimes);
- training on T, selection on V;
- results in `results/H34/X/h5_results.json`.

**Supervision.** Reference intents computed from **privileged simulator truth**:
blend(relaxed, cautious, true hidden risk). These are designer references, not human judgements.

## Results (X, end-of-lifetime probes)

| mapping | params | CE (lower is better) | E1 | E2 history effect | E4 revision median (fraction) | E6 kind generalisation |
|---|---|---|---|---|---|---|
| L0 fixed lookup on A0 memory | 0 learned | 0.167 | 0.70 | **0.84** | 20 (0.63) | **0.78** |
| L1 MLP on memory readout | 648 | 0.146 | 0.72 | 0.59 | 16 (0.67) | 0.59 |
| L2 GRU, end to end | 6,568 | 0.126 | **0.77** | 0.49 | **8 (1.00)** | 0.48 |
| L3 GRU + memory features | 6,920 | **0.124** | 0.75 | 0.57 | 8 (1.00) | 0.53 |

CE differences vs L0 (95% bootstrap CI):

| mapping | CE difference |
|---|---|
| L1 | [−0.029, −0.014] |
| L2 | [−0.058, −0.026] |
| L3 | [−0.055, −0.031] |

**Preregistered verdicts:**
- **All three learned mappings "beat" the fixed lookup:** CE lower, E1 not lower, revision not slower.
- **Preferred (simplest within +0.02 of the best CE): L2, the ordinary GRU.** L3 adds explicit memory inputs for no
  meaningful gain. L4 (fast/slow) was not run: H3 found S6 within the non-inferiority margin of simpler systems, not
  better.

## Reading

- **Learning helps calibration and revision.**
  - The learned mappings correct the lookup's systematic over- and under-statement. For example, A0 memory's Beta
    prior plus the lookup overshoots high-risk entities.
  - The GRUs revise immediately when the world changes.
- **It costs specificity.**
  - Trained to minimise squared error against references averaged over many lifetimes, the learned mappings
    regress toward the mean.
  - The difference between two robots with different histories shrinks from 0.84 to 0.49–0.59.
  - Kind generalisation shrinks similarly.
- **Calibrated and expressive are not the same goal.** A robot whose expression is calibrated to the true risk
  differs *less* between histories than one that commits to what it has seen.
- **Which one people find more meaningful is unknown without H2.**
- **The supervision is privileged truth that only exists in simulation.** A deployed robot cannot be trained this way
  from its own experience. Learning the mapping from human judgements (RLHF-style) would be the legitimate route; it
  needs H2's approved study.

## Consequence for the architecture

The learned mapping is not adopted on these grounds alone. H8 tests "A0 + learned mapping" (L1, which adds no
developmental latent) against the A0 baseline on fresh data under its own rule.
