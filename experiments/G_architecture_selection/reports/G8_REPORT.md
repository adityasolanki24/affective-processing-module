# G8 — Minimising A0 (2026-10-12)

Preregistered in `prereg/G8_ADDENDUM.md`. Every variant retuned with the same budget (K = 48 λ draws on G8-V).
Stage 1 on G8-X (280 lifetimes × 10 episodes; commit `99318e9`, lock passed; `results/G8/g8_X.json`); stage 2 on
G8-S (280; commit `b84f37d`, lock passed; the run metadata's "dirty" flag only reflects untracked files written by
the preregistered on-VM tuning step; `results/G8/g8_S.json`). Total-collision metric (diluted by defect D8); the
avoidable-collision re-test is in G9.

## Stage 1 (Holm over five)

| variant vs A0 | collisions | completion | decision |
|---|---|---|---|
| A0_static (no persistent regulator) | +1.6% [−3.4, +7.0] (upper 6.1%) | −0.4 pp | removable (non-inferior) |
| A0_single (1 predictor member instead of 5) | +0.0% [−6.3, +6.4] (upper 5.4%) | −0.4 pp | removable |
| A0_nomem (no entity memory) | **+28.9% [+18.8, +40.7]** | −1.4 pp | **needed** |
| A0_heur (hand-written consequence model) | **+150% [+125, +181]** | −5.5 pp | **needed** |
| A0_mem (G2b memory; λ retuned → 1.21) | +22.8% [+14.1, +32.8] | −0.2 pp | not added |

## Stage 2 — combined minimal model

A0_min (static penalty + single member, λ retuned) vs A0 on fresh S lifetimes: collisions **+17.4% [+9.4, +26.5]**,
completion +0.0 pp → **not non-inferior**. Individually removable components are not jointly removable; A0 is kept as
is. Compute: A0_min 278 µs per decision vs A0 629 µs (−56%), parameters 7.3k vs 36.5k.

## Notes

- Neural prediction and entity memory are the two components that clearly earn their place.
- The memory assembly's result depends on the risk weight: with A0's calibrated penalty (G6) it is better than A0
  everywhere; after equal retuning on validation J (which rewards completion), the tuner chose a much lower penalty
  and the result was worse. The penalty should be calibrated to a safety target, not to J alone.
