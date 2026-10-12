# H5 addendum: learned expression mappings (fixed before H5 test data, 2026-10-12)

**Prerequisites:**
- H1 passed (H1b).
- H5 runs on the H3/H4 experience streams. It is interpreted only if the H4 gate passes; otherwise it is reported
  as exploratory.

**Script:** `scripts/h5_mapping.py` at the commit that adds this file.

## Supervision (stated explicitly)

There are no human perception data. The only available expressive supervision is a **reference style computed from
privileged simulator truth**:

    reference intent = blend(relaxed, cautious, true hidden risk of the probe)

This is the designer's intended style for the true situation. It is not validated by observers. A learned mapping
that matches it is "learned to reproduce designer references", never "learned to express emotions". No reward called
"emotion quality" is used.

## Systems and budget

| system | description |
|---|---|
| L0 | A0 memory (M_id) + fixed lookup |
| L1 | MLP on memory-readout features |
| L2 | GRU, end to end from event history |
| L3 | GRU + memory features |

- **Equal budget per learned family:** 8 configurations (hidden 16/32 × lr 1e-3/3e-3 × epochs 40/80) on seed 0,
  chosen by V loss; then a 5-seed ensemble.
- Training data: T streams. Selection: V streams. Test: X streams.
- **L4 (fast/slow)** runs only if H3 finds S6 better than S5 on E1 (the H3 rule). **L5 (learned motion latent) is
  not run** (H1 §X3).

## Metrics (X, from the expressed intents)

- **CE (primary):** mean |κ_e − true risk| over end-of-lifetime probes with truth.
- **E1, E2, E4, E6:** as in `H34_ADDENDUM.md`, computed by `h34_analyze` functions.

## Decision rules

- **"A learned mapping beats the scripted L0" only if all of:**
  - its CE is lower than L0's, with a 95% bootstrap CI (over lifetimes) of the difference entirely below 0;
  - its E1 is not lower (point estimate);
  - its E4 revision is not slower.
- **Preferred mapping:** the simplest in the order L0 < L1 < L2 < L3 whose CE is within +0.02 of the best
  (one-sided 95% bootstrap bound).
- A GRU (L2) that is as good as a specialised model is preferred to it. A simple MLP (L1) that is as good as both is
  preferred to them.
