# H4b addendum: developmental-expression re-test with a correctly specified outcome-shuffle control (2026-10-13)

## Why

The H4 gate (`prereg/H34_ADDENDUM.md` §7) **failed for every system** on criterion 3 ("shuffled outcomes"). That
result stands.

**Diagnosis.** The control shuffled outcomes *within* each lifetime.
- That preserves each lifetime's overall outcome rate: about 0.32 under risky_B and 0.05 under safe.
- So it removes the entity–outcome association but not the difference in total experienced harm.
- It therefore cannot null the between-history effect E2, even for a system whose expression is entirely
  experience-driven.

**Exploratory evidence** (X data, reported in `reports/H34_REPORT.md`): within risky_B lifetimes, the
entity-specific contrast κ(B1) − κ(A1) vanishes under that shuffle for M_id (+0.78 → +0.02) and the GRU
(+0.41 → −0.01), but not for S7 (+0.66 → +0.10), which shows a training-set prior.

H4b re-tests the developmental claim on **fresh experience streams** with a control that nulls E2 by construction.
It is fixed before those streams exist.

## Data

- New X streams for risky_B and safe, 30 lifetimes each.
- Lifetime indices are 1000 + i: never used.
- Same generator and measurement (`h34_experience.py --index-offset 1000`).
- All models are frozen: H3 tuning (`tuning.json`) and trained ensembles (`models/`). Nothing is retrained.

## Controls (per system, paired by lifetime index i)

| control | definition |
|---|---|
| **pooled shuffle** (replaces criterion 3) | pool the 96 observed outcomes (y, magnitude) of the pair (risky_B i, safe i); permute them (seed i); reassign the first 48 to robot 1's encounters and the rest to robot 2's. Identities, zones and order of encounters are unchanged. Both robots have the same expected experience, so E2 has expectation 0 for any system whose expression depends only on experience. |
| within-lifetime shuffle (secondary) | as in H4; reports the entity-specificity contrast |
| state reset before evaluation | as in H4 |

## Measures

- **E2:** κ_e(B1/plain | risky_B) − κ_e(B1/plain | safe) at lifetime end (bootstrap CI over the 30 pairs).
- **Specificity (secondary):** κ_e(B1) − κ_e(A1) within risky_B, original vs within-lifetime shuffle.
- **Physical** (the "most important test" on the fresh streams):
  - for M_id, S5, S6, S7, S4, S0 and S1;
  - each pair of identical robots executes its end-of-lifetime B1 intent (H34 §6 protocol, sim seed
    `sim_seed("H4", "X", 900000 + 1000 + i)`);
  - non-speed leave-one-pair-out history classification, falls, arrival.

## H4b gate (per system)

1. E2 > 0.20 with CI lower bound > 0.
2. E1 ≥ 0.50, carried from the H34 X evaluation (unaffected by the control definition).
3′. Under the pooled shuffle, the E2 CI includes 0 or |E2| < 0.05.
4. Under state reset, E2 = 0.
5. Physical: history classification ≥ 0.80, 0 falls, arrival ≥ 95%.

- **The developmental claim (claim C, simulation, interpretability untested) is supported for a system only if 1–5
  hold.**
- "Beats A0 + memory + fixed lookup" uses the H34 §8 rule on the H34 X evaluation, plus not-worse E2 (point) here.
- **If no system passes, there is no developmental claim.**
