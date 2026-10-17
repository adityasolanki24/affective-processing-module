# H6 addendum: expressive dispositions (fixed before H6 test data, 2026-10-12)

**Prerequisites:** H1 passed. H6 uses the H3/H4 X experience streams and the H3-tuned and H3-trained models.

**Script:** `scripts/h6_dispositions.py` at the commit that adds this file.

## Design

**Long lifetimes.** 96 encounters, built by joining two physically measured X streams (30 robots per world):

| world | first 48 | last 48 |
|---|---|---|
| stable_risky | unpredictable | unpredictable |
| stable_safe | safe | safe |
| changing | unpredictable | safe |
| reverse | safe | unpredictable |

**Disposition D(k).** The expressed caution κ_e toward D1, an entity of a kind never encountered. Neither identity
nor kind knowledge can explain it, so it can only reflect the robot's general experience. There are no personality
presets: every system starts identical.

## Measures

| measure | definition |
|---|---|
| world effect | D(96) of stable_risky minus stable_safe (paired by robot index), mean and 95% bootstrap CI; also at k = 48 |
| repeatability | ICC(1) of D at k ∈ {60, 72, 84, 96} across the 60 robots of the two stable worlds (the animal-personality definition: consistent between-individual differences) |
| revisability | in the changing world, encounters after k = 48 until D drops below the midpoint of the stable_risky and stable_safe levels at k = 96 (KM median; censored at 48) and the fraction revised |
| acquisition | the same, upward, in the reverse world |

## Criterion

**A system "shows a revisable disposition" if all of:**
- the world effect at k = 96 is > 0.10 with CI lower bound > 0;
- ICC ≥ 0.5;
- revision median ≤ 48 encounters, with ≥ 80% of robots revised.

A system that stays cautious after the world becomes safe (revision fails) is reported as **not** showing a
desirable disposition, even with a large world effect.

**Not claimed:** personality, temperament or feelings. These are measured tendencies of expression in simulation.
