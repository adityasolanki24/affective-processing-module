# H8 report: integrated evaluation and selection (2026-10-13)

**Setup:**
- Governed by `prereg/H8_ADDENDUM.md`.
- Fresh X streams: 210 lifetimes, 10,080 physically measured encounters, never used before.
- 540 physical NERVA episodes.
- Results in `results/H34/X8/h8_results.json`.

## Intent level (fresh data)

| system | E1 | CE | E2 history effect [CI] | pooled-shuffle E2 | revision median (fraction) | E6 | E7 context | disposition (k = 96) / meets H6 | µs per update / per expression |
|---|---|---|---|---|---|---|---|---|---|
| **B0: A0 + memory + fixed lookup** | 0.68 | 0.168 | **+0.84 [0.79, 0.88]** | +0.04 [−0.04, 0.11] | > 24 (0.43) | **+0.80** | 0 | +0.07 / no | 2 / 193 |
| I: context + change point + global EMA α 0.07 | 0.73 | 0.158 | +0.78 [0.71, 0.84] | +0.09 [0.01, 0.16] | 20 (0.70) | +0.71 | **+0.65** | +0.08 / no | 43 / 138 |
| I − context | 0.72 | 0.165 | +0.85 | +0.04 | 16 (0.87) | +0.82 | 0 | +0.08 / no | 58 / 173 |
| I − change point | 0.72 | 0.161 | +0.78 | +0.09 | 24 (0.57) | +0.71 | +0.65 | +0.08 / no | 2 / 124 |
| I − global | 0.71 | 0.136 | +0.70 | +0.09 | 24 (0.63) | +0.63 | +0.65 | 0 / no | 57 / 165 |
| global only (scalar EMA) | 0.41 | 0.239 | +0.36 | −0.01 | 8 (1.00) | +0.36 | 0 | **+0.34 / yes** | 0 / 132 |
| S6 fast/slow GRU | 0.73 | 0.147 | +0.29 | +0.03 | 8 (0.97) | +0.31 | +0.15 | +0.10 / yes | 1,917 / 1,120 |
| A0 + learned mapping (L1) | 0.71 | 0.151 | +0.59 | +0.01 | > 24 (0.40) | +0.60 | +0.15 | +0.01 / no | 2 / 548 |

## Preregistered rules

**1. I does not replace B0.**
- I is non-inferior on E1 (lower bound +0.02).
- I is better on revision and context.
- But its E2 non-inferiority fails: the difference has a lower bound below −0.05.

**2. The learned mapping does not replace the fixed lookup.**
- It has better calibration (CE difference [−0.023, −0.012]).
- But its history effect is lower (0.59 vs 0.84).

**3. Ablations.**

| element | necessary? | evidence |
|---|---|---|
| context keying | yes, for context | E7 0 → 0.65 |
| change-point beliefs | yes, for revision | 0.57 → 0.70 revised |
| memory | yes, for E1 and E2 | global only: E1 0.41, E2 0.36 |
| the global term inside I | no | at weight 0.3 it does not produce a disposition in I, so it cannot be necessary |

**4. Physical** (B0, I and A0 + L1; 30 pairs each; 0 falls and 100% arrival in all 540 episodes). History
classification:

| system | (a) B1, original situation | (b) novel B3, marked zone, new geometry | (c) shift geometry (1.8 m, 2.4 rad) |
|---|---|---|---|
| B0 | 1.00 | 1.00 | 0.98 |
| I | 0.95 | 1.00 | 0.97 |
| A0 + L1 | 1.00 | 0.98 | 1.00 |

## Reading

- **The conventional baseline survives every comparison.**
  - It expresses history most strongly.
  - It generalises across entities of the same kind.
  - Its expression stays recoverable in new situations and geometries.
  - It costs microseconds.
- **Its weaknesses are specific and fixable with explicit, transparent parts:**
  - slow revision → change-point beliefs;
  - no context → context-keyed memory;
  - weak disposition → a global state with enough weight.
  Bundling them (I) bought revision and context at a cost in history effect. Under the preregistered rule that is
  not an improvement.
- **The integrated I's pooled-shuffle E2 (+0.09 [0.01, 0.16]) does not quite null.** Neither do its two ablations
  that keep context keying. This needs checking before context keying is adopted. B0's control nulls.
- **Dispositions remain the job of a global state.** Inside a memory-centred lookup with weight 0.3 they are too
  weak. A scalar EMA alone shows them clearly (+0.34, ICC 0.88). The right combination weight is open.
