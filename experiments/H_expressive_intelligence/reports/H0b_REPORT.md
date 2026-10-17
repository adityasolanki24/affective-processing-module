# H0b report: interface re-test after the H0 failure (2026-10-12)

## Verdict: **H0b gate PASSES**

**Setup:**
- Governed by `prereg/H0b_ADDENDUM.md` (commit `64590af`);
- partition X with never-used seed indices;
- 760 trials, local CPU;
- results in `results/H0b/X/`; the gate is `results/H0b/X/gate_b.json`.

The H0 failure (`H0_REPORT.md`) stays on record. H0b tested the fixed interface. H1 may start on it.

| criterion | result |
|---|---|
| (a) reproduction, (b) dwell 0.4 s | carried over from H0 (passed) |
| (c) channel families | **pass**: temporal, spatial and posture all realised in 100% of paired seeds, now with gait-sway-corrected features |
| (d′) standing push and fallback | **pass**: m* = 0.15 m/s; 0/40 falls in expressive episodes pushed at m*; at 0.30 m/s, 0 enforced-only vs 0 shadow-only falls; every fallback step neutral |
| (e′) nominal adapter | **pass**: 0 falls and 0 trips in 100 trials; arrival 100% for every probe (slow_pausing: 65% → 100%); head offsets ≤ 0.0045 rad while moving |

## Standing push robustness (new measurement; 5 seeds × 8 directions per cell)

| magnitude (m/s) | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|
| neutral candidate: falls / 40 | 0 | 0 | 5 | 13 | 15 |
| E1′: falls / 40 | 0 | 5 | 14 | 15 | 15 |

- All falls come from pushes from behind (135–225°).
- **Standing on its validated controller, NERVA tolerates 0.15 m/s from every direction** (E1′: 0.10 m/s).
- While walking it tolerates 0.30 m/s (NERVA's gate).

## The exposure cost of expression (reported, not gated)

These are 0.30 m/s pushes at t = 6 s, 8 directions × 5 seeds, monitor enforced:

| probe | falls / 40 |
|---|---|
| neutral approach (walking at t = 6 s) | 0 |
| slow_pausing (often standing during a pause) | 8 |

Expressive pauses put the robot more often in its weaker standing state. **The interface therefore guarantees
safety only up to the standing-validated magnitude (0.15 m/s).** Making pauses safer needs a controller that is
robust while standing. That is a NERVA controller task, outside the expression layer.

## Interface used from H1 on (`results/H0/X/admitted_interface.json` + H0b adapter)

- **Controller:** NERVA's validated neutral candidate, behind NERVA's deployment contract.
- **Channels:**
  - command sequencing with dwell ≥ 0.4 s: tempo, pauses/hesitation, retreat, path directness, stopping distance,
    orientation;
  - head offsets at rest only (|pitch|, |yaw| ≤ 0.4 rad, ≤ 0.75 rad/s), returned to neutral before moving.
- **Task guarantee:** average progress ≥ 0.025 m/s; forward bursts ≥ 1.5 s; retreat ≤ 25% of forward time.
- **Independent monitor:**
  - NERVA's tilt/angular-speed rule, plus joint-margin < 0.02 rad and 25-step force-saturation trips;
  - fallback to neutral rest.
- **Push tolerance:** ≤ 0.15 m/s from any direction (standing-limited).
- **Secondary configuration (rule S satisfied):** E1′, with pitch lean of 1–2° while walking, dwell 0.6 s, head box
  ±0.2 × ±0.4, standing tolerance 0.10 m/s.
