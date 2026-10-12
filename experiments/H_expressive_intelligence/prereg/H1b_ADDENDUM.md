# H1b addendum: re-test of H1's safety criterion after the stop-transition fall (2026-10-12)

## Background

H1 failed G1 (`reports/H1_REPORT.md`): one fall in 112 pushed trials at 0.15 m/s. The push arrived from behind just
after a walk-to-rest transition.

This addendum is fixed before any H1b X data. It:
- re-specifies the push requirement as the tolerance measured **across stop transitions**;
- re-tests G1 on fresh seeds.

The H1 failure and its criteria are unchanged. G2–G5 passed in H1, do not depend on the push magnitude, and are
carried over.

**Pilot (P, 480 trials, `results/H1b/pilot/`):** no falls at ≤ 0.125 m/s, at any delay; 2/16 falls at 0.15 m/s,
0.4 s after the stop.

## Suites (partition X, never-used H1 indices)

1. **Transition push sweep** (`scripts/h1b_transition_push.py`, index offset 700,000, 5 seeds):
   - neutral controller walks forward 5 s, then rests;
   - push of m ∈ {0.05, 0.075, 0.10, 0.125, 0.15} m/s at delay d ∈ {0, 0.2, 0.4, 0.7, 1.0, 2.0} s after the stop;
   - 8 heading-relative directions;
   - 1,200 trials.

   **m\*\*** = the largest magnitude with no fall at it and at every smaller magnitude, over all delays and
   directions. The transition-validated tolerance of the interface is **min(m\*\*, m\* = 0.15)**.
2. **Push arm re-test** (`h1_repertoire.py --arm push --push m**`, trial indices offset to 600,000 + i): X0 and
   the six X1 styles × 8 geometries × 3 seeds = 168 trials, one push of m** at a random time in [4, 10] s from a
   random direction.

## Criteria

- **G1′:** m** ≥ 0.10 m/s **and** 0 falls and 0 trips in the 168 re-test trials. Push-arm arrival is reported.
- **H1 passes if G1′ holds** (G2–G5 carried over). The repertoire is the H1 repertoire, and the interface's
  documented push tolerance becomes m**.
- **If m** < 0.10 m/s, or any re-test trial falls, H1 fails definitively.** H3/H4 are then not run, and the
  blocker (controller robustness during expressive stop transitions) is reported for NERVA.
