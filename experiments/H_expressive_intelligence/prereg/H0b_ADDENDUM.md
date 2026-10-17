# H0b addendum: re-test of the expressive-action interface after the H0 gate failure (2026-10-12)

## Why this addendum exists

The H0 gate **failed** on (d) and (e) (`reports/H0_REPORT.md`). That result stands and is not edited. H0b fixes the
two diagnosed blockers and re-tests on **fresh seeds**. It is fixed before any H0b data on X.

**The diagnosed blockers:**
1. **Standing push robustness.** NERVA's validated controller falls from ≥ 0.20 m/s backward pushes while standing.
   This was never covered by NERVA's gate. Expressive pauses add standing time.
2. **No progress guarantee in the adapter.** The slowest probe stalled.

**Fresh seeds.** H0's X partition with new index ranges, never used before:
- 1,000,000 + i for the nominal and fallback suites;
- 3,000,000 + i for the standing-push suite.

**Pilot evidence used to design H0b (partition P only, `results/H0b/pilot/`):**
- Standing-push sweep, 3 seeds: the neutral candidate survives ≤ 0.15 m/s in all directions; E1′ survives ≤ 0.10.
- The revised adapter, 4 seeds per probe and configuration: 100% arrival, no falls.
- `fallback_b` at 0.15 m/s: no falls.
- At 0.30 m/s the expressive probe fell 3/8, enforced and shadow alike; the neutral probe fell 0/8.

## Changes under test (code at the commit adding this file)

1. **Adapter task guarantee** (`apm_express.adapter`).
   - Forward bursts last ≥ 1.5 s.
   - If average approach progress since the target appeared drops below 0.025 m/s (after 4 s), catch-up mode
     starts. Catch-up allows no new pauses, retreats or scanning stops and uses tempo 1, until progress reaches
     0.03 m/s.
2. **Push requirement re-specified.** Expressive episodes must tolerate the push magnitude **validated for both
   walking and standing**:
   - m* = min(0.30, the largest standing magnitude with no falls in any direction);
   - m* is measured on X below.
   - Robustness at 0.30 m/s while standing is reported as a controller limitation. It is not something the
     expression layer can provide.
3. **Spatial features smoothed over one gait period** (`features.py`, with a test). This was fixed after H0 (see
   `DEVIATIONS.md`).

**Unchanged and carried over from H0 (those suites do not use the adapter):** the admitted command dwell (0.4 s),
the head-at-rest box (±0.4 × ±0.4 rad) and the E1′ pitch admission. H0's criteria (a) reproduction and
(b) sequencing are carried over as passed.

## Suites (partition X, fresh indices)

| suite | grid | seeds | monitor |
|---|---|---|---|
| standing push (`h0b_rest_push.py`) | neutral and E1′; rest; push at 3 s of 0.10, 0.15, 0.20, 0.25, 0.30 m/s in 8 heading-relative directions; 8 s | 5 (400 trials) | shadow |
| `fallback_b` | primary interface; probe slow_pausing at m* (enforced); slow_pausing at 0.30 (enforced and shadow, paired); neutral probe at 0.30 (enforced); push at t = 6 s, 8 directions; 20 s | 5 per cell (160 trials) | as stated |
| `nominal` | primary interface and secondary E1′ interface (the admitted ones from H0); 5 probes; 60 s; targets as in H0 | 20 per probe per configuration (200 trials) | enforced |

**Order:**
1. standing push;
2. `h0_analyze.py push_star`, which fixes m*;
3. `fallback_b --push-star m*` and nominal;
4. `h0_analyze.py gate_b`.

## H0b gate (all required)

| criterion | definition |
|---|---|
| (a), (b) | carried over from H0 (passed) |
| (c) | as in H0, recomputed on the H0b nominal suite: ≥ 2 families {temporal, spatial, posture} realised (≥ 95% of paired seeds in the designed direction) |
| (d′) | m* ≥ 0.10 m/s; **no fall** in any enforced slow_pausing episode pushed at m*; at 0.30 m/s, the enforced-only fall pairs are ≤ the shadow-only fall pairs (non-worsening); every fallback step neutral |
| (e′) | primary nominal: no fall and no trip; every probe ≥ 95% arrival within ±0.15 m of its stopping distance; head offsets ≤ 0.02 rad whenever a locomotion command is active |

**H0b passes** if all hold. In that case:
- H1 may start on this interface;
- the H1 addendum states the standing-push limitation and fixes H1's push magnitude at m*.

E1′ is admitted as a secondary H1 configuration if its nominal suite has no fall, no trip and ≥ 95% arrival per
probe (rule S). Its standing robustness is reported.

**If H0b fails, H1 does not start.** The blocker is reported.

## Reported but not gated

- The falls of the expressive probe vs the neutral probe at 0.30 m/s: the exposure cost of expressive pauses.
- The standing-robustness map for both controllers.
