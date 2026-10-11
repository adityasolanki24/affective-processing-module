# G0 — audit of programme F's E2 comparison and the fair-comparison framework (2026-10-12)

## 1. What F's E2 actually compared (code and data read, not summaries)

| aspect | B1 | B2 | consequence |
|---|---|---|---|
| tuning budget (`F/scripts/e2_tune.py`) | 18 configurations (λc ∈ 6 values × λs ∈ 3) | 162 (the same 18 × k_pad ∈ 3 × k_t ∈ 3) | **9× more tuning freedom for B2**, acknowledged by F as a bias in B2's favour |
| tuning data | E2-V lifetimes 100–129, 4 episodes, max mean J | same | equal |
| tuned validation J | 22.18 (λc 1.0, λs 2.0) | 22.37 (λc 0.5, λs 2.0, k_pad 2, k_t 0.25) | nearly identical on validation; B2's selected configuration had a lower validation collision rate (0.100 vs 0.133/min) |
| tested collision cost | static λc = 1.0 | λc·(1 + 2·(max(0, A) + max(0, −V))) with λc = 0.5, i.e. a *state-dependent* weight; with the typically negative valence of v0.1 (F AUDIT D6) its average is above 1 | B2 = "B1 with a different, time-varying collision weight + two tendency bonuses" |
| observations | the same `Obs` stream (20 Hz) | the same stream, re-encoded by the adapter for the APM | equal information; B2 integrates it over a longer history (persistent affect), B1 sees only the last decision period plus a 5 s collision trace |
| predictor, memory, safety, actions | identical | identical | — |

**Exposure re-analysis (exploratory, F's E2 X raw data):** B2's collision reduction is not bought by moving less.

| system | collisions/min | collisions/100 m | m travelled/min | action use GO/SLOW/YIELD/DETOUR |
|---|---|---|---|---|
| B1 | 0.131 | 0.764 | 17.19 | 0.63 / 0.26 / 0.11 / 0.00 |
| B2 | 0.098 | 0.577 | 16.97 | 0.61 / 0.15 / 0.22 / 0.02 |
| B2_shuffled | 0.113 | 0.668 | 16.91 | 0.60 / 0.19 / 0.18 / 0.03 |
| Oracle | 0.122 | 0.704 | 17.28 | 0.62 / 0.27 / 0.11 / 0.00 |

Per distance the reduction is −24.5% (per minute −25.4%), while distance per minute fell only 1.3%. B2 swaps SLOW
for YIELD. DETOUR is almost never used by any system.

**Conclusion of the audit:** F's E2 result is internally valid (independent test lifetimes, locked runner), but it
cannot separate "PAD/persistent regulation helps" from "B2's family had more tuning freedom and a different
effective collision weight". G1 removes the confound by giving every family the same number of tuned
configurations, a wider λ range, and simple persistent alternatives.

## 2. Framework built (code: `src/apm_select/`, tests: `tests/test_g0_integrity.py`)

- **Regulators** R0–R8 and replay-based negative controls (`regulators.py`). R6 reproduces F's B2 decisions exactly
  and the static regulator reproduces F's B1 exactly (tests).
- **Instrumentation** (`instrument.py`): encounters (< 1.0 m) and close encounters (< 0.5 m), close-exposure time,
  action switches, A→B→A oscillations, hesitation (SLOW/YIELD with nobody near), decision latency. It forwards
  every call unchanged (test: identical actions and outcomes with and without it).
- **Statistics** (`stats.py`): paired cluster bootstrap for any ratio-of-totals metric (one- or two-level),
  superiority rule, non-inferiority with p-values for Holm.
- **Equal-budget tuning** (`tuning.py`): K seeded Latin-hypercube draws per family from preregistered spaces that
  contain F's tuned values; R5 trained with the same number of lifetime evaluations.
- **Provenance** (`provenance.py`): F's confirmatory lock plus G source hashes.
- **Seeds**: new programme entropy; no G seed equals any F seed (test over 15 × 50 × 8 × 5 seeds).

## 3. Integrity tests (14, all pass)

static regulator ≡ F's B1 · R6 ≡ F's B2 · instrumentation has no behavioural effect · R2/R3/R4 are causal (a prefix
of the observation stream gives the same multiplier) · observations contain no hidden state (no disposition,
friction or pose fields) · regulators run on plain observation fields only · G seeds independent of F · equal and
deterministic tuning budgets within the declared ranges · perpetual YIELD and perpetual GO score below B1 on J ·
replaying a system's own recorded modulation reproduces its decisions · the miscalibration condition multiplies
estimates as specified · dense/sparse worlds contain the specified numbers of people.

**Finding during testing:** a robot that never moves can still be credited with an occasional goal, because the
pre-generated goal sequence sometimes places a later goal within the 0.3 m tolerance of where it stands (4 of 237
goals over 20 pilot lifetimes; distance travelled 0). F's E0 test (4) used specific seeds where this did not occur.
It cannot be exploited (J of perpetual YIELD stays far below B1's) and affects all systems equally; recorded, not
changed (the testbed is frozen).

## 4. Gate

All integrity tests pass, the comparison framework equalises information, actions, predictor, memory, safety and
tuning budget, and records exposure. **G0 gate: passed.** Model selection (G1) may proceed.
