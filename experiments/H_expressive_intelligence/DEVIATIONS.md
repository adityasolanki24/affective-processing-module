# Programme H deviations

Every departure from `MASTER_PREREGISTRATION.md` or an addendum is recorded here, dated, with its reason and
consequence. Entries are never deleted.

## 2026-10-12: H0 reproduction comparison included wall-clock time

- **What happened.** `h0_analyze.py reproduce` compared NERVA's summary files whole, including `wall_seconds`, the
  run's wall-clock duration. That field differs between machines (531.8 s here vs 336.6 s on NERVA's VM) and is not
  a result.
- **Fix.** The comparison now excludes `wall_seconds`. It was made after the X suites ran and before the H0 gate was
  computed.
- **Evidence.** All other summary fields, and all 698 trial records, are byte-identical.
- **Effect.** Criterion (a) passes. Without the fix it would have failed solely because of the timing field.

## 2026-10-12: H0 gate failed; H0b re-test on fresh seeds

- **What happened.** H0 failed on (d), standing push falls at 0.30 m/s, and on (e), the slowest probe's arrival.
  As preregistered, H1 did not start.
- **Response.** A new addendum (`prereg/H0b_ADDENDUM.md`) adds:
  - an adapter progress guarantee;
  - a push requirement re-specified as the magnitude validated for both walking and standing;
  - a re-test of the nominal and fallback suites on never-used X seed indices (offsets 1,000,000 and 3,000,000).
- **Unchanged.** The H0 result and criteria.

## 2026-10-12: spatial motion features counted gait sway as path

- **What happened.** `features.py` computed path length, lateral deviation and retreat from raw base positions, which
  include step-to-step lateral sway. Straight walking scored a straightness of ≈ 0.37.
- **Fix.** Positions are now averaged over one gait period (27 control steps) before spatial features are computed,
  with a unit test.
- **Effect.** H0's stored nominal features are the uncorrected ones. H0 criterion (c) compares probes with each
  other and its direction is unaffected. Every later experiment uses the corrected features.

## 2026-10-13: Spearman helper returned NaN for constant expression (analysis-code fix)

- **What happened.** `h34_analyze.spearman` tested `std == 0`. S0's constant intent has std ≈ 1e-17, so it returned
  NaN, and the H3 non-inferiority bootstrap could not select a system.
- **Fix.** The test is now `std < 1e-12`, and constant output counts as having no rank information (ρ = 0).
- **Effect.** No other value changed.

## 2026-10-13: H6 ICC for constant expression

- **What happened.** `h6_dispositions.icc1` was guarded by `std > 0`. Constant output (S0, std ≈ 1e-17) gave ICC = 1.0 instead of "undefined".
- **Fix.** The guard is now `std > 1e-12`, returning NaN.
- **Effect.** No disposition verdict changes. S0 and S1 fail either way.
