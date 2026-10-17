# E7 addendum: fixed before the confirmatory run (2026-10-11)

Supplements `MASTER_PREREGISTRATION.md` §2 E7. Nothing here relaxes a criterion. Written after the pilot (partition
P, 20 lifetimes × 15 fault conditions × B0/B1/B2; run on a temporary cloud VM), whose only uses were debugging and
the sizing below. For sizing only the standard errors were computed; the per-fault collision-rate estimates were not
inspected.

## Frozen inputs and code

As in the E2 addendum (predictor, tuned weights). Code: the commit that contains this file; `scripts/e7_run.py`
refuses partition X unless this file is committed and the checkout is clean.

## Design

- **Partition X, N = 100 lifetimes** (indices 0–99) × 5 episodes of 60 s for each of the 15 conditions in
  `e7_run.py` (`none` plus 14 faults) and each of B0, B1, B2; the same lifetimes for all systems within a fault.
- **Sizing rule** (not fixed by §4.1, fixed here): the smallest N whose largest per-fault standard error of the
  relative collision difference (B1 or B2 vs B0) gives a 95% half-width ≤ 0.10, with bounds 100–400 (the E2
  bounds). Pilot: largest SE 0.103 (sensor outage, B2, n = 20) → N = 81 → floor 100.
- **Analysis:** `scripts/e7_analyze.py --partition X` as committed (paired cluster bootstrap over lifetimes, 10,000
  resamples, seed 0, ratio of totals).

## Criteria (operational definitions)

1. **No exception into the robot loop:** the number of exceptions raised by B1/B2 (their own code: observe, decide,
   episode hooks) that reach the harness is 0 under every condition. Under `model_failure` the predictor failure
   is the injected fault itself: the runner raises it deliberately (10% of decisions) and counts it; those injected
   exceptions are excluded from (1) and scored under (2). Any other exception, under any fault including
   `model_failure`, counts against (1). Disclosed from the pilot: all 2,428 counted exceptions were injected
   `model_failure` events, each followed by a GO fallback.
2. **Fallback to GO within one decision period:** unit test `tests/test_e7_robustness.py`; plus, under
   `model_failure`, fallbacks = injected failures.
3. **Safety supervisor independence:** unit test `tests/test_e7_robustness.py`.
4. **Not worse than B0:** per fault and per system (B1, B2), the CI upper bound of the relative collision-rate
   difference vs B0 ≤ +5%. Reported per fault; "all faults pass" requires every fault × system to pass. No
   multiplicity adjustment (each fault is its own robustness requirement; adjusting would make the criterion
   weaker).
