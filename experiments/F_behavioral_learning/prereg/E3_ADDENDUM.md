# E3 addendum: fixed before the confirmatory run (2026-10-11)

Supplements `MASTER_PREREGISTRATION.md` §2 E3. Nothing here changes a hypothesis, metric, criterion or margin.
Written after the pilot (partition P, 20 lifetimes per scenario family, 10 retention lifetimes; run on a temporary
cloud VM, see `DEVIATIONS.md`) whose only uses were debugging and the sizing below. The pilot was inspected only in
the analysis's power mode, which prints standard errors and required sizes, not criterion estimates.

## Frozen inputs

Same as the E2 addendum: predictor `runs/E2/predictor.json` (SHA-256 `0735d854…be314`), tuned weights
`configs/E2_tuned.json` (`aa8bf4b8…c09c4`), Oracle rates `configs/oracle_rates.json` (`215efe2a…4119a8a`). Code: the
commit that contains this file; `scripts/e3_run.py` refuses partition X unless this file is committed and the
checkout is clean.

## Design

- **Partition X, N = 60 lifetimes per scenario family** (indices 0–59) × 30 episodes of 60 s, every system on the
  same lifetimes. Systems per family as committed in `e3_run.py`: S1 and S6: B0, B1, B1-ε, B2, Oracle; S2–S5, S7:
  B0, B1, B2, Oracle. The naive temporal-association memory rides along as an evaluation-only reference.
- **Sizing (§4.1, bounds 60–240):** `e3_analyze.py --power` on the pilot required 46 lifetimes for (i), 3 for (iii)
  and 1 for (iv) (80% power at the criterion margin from pilot SEs); the largest is below the floor, so N = 60.
  Criterion (ii) is a median of a bounded, censored delay and has no power rule; its CI is reported.
- **Retention/history dependence:** `e3_run.py --mode retention`, partition X, N = 60 lifetimes (save, kill and
  restore at the switch episode; memory identity, replay decision agreement, closed-loop action identity (B1),
  history divergence against a different-history robot on lifetime index + 10,000).
- **Operational definitions** are those in the docstring of `scripts/e3_analyze.py` (ranking AUROC at the end of
  episode 29; adaptation delay with "not established" exclusions and censoring at 16; false attribution relative to
  same-lifetime careful identities met at least once; usefulness as careless-identity collisions per minute, B1 vs
  B0, ratio of totals, paired cluster bootstrap 10,000 resamples, seed 0).
- **Criteria (unchanged, B1 memory):** (i) S3 AUROC ≥ 0.80 with CI lower ≥ 0.70; (ii) S1 and S2 median adaptation
  delay ≤ 6 episodes (S1 under B1-ε reported separately); (iii) S5 false-attribution CI within ±0.03;
  (iv) S3 careless-collision rate r ≤ −20% with CI upper < 0 and completion-difference CI lower ≥ −3 pp.
  "Adaptive dispositions" requires all four; otherwise the verdict names each failed criterion as in the analysis.

## Machine

The confirmatory run executes on the same temporary VM as E2, E4 and E7 (`cloud/confirmatory.sh`).
