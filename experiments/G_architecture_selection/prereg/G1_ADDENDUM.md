# G1 addendum: fixed before tuning and before any partition-X/S data (2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G1. Nothing here relaxes a margin. Written after the pilot (partition P,
40 lifetimes × 10 episodes, F's tuned values for R0/R6 and untuned mid-range values for R2–R4, R7, R8; run locally),
whose only uses were debugging, timing and the sizing below. Only integrity fields (fallbacks, decision counts,
encounter counts, latency) and the power-mode standard errors were inspected.

## Frozen inputs

Predictor `runs/E2/predictor.json` (SHA-256 `0735d854…be314`, F's, unchanged, checked by the runner). Code: the
commit that contains this file; `scripts/g1_run.py` refuses X/S unless this file is committed and no tracked file
is modified. `configs/G1_selected.json` is produced from tuning by `scripts/g1_select.py` and committed (its SHA-256
is recorded by the runner) before X/S are run.

## Tuning (partition V and T only)

- **K = 48 configurations per family** for R1, R2, R3, R4, R6, R7, R8: seeded Latin hypercube
  (`apm_select.tuning`, sample seed 20261012, per-family offset) over the spaces in `apm_select/tuning.py`
  (λc log-uniform 0.1–10, λs 0–4; filter time constants, gains and impulses as listed there; every space contains
  F's tuned values). Each configuration is evaluated on **G1-V lifetimes 0–29 × 10 episodes** (common random numbers).
  Selection: highest mean J.
- **R5** (learned scalar regulator, 10 parameters): antithetic ES (8 pairs, σ 0.25, Adam lr 0.08, centred ranks),
  6 G1-T lifetimes × 10 episodes per candidate per generation, fresh lifetimes each generation, start = F's B1
  (m ≡ 1, λc 1, λs 2). Budget = 48 × 30 = 1,440 lifetime evaluations: 14 generations × 96 + the final evaluation on
  G1-V 0–29 (30) = 1,374 ≤ 1,440.
- **R_best** = the family among R2–R5 with the highest validation J (`g1_select.py`).

## Test

- **Partition X, N = 280 lifetimes** (indices 0–279) × 10 episodes, nominal condition, all systems of
  `configs/G1_selected.json` (B0, R0–R8) plus the negative controls R6_shuffled, R6_shifted, R_best_shuffled,
  R_best_shifted (trace of lifetime (i + 1) mod N; own trace rotated by 120 decisions).
- **Sizing (§6, bounds 200–400):** pilot SEs at 40 lifetimes required 264 (R6 vs R0, superiority), 196 (R2 vs R0,
  superiority) and 277 (R2 vs R6, non-inferiority at +10% / −2 pp) lifetimes for 80% power → N = 280.
- **Partition S** (shift, exploratory/secondary), N = 100 lifetimes per condition: nominal, miscal_x3, dense, sparse
  (`scripts/g_common.py`), the selected systems unchanged (no retuning), no negative controls.

## Analysis (`scripts/g1_analyze.py`, unchanged after this commit)

Primary family (X, nominal), Holm over five one-sided bootstrap p-values (10,000 resamples, seed 0):
P1 R6 beats R1 (p_A, p_B; superiority rule); P2 R_best non-inferior to R6 (collision r one-sided 95% upper < +10%
and completion difference one-sided 95% lower > −2 pp; p_NI = max of the two); P3 R_best beats R1 (p_A, p_B).
Decisions as in the master preregistration. Everything else is secondary and unadjusted: each system vs R1 and vs
R6 (superiority rule and non-inferiority), negative controls vs their base, R1 vs R0 (what equal tuning alone buys),
exposure-normalised rates (per 100 m, per encounter), close-encounter fraction, hesitation, switching, oscillation,
compute per decision (decide + observe time), action use, the validation tuning tables (frontier), and the S
conditions (each system vs R1 within condition; each system's nominal-vs-shifted change is descriptive because S
conditions use different worlds).
