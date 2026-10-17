# G2 addendum: fixed before G2 tuning and before any partition-X data (2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G2. Nothing here relaxes a criterion. Written after the pilot (partition
P, 6 lifetimes × 9 scenarios × 8 systems with untuned mid-range parameters; local), whose only uses were debugging,
timing and the power-mode SEs, and after G1's confirmatory result (which only fixes the regulator below).

## Fixed context

- Rule and regulator: the G1 validation selection (`configs/G1_selected.json`, best simple = R2, λc 3.95, λs 0.99,
  EMA τ 0.62 s, k 0.45, c 0.017), held fixed for every memory. Frozen predictor as in G1.
- Lifetimes: 30 episodes of 60 s, scenario families S1–S9 (`src/apm_select/scenarios.py`), seeds from `seeds.py`
  (experiment G2).

## Candidates (`src/apm_select/memory.py`)

M0 cumulative (F's rule) with **all**-encounter evidence (`M0_all`, the reference) and with **exposed**-only
evidence (`M0_exp`); M1 forgetting (γ), M2 window (W), M3 hierarchical Bayes (empirical-Bayes prior strength),
M4 Bayesian online change-point (hazard h), M5 two-state hidden-Markov filter (q_low, q_high, switch), all with
exposed evidence; M6 learned recurrent per-identity memory (GRU, hidden 8), exposed evidence.
Oracle memory (true-disposition rates) is an evaluation-only reference.

## Tuning and selection (partitions T and V only; `cloud/g2_tune.sh`)

- **M6 training data:** G2-T lifetimes 0–19 of every scenario (180 lifetimes) logged with the regulator + M0_exp;
  trained by `scripts/g2_m6_train.py` (80/20 lifetime split, BCE, Adam 3·10⁻³, ≤ 60 epochs, patience 8, seed 0).
- **Stage A:** K = 16 Latin-hypercube configurations per parametric memory (M1 γ ∈ [0.5, 0.99]; M2 W ∈ [3, 40];
  M4 h ∈ [0.005, 0.3] log; M5 q_low ∈ [0.005, 0.1] log, q_high ∈ [0.1, 0.6], switch ∈ [0.005, 0.3] log; sample
  seed 20261012), on G2-V lifetimes 0–5 of S1, S2, S3, S5; objective = per-encounter Brier on observable encounters;
  one configuration per memory.
- **Stage B (selection):** M0_all, M0_exp, the tuned M1–M5 and M6 on G2-V lifetimes 100–111 of S1–S9. **M\*** = the
  memory with the fewest state variables per identity that meets on validation: (i) S1 and S2 Kaplan–Meier median
  adaptation ≤ 6 episodes; (ii) mean collision rate ≤ 1.10 × M0_all's; (iii) |false blame| < 0.03; ties → lower
  Brier. If none passes, no M\* is selected and G2 reports the failure (the X run then still evaluates all memories,
  secondary only).

## Test (partition X; `cloud/g2_test.sh`, systems written by `scripts/g2_systems.py` from `configs/G2_selected.json`)

- **N = 60 lifetimes per scenario family** (indices 0–59) × 9 families. Pilot power: closed-loop non-inferiority vs
  M0_all needed ≤ 56 pooled lifetimes; adaptation and false blame have no closed-form power rule, so F's E3 floor of
  60 per family applies (bounds 60–240).
- Systems: M0_all, M0_exp, M1–M6, Oracle, plus three variants of M\*: M\*_all (evidence factor), M\*_probeU
  (uncertainty-guided probing: budget 2 per episode, memory confidence < 0.5, nobody else within 2 m) and
  M\*_probeR (random probing, p = 0.1 per eligible decision, same budget).
- **Primary family** (M\* vs M0_all; Holm over three p-values; all must hold): (i) S1 and S2 KM median adaptation
  ≤ 6 episodes with bootstrap upper CI ≤ 10 (p = share of bootstrap medians > 6, larger of S1/S2); (ii) pooled
  closed-loop collision non-inferiority (+10%, −2 pp completion); (iii) S5 false blame CI within ±0.03 (TOST p).
  Lifetime-cluster bootstrap, 10,000 resamples (adaptation 2,000), seed 0. `scripts/g2_analyze.py --candidate M*`.
- **Secondary:** every memory vs M0_all (behaviour, Brier, calibration, Oracle regret, adaptation, false blame,
  selective-feedback shares, unnecessary avoidance, state size); evidence factor (M\* vs M\*_all); probing (M\*_probeU
  vs M\*_probeR vs M\*, including collisions and probes used); M6 vs the best statistical memory (does neural memory
  add anything?).
