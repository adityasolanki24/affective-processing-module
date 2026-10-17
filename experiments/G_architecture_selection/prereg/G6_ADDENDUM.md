# G6 addendum: out-of-distribution robustness (fixed before any G6 data and before the G4 test, 2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G6. Nothing here relaxes a criterion.

## Conditions (`src/apm_select/variations.py`), none used for training or tuning anywhere in programme G

nominal (reference) and 13 held-out: obstacles_dense (7–10), crowd_dense (4–6 of 8 people), fast_pedestrians
(U(0.8, 1.6) m/s), careless_heavy (80%), slippery (3–6 patches), identity_uncertain (30% wrong, 20% ambiguous),
deadline_tight (2 s + d/0.45), estimator_x0.5, estimator_x3, sensor_noise (σ 0.1 on distance, bearing, closing;
estimate recomputed), obs_delay_0.25s, slow_actuation (acceleration limit 0.5 m/s²), speed_limit_0.6 (GO cap).
Arena size is not varied (hard-coded layout in the frozen testbed). No retuning per condition.

## Systems (fixed now by rule)

Rules: **A0** (reference; G1 selection R2 + F's predictor + F's memory); **R1** (equal-tuned static penalty);
**A0_mem** = A0 with G2b's best memory design (time-decay M1t, τ 356 s, exposure-gated evidence, uncertainty-guided
probing; `configs/G2b_selected.json` system M1t_probeU); **A1** (G1's R6 = APM v0.1, reference); **B0** (absolute
reference). Neural: for each of A2, A3, A4 the training cell with the highest mean best-validation J
(G4/G5 training records only; the same rule as the G4 shortlist), all 5 seeds.

## Design and analysis

G6-X lifetimes 0–149 per condition × 5 episodes (all systems on the same lifetimes within a condition).
`scripts/g6_analyze.py`: per condition, each system vs A0 (non-inferiority +10% collisions / −2 pp completion,
two-level bootstrap for neural systems) and the collision ratio. **Eligibility** (master §G6): non-inferior to A0 in
≥ 80% of the 13 held-out conditions and never above 2× A0's collision rate. Reported separately: each system's own
nominal → condition change (descriptive), failures per condition, absolute collision and completion levels, and
comparisons with R1 and B0. A0's own robustness is described against R1 and B0 (it cannot be judged against itself).
