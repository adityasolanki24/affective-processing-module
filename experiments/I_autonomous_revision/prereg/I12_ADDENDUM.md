# I1 / I2 addendum: fixed values and test runs (committed before any partition-X data, 2026-10-17)

This addendum supplements `MASTER_PREREGISTRATION.md`. Nothing in it relaxes a criterion. One deviation applies:
D1, the p_random range (`DEVIATIONS.md`).

## Fixed from T (I0-T 0–399, S3, A0's own experience; no ground truth read)

| model | file | git-normalised SHA-256 | details |
|---|---|---|---|
| dense | `configs/types_T_dense.json` | `67b3569b…f8dc2` | BIC selected **K = 3**: π = (0.575, 0.185, 0.240), r = (0.0087, 0.0260, 0.0262) |
| sparse | `configs/types_T_sparse_K2.json` | `35bb750a…a7d81` | BIC selected K = 1 (collisions alone show no individual differences); the preregistered comparator uses **K = 2** |

## Fixed from V (`results/V/`)

| parameter | value | notes |
|---|---|---|
| λ (R_dense) | **7.208 × 10⁻⁴ s⁻¹** | time constant ≈ 23 min of experience. Objective: self-supervised predictive log-likelihood, −1.0863 |
| λ (R_sparse) | **5.281 × 10⁻³ s⁻¹** | objective flat to 4 decimals across the grid: λ is not identifiable from collision-only evidence |
| ν (VOI) | **3.939** | J 137.8, constraint met (0.0061 ≤ 0.0064). VOI probes at almost every keep-away encounter for every ν (20.2–21.7 probes per lifetime) |
| conf_max (uncertainty) | **0.914** | top of the grid; J 135.9 |
| p_random (random control) | **0.868** | D1: re-selected over [0.02, 1.0]; 20.8 probes vs VOI's 21.2. The preregistered-range value, 0.575 (10.8 probes), is reported alongside |

**System files:**
- `configs/I_systems_X_I1.json`: `8cb6a4a3…bc040`
- `configs/I_systems_X_I2.json`: `e38f4a1a…3d77e`

Frozen predictor: `runs/E2/predictor.json` `0735d854…be314`.

## Test runs

- **I1:** `i_run.py --experiment I1 --partition X --n 200 --systems configs/I_systems_X_I1.json --scenarios S1,…,S10
  --addendum I12_ADDENDUM.md`.
- **I2:** the same with `--experiment I2` and `I_systems_X_I2.json`.
- Indices 0–199 of each family, 10 families.
- I1: 6 systems = 12,000 lifetimes. I2: 8 systems = 16,000 lifetimes.
- Submitted with `--commit`, so each runs in a clean clone of the commit that adds this file, and the confirmatory lock
  must pass.
- Analysis: `i_analyze.py --experiment I1|I2`, criteria exactly as in §6 of the master preregistration.
