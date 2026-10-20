# J1 addendum: frozen calibration and test run (committed before any J1-X data, 2026-10-17)

This addendum supplements `MASTER_PREREGISTRATION.md` §2. Nothing in it relaxes a criterion.

## Frozen inputs (git-normalised SHA-256)

| file | hash | contents |
|---|---|---|
| `configs/J1_systems.json` | `da512a9d…18d05` | programme I's frozen I1 systems, plus R_decay (freeze at 1,050 s) |
| `configs/J1_readout_bank.json` | `747f3226…12350` | 1,000 decision vectors from A0 on J1-T S3 0–39 |
| `configs/J1_calibration.json` | `a3d23e1c…3f2fe` | J1-V S3 0–199 for every system |

**Calibration values:**

| system | θ (belief) | gap | read-out careless / careful | observed careful-action share careless / careful |
|---|---|---|---|---|
| A0 | 0.0162 | +0.0156 | 0.467 / 0.455 | 0.566 / 0.389 |
| G2b | 0.0420 | **−0.0141** (inverted) | 0.488 / 0.495 | 0.617 / 0.443 |
| R_sparse | 0.0132 | +0.0000 (no separation) | 0.447 / 0.447 | 0.539 / 0.369 |
| R_static | 0.0174 | +0.0174 | 0.485 / 0.446 | 0.587 / 0.392 |
| R_dense | 0.0170 | +0.0155 | 0.488 / 0.458 | 0.599 / 0.407 |
| Oracle | 0.1464 | +0.2456 | 0.540 / 0.452 | 0.621 / 0.414 |
| R_decay | as R_dense (§2) | | | |

R_sparse's calibration gap is about 0. For it, every threshold-based adaptation measure is degenerate, and is reported
as such.

## Test run

- **Command:** `j_run.py --experiment J1 --partition X --n 200 --systems configs/J1_systems.json --scenarios
  S1,…,S10 --addendum J1_ADDENDUM.md`.
- **Size:** 7 systems × 10 families × 200 = 14,000 lifetimes.
- **Code:** submitted with `--commit`, so it runs in a clean clone of the commit that adds this file.
- **Analysis:** `j1_analyze.py test`, with the criteria and validity rule exactly as in §2.
