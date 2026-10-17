# G — architecture discovery and model selection (APM v0.3)

Finds the smallest learning and regulatory architecture that makes the robot behave better through experience,
by controlled, equally tuned comparisons. Builds on programme F (`../F_behavioral_learning/`), whose results are
preserved and not re-tested.

| file | content |
|---|---|
| `MASTER_PREREGISTRATION.md` | frozen hypotheses, systems, metrics, margins, seeds, budgets for G0–G8 |
| `LITERATURE.md` | established methods behind every candidate, with the simpler baseline each must beat |
| `EXPERIMENT_MATRIX.md` | experiments, dependencies, status |
| `DEVIATIONS.md` | every deviation from the preregistration |
| `prereg/` | per-experiment addenda, committed before each confirmatory run |
| `seeds.py` | the only source of G world seeds (new programme entropy; independent of F) |
| `scripts/` | runners (`g_common.py`, `g1_*.py`, …); `configs/` their inputs; `results/`, `figures/`, `manifests/` |
| `reports/` | G0 audit, compute plan, F accounting, phase reports, `ARCHITECTURE_DECISION.md` |

Code: `src/apm_select/` (regulators, instrumentation, statistics, tuning, provenance); integrity tests in
`tests/test_g0_integrity.py`. Large raw data under `runs/` (git-ignored), described by manifests.
