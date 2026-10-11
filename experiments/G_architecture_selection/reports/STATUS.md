# Programme G status / handoff (2026-10-12, ≈14:15 UTC)

## Done
- G0: framework, 14 integrity tests, audit (gate passed) — `reports/G0_AUDIT.md`.
- G1: pilot, addendum (N_X = 280, S 4 × 100), equal-budget tuning on V (VM 13:48–14:06, ≈ US$0.87),
  frozen selection `configs/G1_selected.json` (R_best = R2, EMA).
- E1 (F) stage 1: wheeled and NERVA P/T/V data regenerated **bit-identically** to F's manifests; models training.
- G2: memories M0–M6 (`src/apm_select/memory.py`; M6 added last, untested), scenarios S1–S9, runner, tuning,
  analysis, KM survival; local pilot done (power: closed-loop NI needs ≈ 56 pooled lifetimes → N = 60/scenario floor).

## Running at hand-off (self-deleting; hard caps)
| run id | job | cap (UTC) |
|---|---|---|
| 20261012-010834-g1test | G1 confirmatory X + S | 14:49 |
| 20261012-005126-e1s1 | E1 wheeled training (T/V/P) | 14:52 |
| 20261012-005704-e1nerva | E1 NERVA data + training | 14:47 |

## Next steps
1. After the VMs are gone: `python experiments/G_architecture_selection/cloud/launch.py fetch <run>` for each run,
   then `network-down` and `audit` (env vars APM_GCP_*), delete `apm-g/` bucket objects after fetching; update
   `reports/COST_LOG.md`.
2. G1: copy `runs/cloud/<g1test>/runs/G1/{X,S}` to `runs/G1/`, run `scripts/g1_analyze.py --partition X` and `--partition S`,
   commit results; write the G1 report and decision.
3. E1: power from P preds (`F/scripts/e1_analyze.py --part P --power`), write `F/prereg/E1_ADDENDUM.md`, stage 2
   (generate X, retrain with X predictions), analyse.
4. G2: add a test for M6 + `scripts/g2_m6_train.py` (T logs) and probing factor, write `prereg/G2_ADDENDUM.md`
   (K 16, N 6 tuning, NV 12 validation, N_X 60/scenario), run `g2_tune.py` then the test.
5. Then G3 → G8 per `MASTER_PREREGISTRATION.md`; budget approved US$30 (phase 1 cap US$6).
