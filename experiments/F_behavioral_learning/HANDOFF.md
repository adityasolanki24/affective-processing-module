# Handoff: APM v0.2 (programme F) moves from the cloud session to local sessions (2026-10-11)

The v0.2 programme was started in a Claude Code cloud session ("APM — Autonomous Affective Intelligence Research"),
running in a 4-vCPU container, on branch `research/apm-v0.2-experiments-t8kijm`. It continues in local desktop
sessions in the "APM" sidebar group, working in `C:\Users\<user>\OneDrive\affective-processing-module` on this branch.
This file is the continuation point. Read `MASTER_PREREGISTRATION.md`, `prereg/E2_ADDENDUM.md`, `DEVIATIONS.md`
and `AUDIT.md` before doing anything.

## State at handoff (last pushed commit `88c23b0`)

| item | state |
|---|---|
| audit (D1–D7) | done; D1, D2 and D7 fixed with regression tests |
| E0 integrity gate | passed (`results/E0/e0_results.json`) |
| E1 tooling | committed (`e1_data.py`, `e1_train.py`, `e1_analyze.py`); confirmatory results not in the repository |
| E2 | predictor trained and frozen; weights tuned and frozen; Oracle rates frozen; addendum committed (`prereg/E2_ADDENDUM.md`). **Confirmatory run on partition X (N = 240 lifetimes × 9 systems × 10 episodes) not yet run** |
| E3 runner/analysis, E4 trainer (B3/B4 recurrent PPO), E7 runner/analysis | committed; confirmatory runs not done |
| E5, E6 | gated, not started |
| local verification | branch checked out locally; 65/65 tests pass |

## Possible loss to prevent (must be done in the cloud session before it stops)

The E2 confirmatory runner requires the frozen predictor `runs/E2/predictor.json` (SHA-256
`0735d85487ac031cccbb171745ec8e3d5daf6f111d5b3daadc3401b2ed4be314`). `runs/` is git-ignored, so this file (and any
other uncommitted results or logs) exists only in the cloud container. It must be committed (force-add or copy into
a tracked path such as `experiments/F_behavioral_learning/frozen/`) and pushed before the cloud session stops.
Retraining it locally would very likely not reproduce the hash and would be a recorded deviation.

## Rules that continue to apply

- Confirmatory runs only on frozen, committed inputs. Thresholds, rewards, seeds and criteria never change after
  data are seen; every change goes in `DEVIATIONS.md`.
- Seeds only from `seeds.py`. Large raw data under `runs/` (ignored), described by manifests.
- **Compute (§4.5):** the original limit was "this container, 60 CPU-hours, no paid cloud without the user's approval
  of a specific ceiling". Moving to local execution (12 logical CPUs) and/or a cloud VM needs:
  1. a `DEVIATIONS.md` entry, since the machine changes;
  2. for paid cloud, the user's explicit dollar ceiling;
  3. VMs with a hard maximum run duration that delete themselves, results synced, and nothing billable left
     afterwards.

## Next steps (in order)

1. Confirm `runs/E2/predictor.json` (or its committed copy) is present and matches the hash; restore it to
   `runs/E2/`.
2. Record the machine change in `DEVIATIONS.md` (before any confirmatory data).
3. Run E2 confirmatory (partition X) exactly as the addendum specifies; analyse with `scripts/e2_analyze.py --partition X`.
4. Continue with E3, E4 and E7 per the master preregistration, with an addendum committed before each confirmatory run.
5. Final report per §5 (GO / PIVOT / REDESIGN / STOP).
