# Deviations from MASTER_PREREGISTRATION.md

Every change to the preregistered plan is recorded here, dated, with its reason and whether it happened before or
after the affected data were seen.

## 2026-10-11 — testbed calibration during E0 (before any B1/B2 comparison data)

Made on E0 pilot runs of fixed-action policies (always GO / SLOW / YIELD / DETOUR), before any system comparison:

| item | preregistered | implemented | reason |
|---|---|---|---|
| goal deadline | 3 s + d / 0.4 m/s | 4 s + d / 0.35 m/s | at 0.4 m/s even an unobstructed robot missed many deadlines among people |
| safety brake | 0.15 m clearance, ±70° | 0.05 m clearance, ±60° (last resort) | at 0.15 m the brake prevented ≈ all collisions under every policy, leaving no behavioural question |
| severe collision | robot speed ≥ 0.4 m/s | ≥ 0.2 m/s | navigation already slows to ≤ 0.24 m/s near people, so 0.4 never occurred |
| careless pedestrians | "keep going and drift toward the robot when close" | cut across a *moving* robot's path (aim 0.6 m ahead of it); all pedestrians walk around a *stopped* robot | the first version walked into a stationary robot, so collisions were unavoidable by any action |
| navigation speed near people | v0.1 rule (nearest entity) | obstacle and person proximity factors separate (person factor ≥ 0.3 within 0.4 m) | people were slowing the robot to a crawl everywhere |

None of these change a hypothesis, metric, criterion or seed.

## 2026-10-11 — D7 found in the E7 pilot (before any E2/E7 confirmatory data)

| item | change | reason |
|---|---|---|
| `apm/adapters/wheeled_ctrl.py` | a TTC hypothesis is emitted only when `ttc_p > 0` **and** closing speed `> 0` | E7 pilot (fault "conflicting predictions", partition P) showed a `ZeroDivisionError` in the v0.2 adapter when an estimate arrives without closing motion; the exception reached the harness and B2 fell back to GO for the rest of the episode. Under unfaulted inputs `ttc_p > 0` implies closing > 0, so E0–E2 behaviour is unchanged. Regression test `test_d7_…`. |

## 2026-10-11 — E2 confirmatory run started in the cloud container and aborted (no data)

`scripts/e2_run.py --partition X --n 240` was started at 10:27 UTC (commit `88c23b0`, lock passed) in the 4-vCPU
cloud container and terminated at the user's request to stop that session, before it wrote any output
(`runs/E2/X/` never existed; the runner writes only at the end). No partition-X result was produced or seen. The run
is to be executed once, in full, by the continuing session (see `HANDOFF.md`). E1 wheeled training was also stopped:
collision models complete except those listed missing in `runs/E1/preds/`; stability_loss models partial. Training
resumes with `--skip-existing`; finished models are unaffected.

## 2026-10-11 — machine and compute-limit change for the remaining programme (before any partition-X data)

Recorded before any confirmatory (partition X) data of any experiment exist; the only partition-X run so far was
aborted without output (entry above).

| item | preregistered (§4.5) | now | reason |
|---|---|---|---|
| machine | the 4-vCPU cloud container | pilots and debugging: the user's Windows 11 PC (12 logical CPUs, Python 3.12.4, NumPy 2.5.4, torch 2.14.1+cpu). Confirmatory runs: one temporary Google Cloud CPU VM (Linux x86-64, Python 3.12, same NumPy/torch versions where installable), hard maximum run duration with automatic deletion | the cloud container was stopped at the user's request (`HANDOFF.md`) |
| compute limit | 60 CPU-hours in the container; no paid cloud without an approved ceiling | the user approved paid cloud with a hard ceiling of **US$10 total** for the remaining programme-F compute (`HANDOFF.md`, 2026-10-11); a large VM for ≤ 1 h can exceed 60 vCPU-hours, so the dollar ceiling replaces the CPU-hour figure | user decision |
| worker count | E2 runner fixed at 4 worker processes | `scripts/e2_run.py --workers N` (default 4) | parallelism only: `Pool.map` returns results in job order and every job is seeded from `seeds.py`, so outputs do not depend on N. The E2 addendum's "code = the commit that contains this file" therefore becomes "that commit plus this change" |

Floating-point results may differ in the last bits between machines/BLAS builds; no criterion depends on that, and
all systems of one experiment run on the same machine. The frozen E2 predictor was restored from the committed copy
`runs/E2/predictor.json` (force-added in `8601f6e`), SHA-256 verified equal to the addendum's
`0735d854…be314`; it was not retrained.

## 2026-10-11 — pilots on a cloud VM; code added or changed before the E3/E4/E7 addenda (no partition-X data)

- Local pilots on the user's laptop were stopped after ≈ 15 min (each process got ≈ 20% of a core) and discarded
  unused. All pilots (E3, E3 retention, E4 learning curves and pilot evaluation, E7; partition P / E4-T/V only)
  ran on one temporary VM (n2d-highcpu-64, 21 min, self-deleted; `cloud/pilot.sh`).
- **E4:** `scripts/e4_eval.py` and `scripts/e4_analyze.py` (no test runner or analysis existed) and
  `stats.paired_lifetimes_two_level` (the two-level bootstrap §2 requires; regression-tested to equal the
  one-level bootstrap with one seed); `e4_train.py --collect-workers` (parallel rollout collection, verified
  output-identical). Details fixed in `prereg/E4_ADDENDUM.md`.
- **E7:** the runner counted deliberately injected `model_failure` exceptions together with genuine policy
  exceptions; they are now counted separately (`injected`) and `e7_analyze.py` reports criterion (1) (genuine
  exceptions = 0) and criterion (2) evidence (fallbacks under `model_failure`). Instrumentation only; behaviour is
  unchanged. The pilot showed that all counted exceptions were injected ones (`prereg/E7_ADDENDUM.md`).
