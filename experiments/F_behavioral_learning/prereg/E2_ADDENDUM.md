# E2 addendum: fixed before the confirmatory run (2026-10-11)

Supplements `MASTER_PREREGISTRATION.md` §2 E2. Nothing here changes a hypothesis, metric, criterion or margin.
Written after the pilot (partition P, 40 lifetimes) whose only uses were debugging and the variance estimate
below; the pilot's B2-vs-B1 point estimates were not inspected (the power mode of `e2_analyze.py` prints SEs only).

## Frozen inputs

| item | value |
|---|---|
| predictor | `runs/E2/predictor.json`, SHA-256 `0735d85487ac031cccbb171745ec8e3d5daf6f111d5b3daadc3401b2ed4be314` (5-member MLP ensemble; exploration lifetimes E2-T 0–149, early stopping E2-V 0–39; report `results/E2/predictor_report.json`) |
| tuned weights | `configs/E2_tuned.json`, SHA-256 `aa8bf4b84ebdf0e17a16c03fe2458a9d30650391a757d7dea1bf4d37866c09c4`. B1: λc = 1.0, λs = 2.0. B2: λc = 0.5, λs = 2.0, k_pad = 2.0, k_t = 0.25 (grid of §1.2 on E2-V lifetimes 100–129, 4 episodes, max mean J) |
| Oracle rates | `configs/oracle_rates.json`, SHA-256 `215efe2aff63df76e1ab18098f4ec0d368f626021344c57d6ef2814b44119a8a`: converged per-encounter collision rate by disposition on exploration lifetimes E2-T 200–259 (careful 0.024, careless 0.269). In each lifetime the Oracle's memory feature for a person = the rate of that person's true disposition |
| code | the commit that contains this file; runner `scripts/e2_run.py` refuses partition X unless this file is committed and the tree is clean |

## Design

- **Confirmatory partition X, N = 240 lifetimes** (indices 0–239) × 10 episodes of 60 s, every system on the same
  lifetimes (common random numbers). Sizing (§4.1): pilot SE of the relative collision difference 0.087 and of the
  completion difference 0.0058 at 40 lifetimes; N = max(40·(2.8·0.087/0.10)², 40·(2.8·0.0058/0.03)²) = 237 →
  240 (within the 100–400 bounds).
- **Systems:** B0, B1, B2, B2_noPAD (k_pad = 0), B2_noTonic, B2_noMemoryFrames, B2_noPersistence, B2_shuffled
  (B2's PAD/tendency readouts replayed from lifetime (i + 1) mod N of the same run), Oracle.
- **Analysis:** `scripts/e2_analyze.py --partition X` as committed. Primary: B2 vs B1, criteria (A)/(B) with Holm over
  the two, anti-freezing rule, paired cluster bootstrap over lifetimes (10,000 resamples, seed 0), ratio of totals.
  Secondary (unadjusted, labelled): B1 vs B0, B2 vs B0, Oracle vs B1, each ablation vs B2, severe collisions,
  interventions, idle fraction, person/careless collisions.
- Pre-run code changes since the master preregistration: D7 (`DEVIATIONS.md`), which does not change behaviour on
  unfaulted inputs.

## Decision rule (unchanged)

B2 beats B1 iff (A) or (B) holds with no degenerate freezing. If it does not, the hand-written affective layer has not
justified its complexity in this testbed and is kept as a baseline only.
