# E4 addendum: fixed before confirmatory training and evaluation (2026-10-11)

Supplements `MASTER_PREREGISTRATION.md` §2 E4. Nothing here relaxes a criterion. Written after the pilot (one
training seed of B3 and of B4 to 1.2M decisions on E4 partitions T/V, evaluated greedily on 40 lifetimes of E4
partition P; temporary cloud VM). Pilot uses: debugging, learning curves for the budget, SEs for the test size. Only
the power mode of `e4_analyze.py` (SEs, required sizes) was run on the pilot evaluation.

## Frozen inputs

Predictor `runs/E2/predictor.json` (SHA-256 `0735d854…be314`); λc, λs of the predicted-consequence utility = the
tuned B1 values in `configs/E2_tuned.json` (`aa8bf4b8…c09c4`); B2 reference with its tuned values. Code: the commit
that contains this file (`scripts/e4_train.py`, `scripts/e4_eval.py`, `scripts/e4_analyze.py`,
`apm_research/rl.py`, `apm_research/stats.py`). `e4_eval.py` refuses partition X unless this file is committed and
the checkout is clean, and records the SHA-256 of every evaluated checkpoint.

## Training (implementation details fixed here)

- Recurrent PPO as committed in `e4_train.py`/`rl.py`: training lifetimes of 5 × 60 s episodes from E4 partition T
  (indices seed × 10⁶ + k), 8 lifetimes per update, Adam lr 3·10⁻⁴, PPO settings as in `ppo_update`; validation every
  10 updates on E4-V lifetimes 0–7 (greedy, J); the checkpoint with the highest validation J is kept (§4.3).
  Rollout collection may run in parallel processes (`--collect-workers`); every lifetime is seeded independently, so
  results do not depend on that number (verified identical on a short run).
- **Budget: 1,200,000 decisions for every variant and seed** (equal environment-step budgets). Pilot validation-J
  curves: B3 flat from initialisation (≈ 26–29 throughout 0–1.2M); B4 rising from ≈ 3 to ≈ 31 and not clearly
  converged at 1.2M. 1.2M is the largest budget covered by the pilot and fits the compute ceiling; B4 may still be
  under-trained at this budget, which is reported as a limitation (it can only favour B3 against B4, never against
  B1).
- Variants × seeds 0–4: B3, B4 (capacity-matched within ±10%: 14,847 vs 15,185 parameters), B3_noMemory,
  B3_noPersist, B3_fastOnly, B3_slowOnly, B4_half, B4_double. Evaluation-only negative controls on the B3 networks:
  identity-shuffled memory (`B3_shuffledIdentity`) and history reset at every decision (`B3_historyReset`).

## Test

- **E4 partition X, N = 400 lifetimes** (indices 0–399) of 5 × 60 s episodes (the training/validation length), every
  system on the same lifetimes; learned policies act greedily. B1 and B2 (deterministic) are evaluated once.
- **Sizing (§4.1, bounds 120–400):** pilot SEs (one training seed, 40 lifetimes) required 414 (B3 vs B4) and 506
  (B3 vs B1) lifetimes for 80% power at r = −10% / ±3 pp; the cap applies (N = 400), and achieved power is
  reported as below 80% for those effect sizes.
- **Analysis:** `scripts/e4_analyze.py --partition X`. Two-level paired bootstrap (lifetimes shared by both systems,
  then each system's training seeds with replacement; 10,000 resamples, seed 0), ratio of totals. E2 rule per
  comparison: (A) collision r ≤ −10% with p < 0.025 and completion-difference CI lower ≥ −3 pp, or (B) completion
  difference ≥ +5 pp with p < 0.025 and collision-r CI upper ≤ +5%; and idle-fraction difference ≤ +5 pp.
  **Holm over the four one-sided p-values** (A and B for each of B3 vs B4 and B3 vs B1) — stricter than Holm over the
  two comparisons. B3 is a credible candidate iff it beats both. If it beats B1 but not B4: "no evidence for a
  specialised regulation structure". Secondary (unadjusted, labelled): B3 vs B2, B4 vs B1, B2 vs B1, each B3
  ablation/control vs B3, B4 sizes vs B4.

## Gates

E5 (full) and E6 (full) run only if B3 is a credible candidate (§2 E5, E6).
