# G8 addendum: minimise A0 and decide (fixed before any G8 data, 2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G8. Written before the G4/G6 test results are known; it covers the A0
family, which is the bar every neural candidate must beat. If G4 and G6 select a neural architecture, its own
ablations are added in an amendment before their data exist.

## Variants (`configs/G8_variants.json`)

| variant | removes / changes | question |
|---|---|---|
| A0 | — (G1-selected EMA regulator, F's 5-member predictor ensemble, F's entity memory) | reference |
| A0_static | persistent regulator → static penalty | is the persistent state needed? (G1 suggests not) |
| A0_nomem | explicit entity memory unused (columns zeroed) | is identity memory needed? |
| A0_single | ensemble → one member | is ensembling (averaged uncertainty) needed? |
| A0_heur | learned predictor → transparent hand-written consequence model (`apm_select/minimal.py`) | is *neural* prediction needed? |
| A0_mem | memory → G2b's best design (exposure-gated, time decay, uncertainty-guided probing) | assembly: does better memory help? |

PAD, appraisal features, slow/fast recurrent states, learned utility corrections and continual weight updates are
already absent from A0 (removed by G1, E1/G3, G3, G4 and G7 evidence respectively).

**Equal retuning:** every variant (A0 included) gets K = 48 Latin-hypercube (λc, λs) values (λc 0.1–10 log, λs 0–4,
seed 20261014) on G8-V lifetimes 0–29 × 10 episodes; selection by mean J (`scripts/g8_tune.py`).

## Test (partition X, then S)

- Stage 1, G8-X lifetimes 0–279 × 10 episodes (N from G1's sizing rule for the same non-inferiority margins), all
  variants + B0; `scripts/g1_run.py --experiment G8 --addendum G8_ADDENDUM.md --no-controls`.
- **Primary family** (Holm over five p-values): for each removal variant (static, nomem, single, heur), non-inferiority
  to A0 (+10% collisions, −2 pp completion) — non-inferior ⇒ the component is *not justified* and is removed;
  for A0_mem, superiority over A0 (E2 rule, p = min of p_A, p_B entering Holm) — superior ⇒ the memory design is added.
- Stage 2 (confirmation of the combination), G8-S lifetimes 0–279: **A0_min** = A0 with every unjustified component
  removed and A0_mem's memory added iff it was superior; non-inferiority to A0 (same margins, single test). A0_min's λ
  is retuned with the same K = 48 procedure on G8-V lifetimes 100–129 before G8-S runs.
- Complexity: parameters, decision latency (decide + observe µs), memory state per identity, training cost.
- Final selection (master §8): A–E; the simplest variant consistent with the evidence.
