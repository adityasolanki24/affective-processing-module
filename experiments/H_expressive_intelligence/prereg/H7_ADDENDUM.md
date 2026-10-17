# H7 addendum: cross-embodiment expression (fixed before H7 test data, 2026-10-12)

## Question and limits

Can one embodiment-independent `ExpressionIntent` produce *corresponding* observable qualities on two different
bodies?
- **Biped:** NERVA, MuJoCo, with the H0b/H1 interface; data from H1's X core arm.
- **Wheeled:** a kinematic differential-drive base with acceleration limits (`apm_express.wheeled`), with no head and
  no torso.

**Limits stated in advance:**
- The wheeled body is kinematic, not a physics simulation.
- There is no human evaluation, so "interpretable on both bodies" is **untested**. Only correspondence of measured
  motion qualities is tested.
- No learned transfer is claimed. Both bodies are driven through hand-designed adapters.

## Systems (brief items 1–4)

1. **Robot-specific expression systems.** Not built separately. With no perception data, a separately hand-tuned
   wheeled style set would be the designer's own judgement and could not be compared on any external criterion. It
   is reported as not evaluable.
2. **Shared abstract state + embodiment-specific adapters** = **4. a hand-designed expression contract.** This is
   the `ExpressionIntent` architecture; it is evaluated here.
3. **Shared learned latent with output heads.** Not run. No supervision exists that would make such a latent mean
   anything (see the H1 X3 reasoning). Training it to reproduce the adapters would just copy them.

## Data

- **Wheeled:** X0 + the six X1 styles × the 8 H1 geometries × 5 seeds (seeds `sim_seed("H7", "X", i)`), 60 s.
- **Biped:** H1 X core arm (280 trials).

## Measures

**Shared abstract features:** features computable on both bodies with the same code:
- progress_speed, pauses_per_m, mean_pause_s, switch_rate, backward_s;
- straightness, max_lateral_dev, final_distance, final_bearing_abs, body_yaw_rest_std, retreat_m.

| code | measure |
|---|---|
| C1 (within-body distinctness, wheeled) | leave-one-geometry-out shrinkage LDA over the shared features excluding speed (progress_speed removed); balanced accuracy |
| C2 (correspondence) | for each shared feature, Spearman ρ across the 7 styles between the biped's and the wheeled body's per-style means; reported per feature, with the median ρ as the summary |
| C3 (cross-body transfer) | LDA trained on biped trials, tested on wheeled trials, after z-scoring each body's features within that body; balanced accuracy (chance 1/7) |

## Criteria

**H7 supports "the abstract intent is realised with corresponding qualities on both bodies" if all of:**
- C1 ≥ 0.80;
- median C2 ρ ≥ 0.6;
- C3 ≥ 0.50.

**Otherwise the claim is restricted to "the interface accepts both bodies".** Interpretability is untested in
every case.
