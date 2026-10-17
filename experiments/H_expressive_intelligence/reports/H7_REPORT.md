# H7 report: cross-embodiment expression (2026-10-12)

## Verdict: correspondence **supported** for measured motion qualities; interpretability untested; no learned transfer

**Setup:**
- Governed by `prereg/H7_ADDENDUM.md`;
- results in `results/H7/h7_results.json`;
- biped data: H1 X core arm (NERVA, MuJoCo, 280 trials);
- wheeled data: a kinematic differential-drive base, 280 trials over the same 8 geometries.

| measure | result | criterion |
|---|---|---|
| C1 wheeled distinctness (shared features, no speed) | **1.00** | ≥ 0.80 ✓ |
| C2 median cross-body correspondence (Spearman over 7 styles) | **0.71** | ≥ 0.6 ✓ |
| C3 biped → wheeled transfer (body-wise z-scored) | **0.79** (chance 0.14) | ≥ 0.50 ✓ |

### Correspondence per feature (ρ across styles)

| strong (≥ 0.9) | moderate (0.6–0.9) | weak (< 0.6) |
|---|---|---|
| speed 0.96, pause length 0.96, reverse time 1.00, stopping distance 1.00, body-yaw scanning 0.93 | path straightness 0.71, lateral deviation 0.68 | pauses per metre 0.45, switch rate 0.59, retreat distance 0.53, final bearing 0.32 |

## Reading

- The same abstract `ExpressionIntent` yields the same *ordering* of styles on two bodies for most qualities.
- Qualities tied to embodiment-specific channels correspond poorly:
  - **gaze:** the biped averts its head while the wheeled base turns its body;
  - **pause counting:** the biped's tempo rhythm creates stop-go pauses, while the wheeled base modulates speed
    continuously.
- The architecture is BML-like (literature review §11): a shared, hand-designed contract plus per-body realisers.

## Not shown

- **That people read the same quality on both bodies.** No human data.
- **Learned transfer.** Both adapters are hand-designed. Nothing was learned on one body and transferred.
- **Physics on the wheeled body.** It is kinematic.
- **Robot-specific expression systems (brief item 1) and a shared learned latent (item 3):** not evaluable without
  an external criterion. See the addendum.
