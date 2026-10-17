# G7 — Lifelong learning of A0's consequence predictor (2026-10-12)

Preregistered in `prereg/G7_ADDENDUM.md`; run once (commit `f23da27`, lock passed; `results/G7/g7_X.json`): three
environment phases (nominal → faster, more careless pedestrians → slippery and crowded), 60 experience lifetimes per
phase per system, 3-seed predictor ensembles, frozen suites of 150 lifetimes per phase.

## Updates

| system | PH1 | PH2 | PH3 |
|---|---|---|---|
| L2 gated (validation: 20 lifetimes, J ≥ current and collisions ≤ 1.05 × current) | rejected (J +19.0, collisions 11 vs 10) | rejected (J +1.3, collisions 33 vs 26) | rejected (J +11.3, collisions 12 vs 11) |
| L2c corrupted (30% flipped labels) | rejected (J −258) | rejected (J −196) | rejected (J −215) |

No update was promoted → by the preregistered criterion **lifelong learning did not succeed** (rejected updates are
not learning).

## Held-out effects (vs frozen L0, paired lifetimes)

| system, after phase | current-phase suite: J | collisions | completion | first-phase suite: J | collisions |
|---|---|---|---|---|---|
| L1 (replay, no gate), PH2 | **+16.2% [+9.4, +23.2]** | **−30.2% [−37.6, −22.1]** | +1.4 pp | +3.2% [+1.5, +4.9] | +3.5% [−8.5, +18.4] |
| L1, PH3 | +10.2% [+6.6, +13.9] | **+7.4% [+1.0, +15.0]** | +2.1 pp | +2.4% [+0.4, +4.5] | +12.3% [0.0, +28.9] |
| L3 (latest phase only), PH2 / PH3 | −14% / −291% | −33% / ×28 | −8.7 / −16.3 pp | −39% / −114% | ×2.3 / ×28 |

## Findings

1. **Retraining with replay really does learn**: in the faster/more careless phase the updated predictor cut
   collisions by 30% and raised J by 16% on frozen held-out lifetimes. In the slippery/crowded phase it raised J by
   10% but increased collisions by 7%, outside the +5% safety margin, and on the original phase collisions drifted
   up (CI upper +18% to +29%).
2. **The promotion gate is safe but, at 20 validation lifetimes, too noisy to be useful**: it rejected the PH2 update
   that the frozen test shows was strongly beneficial, as well as every corrupted update. Safe gating needs either
   much larger validation sets or a statistical test with an explicit error budget; this is a concrete design item.
3. **Replay is essential**: retraining on the latest phase only destroyed the predictor (collisions ×28).
4. Consistent with F's Milestone C finding ("shadow learning safe but nearly a no-op").
