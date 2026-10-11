# Programme G status (2026-10-12, 17:55 UTC) — complete

All experiments G0–G9 and programme F's E1 are complete, analysed, committed and pushed on
`research/apm-v0.3-architecture-selection`. **Final decision: A — simple predictive regulator**
(`ARCHITECTURE_DECISION.md`).

| item | state |
|---|---|
| G0 fair-comparison framework | done; 120/120 repository tests pass |
| G1 PAD vs simple regulators | done; G9 re-test governs (PAD +99% avoidable collisions; EMA regulator −47% vs static) |
| G2 / G2b memory | done; no memory meets the adaptation criterion; best design = exposure gating + time decay + probing |
| G3 persistent predictor state | done; no state required (fast+slow −38% avoidable, secondary) |
| G4 / G5 tournament and training | done; no neural candidate beats A0; BC→PPO ≫ PPO from scratch |
| G6 robustness | done; A0 + revisable memory most robust; G9 eligibility underpowered |
| G7 lifelong learning | done; no validated weight learning (gate blocked all updates) |
| G8 minimisation | done; G9: no A0 component removable |
| G9 validity re-test (defect D8) | done |
| F's E1 | done; PAD/appraisal add no predictive information |
| Cloud | ≈ US$13.3 of the approved US$30 (estimate; reconcile in the billing console); teardown audit clean |

Open follow-ups (each needs its own preregistration): revisable memory as default with a fixed calibrated penalty;
slow context state in the predictor; safety-constrained learned efficiency layer; powered promotion gate; a testbed
that does not spawn the robot overlapping people.
