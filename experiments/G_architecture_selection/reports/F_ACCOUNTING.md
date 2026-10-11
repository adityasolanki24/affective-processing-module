# Accounting for unfinished programme-F items (2026-10-12)

Programme F's criteria are not modified. Status after reading the branch and the local data:

| item | state found | what completing it needs | plan |
|---|---|---|---|
| **E1** (does affective machinery add predictive information?) | Tooling committed. Only per-model *metadata* survived the cloud container (`runs/E1/preds/*.json`: validation Brier, size, timing): 103 records — wheeled collision complete (all arms × 5 seeds, LR 1 seed), wheeled stability-loss partial (C, G, R × 3 seeds; RL × 2; others missing), NERVA stability-loss complete. No datasets, no model files, no partition-X data, and **no E1 addendum** (sizes never fixed). | (1) E1 addendum (sizes from the existing validation records only); (2) regenerate T/V (hashes in the committed manifests make this verifiable) and generate X; (3) retrain ≈ 120 models (≈ 230 s each on the container's cores); (4) NERVA arm: the NERVA policy file and MuJoCo setup (the user's NERVA folder is on this machine; the container had a copy). | Run as an F item in the cloud once spending is approved: ≈ 15–25 vCPU-hours (≈ US$0.5–1). It does not block G: G1–G3 test behavioural mechanisms directly, and G3 includes its own predictive probe tests. |
| **E5** reduced (lifelong infrastructure on the B1 predictor) | Gated by E4 (no credible candidate) → reduced version only; not started. | — | Superseded by G7 (lifelong learning with a promotion gate, run on the architecture G4–G6 select). Not run as an F item unless the user asks. |
| **E6** reduced (NERVA predictive transfer) | Gated by E4 → reduced version; depends on E1's NERVA data. | E1 NERVA data | Folded into G7's cross-embodiment study (comparable signals only). |
| E2, E3, E4, E7 | Complete, committed, preserved. F's partition X is retired for selection and testing. | — | — |
