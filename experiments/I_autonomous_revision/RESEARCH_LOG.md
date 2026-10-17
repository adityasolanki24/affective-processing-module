# Programme I research log

### 2026-10-17: pilot (partition P) and design

**Pilot.** The pilot measured information per encounter, with these results:
- collision-only evidence: about 0.005 nats per encounter;
- self-observed near-miss evidence: 1.1–1.5 nats per encounter;
- EM with BIC finds 2 person types from near-miss evidence, but only 1 from collisions;
- closed-loop pilots of weak coupling, keep-away, strict keep-away and S10;
- G2b's memory ranked people inverted (S3 AUROC 0.20).

**Infrastructure.** Cloud runner built. The organization enforces OS Login, so no SSH: jobs go through a bucket queue.
Throughput on a 128-core spot VM is about 1,100 lifetimes per minute, against about 60 per minute on the laptop.

**Design.** The master preregistration was written and frozen before any T, V or X data.

### 2026-10-17: T, V and I1-X
- **T:** EM on A0's own experience. Dense evidence → BIC selects K = 3; collision-only → K = 1.
- **V tuning:** done. Deviation D1: p_random range widened, before any X data.
- **I1-X:** 12,000 lifetimes in 10.6 min on one 128-core spot VM. **R_dense passes C1–C5.**
  - Collisions with careless people fall from 0.071 to 0.051 per minute.
  - Collisions with careful people and idle time are unchanged.
  - G2b's inversion is confirmed.
- **Exploratory (labelled, not in §6):** a confidence-channel ablation (R_dense with F's count-based confidence) on new
  indices X 1000–1199. Purpose: separate the effect of better beliefs from the different confidence signal the
  predictor reads.
