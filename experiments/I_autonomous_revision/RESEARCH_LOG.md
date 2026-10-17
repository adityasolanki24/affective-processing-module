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
