# Programme J research log

### 2026-10-17: audit and set-up
- **Audit.** Programme I's branch is clean and pushed (`fb41fb1`); PR #3's CI is green (6/6); PR #3 is not merged.
- **Issue J1.** Programme I's adaptation threshold was calibrated on the confirmatory S3 lifetimes, using labels.
- **NERVA's robot-observable signals:** IMU gyro and accelerometer (with the training noise model), joint
  positions and velocities, foot contacts, and actuator effort. `base_linvel` is simulator truth, and programme H's
  outcome used it: that is noted as a limitation of H.
- **Benchmarks:**
  - wheeled: about 4 s per lifetime per core;
  - NERVA: about 20× real time per core, so 1–2 s per closed-loop encounter.
- **Cloud.** The user approved US$40 for programme J.
- **Decision read-out.** Built from a bank on J1-T. Decision A's rule moves its careful-action share only from 0.45 to
  0.49 across realistic beliefs, and responds more to confidence than to p.
- **Smoke run (J1-P).** The observed-decision measure is confounded by the partner's behaviour: every system
  "adapts". The read-out is not confounded: the decay-only control never adapts.
