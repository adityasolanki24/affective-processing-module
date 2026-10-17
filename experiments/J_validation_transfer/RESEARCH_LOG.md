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

### 2026-10-17: J1 result, J2-T/V, J3 pilot 1
- **J1-X (14,000 lifetimes):** I1 survives frozen calibration. R_dense adapts in 2/3/2 (decisive 3/4/2) and passes C2–C5.
  - The decay-only control fails decisive adaptation everywhere: the metric is valid.
  - The decay-only control also cuts avoidable collisions by 55%: the safety gain is not caused by revision.
  - Report: `reports/J1_REPORT.md`.
- **J2-T:** EM on mixed-driver experience learns a "person type" (r = 0.60) that is the always-GO driver's collisions,
  i.e. a policy artefact. Disclosed in the J2 addendum before the test.
- **J2-V selection:** EC. Calibration is frozen.
- **J3 pilot 1** (1,350 standing NERVA encounters, 3 entity types × 3 distances):
  - Pushes inside the 0.10 m/s envelope are hard to sense; most pilot contacts were tiny (median 0.023 m/s).
  - The a-priori gyro detector (0.35 rad/s) never fired: a design flaw.
  - Graded peak-gyro coding: 0.1–0.3 nats per encounter only at 0.2 m; ≈ 0.05 at the neutral 0.4 m.
  - EM on 6-encounter units: K = 1.
  - Pushes ≥ 0.05 m/s *are* sensed: peak acceleration AUROC 0.99, tilt 0.93, gyro 0.91.
  - **The J3 gate fails on pilot 1.**
- **J3 pilot 2** (disclosed design iteration, P partition, new indices):
  - evidence v2: a self-calibrated detector from the robot's own quiet settle window, filtered impact energy;
  - a 4th type, "pushy": frequent moderate pushes, within the envelope;
  - 12-encounter EM units;
  - EM codings: acc8, energy8, tilt8.

  The pilot-1 result stays on record. Types whose differences lie below the sensing range are, by construction, not
  learnable from experience.
