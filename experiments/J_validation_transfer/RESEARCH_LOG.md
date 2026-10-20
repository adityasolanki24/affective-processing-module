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

### 2026-10-18: J2 result, J3-T gate, J5 pilot findings
- **J2:** not promoted (`reports/J2_REPORT.md`). D1 analysis-bug fix.
- **J3-T gate passes:**
  - graded acc8 carries 0.55 nats; the a-priori detector 0.001; falls 0; the self-calibrated detector 0.13;
  - EM finds K = 2 without labels;
  - NERVA type models fitted (J4 unconditional; J5 conditioned on stopping distance).
- **J5 pilot 1 (no gate):** in the closed loop the robot **mistook its own expressive motion for being pushed**.
  - After arrival, gaze turns, scanning and head motion read as high-acceleration categories even with zero contacts.
  - Beliefs saturated at about 0.5 for everyone, and closed-loop history AUROC was 0.57.
  - The frozen-expression robot learned the history (AUROC 1.00).
  - Expression still changed outcomes: −0.63 contacts per encounter vs frozen.
- **Probe:** head motion alone raises the IMU peak to 1.1–1.9 (still: about 0.45); body turns to 7–10.
- **Fix 1, efference-copy gate:** exclude frames during, and 0.6 s after, the robot's own commanded motion.
  - With the gate alone, L0-expressive robots had **zero** quiet frames: continuous scanning makes the robot unable
    to feel anything.
- **Fix 2, sensing pause:** after arrival the robot holds still for the 4 s observation window, then expression
  resumes. Belief still sets stopping distance, tempo, hesitation and path.
  - Expressive movement competes with perception. An expressive robot needs sensing pauses.
  - Both fixes are pilot-stage and disclosed in the J5 addendum. Pilot-1 data are kept.
- **With both fixes:** calm encounters read 0.34–0.54, pushy contacts 0.57–0.89, and every frame is observable.
  - A cautious robot stopping at 0.6 m gets few contacts, even from a pushy entity: the physical avoidance trap.

### 2026-10-18 (later): J5-X, J4 and J6
- **J5-X:** the loop is not demonstrated with L0 (`reports/J5_REPORT.md`). J6 runs as exploratory (D2).
- **J4:** the bank was re-run (J4bank2) after the spot loss.
  - λ = 0.005 selected on V (flat objective).
  - A recurrent baseline (`GruMemory`) was pre-specified and trained on V before X.
  - J4-X fails C2 (compressed readout) and C3 (slow, asymmetric revision). C1 and C4 pass.
  - The triggered recurrent baseline does not rescue it (`reports/J4_REPORT.md`).
- **J6-T fitting:** the self-learned time model has R² 0.905.
  - The decision-theoretic mapping turned out **belief-independent**: stopping far and fast minimises both time and
    perturbation, for every belief and price (D3).
  - A disclosed proximity value (price 40, value 5) gives a belief-dependent mapping (0.3 / 0.4 / 0.6 m).
  - GRU training was aligned with run time (D3).
- **Video tooling:** render hooks verified not to change the simulation (identical evidence, outcome and truth).
  `j_videos.py` re-simulates median J6-X pairs and checks every row against the log.
- **Cloud:** apm-j-1 reached its cap and self-deleted; the us-central1 NAT and router were removed.
