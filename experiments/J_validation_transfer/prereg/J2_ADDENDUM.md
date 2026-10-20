# J2 addendum: fitted models, frozen calibration, selection and test run (committed before any J2-X data, 2026-10-17)

This addendum supplements `MASTER_PREREGISTRATION.md` §3. Nothing in it relaxes a criterion.

## Fitted on J2-T (nominal S3; drivers A0, B0, cautious, R1, 100 lifetimes each; no labels)

**Models** (git-normalised SHA-256 prefixes):

| model | file | hash |
|---|---|---|
| EU-T | `configs/types_J2T_EU-T.json` | `152894b4` |
| EA | `configs/types_J2T_EA.json` | `c7425c68` |
| EC | `configs/types_J2T_EC.json` | `69da0d83` |

All three select K = 3 by BIC: π ≈ (0.565, 0.327, 0.108), r ≈ (0.008, 0.027, **0.598**).

**Disclosed before testing.** The third type, with a collision rate of 0.60 per encounter, is almost certainly the
always-GO driver's collisions with careless people. EM has learned part of the robot's own policy as a person type,
and conditioning on the robot's speed band (EA) does not remove it. This is exactly the confound §3 asks about.

In J2-V calibration under the B0 driver, EU-T, EA and EC place identities around p ≈ 0.27 (θ = 0.27). EU-I, fitted
only on A0's experience, stays at θ = 0.017. The models enter the test unchanged, as preregistered.

## Frozen from J2-V

- **Calibration:** `configs/J2_calibration.json` (`9b3625f2`), per condition × system, and per policy-group observer.
  Heavy overlap (`overlap_40`) nearly erases the careless–careful gap for every memory: gaps 0.002–0.005.
- **Selection (§3.3)**, by mean dense predictive log-likelihood over all J2-V closed-loop lifetimes:

  | model | log-likelihood |
  |---|---|
  | EU-I | −1.2659 |
  | EU-T | −1.2797 |
  | EA | −1.2745 |
  | EC | −1.2666 |

  **Selected: EC.** Survival is therefore evaluated for **EU-I** and **EC**.

## Test (J2-X; indices 0–199)

**System files:** `configs/J2_systems.json` (`7ddaf379`), `configs/J2_policy.json` (`38bb1682`).

**Closed loop:** A0, EU_I, EU_T, EA, EC × 13 conditions × S1, S2, S3, S5, S10 = 65,000 lifetimes. Split by condition
across three VMs:

| group | conditions |
|---|---|
| A | nominal, overlap_25, overlap_40, erratic, drift |
| B | identity_uncertain, sensor_noise_0.10, sensor_noise_0.20, crowd_dense |
| C | obstacles_dense, fast_pedestrians, speed_limit_0.6, robot_fast |

**Policy group:** the 12 observers × S1, S2, S3, S5 = 9,600 lifetimes.

**Execution:** every job is submitted with `--commit --addendum J2_ADDENDUM.md`.

**Analysis:** `j2_analyze.py test`, with the criteria exactly as in §3.3.
