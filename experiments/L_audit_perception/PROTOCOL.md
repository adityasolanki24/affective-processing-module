# Programme L protocol: audit corrections, real perception, next milestone

**Governance.**
- This file is **append-only**. Each section is frozen by the commit that adds it, before any data it governs
  exist (partition V or X). A frozen section is never edited. Any later change goes in `DEVIATIONS.md` with a
  reason.
- Seeds come only from `seeds.py` (new entropy, independent of programmes F–K). Partition P is for debugging only.
- Confirmatory runs use `--addendum ../L_audit_perception/PROTOCOL.md` on a clean committed tree.
- Programme K's results are preserved unchanged. Phase A (audit fixes, `results/A*`) is code correction and
  reanalysis, not a test. Everything is simulation only.

---

## B. Corrective recovery test (frozen 2026-10-18, before any L V or X data)

**Question.** Which of programme K's conclusions survive when all the audit corrections are applied together on
fresh seeds?

**Corrections in force in every system** (no legacy flags, except in the two reference systems below):
- A3: the risk cap is enforced; with no option within it, the robot executes HOLD (stops at 0.9 m, beyond reach;
  the task is not attempted; no IMU evidence there);
- A5: the camera's 40° field of view is a full angle (±20°);
- the fixed memory clock (`--transition-fix`);
- A1: confirmed event onset (2 consecutive favourable observations, both observed).

**Pilot basis.** One local pilot (LB3, partition P, 12 lifetimes) checked that HOLD, OBSERVE and the new fields
run. It showed that IMU-only DT holds at 0.88 m once confident in a pushy type. No value, criterion or system was
chosen from it.

### B.1 Experiments

| exp | body | systems | histories | V (safe only) | X |
|---|---|---|---|---|---|
| LB1 | NERVA | L0, DT_STATIC, DT, DT_VOI, DT_UNC, DT_CAM, DT_CAM_VOI, ORACLE, **DT_K**, **DT_CAM_VOI_K** | risky_B, safe, change_B, late_B | 0–29 | 0–49 |
| LB2 | wheeled | L0, DT, DT_VOI, DT_CAM, DT_CAM_VOI, ORACLE | same | 0–29 | 0–49 |
| LB3 | humanoid (T1) | L0, DT, DT_VOI, DT_CAM, DT_CAM_VOI, ORACLE | same | 0–29 | 0–49 |
| LB4 | NERVA | L0, DT | risky_B, safe; tasks waypoint, inspect, handoff, separation | – | 0–39 |

- All bodies, configs, type models, time models, edges, task values and VOI settings are K's, unchanged.
- **DT_K and DT_CAM_VOI_K** are DT and DT_CAM_VOI with K's legacy cap and legacy camera. They run on the same
  fresh seeds, so the effect of the corrections is a paired comparison. They are references, not candidates.

### B.2 Recovery target, calibrated independently

- Per body and system, the target is the mean B1 stopping distance at encounters ≥ 10 in the safe history, on
  partition **V**.
- `l_b_targets.py` writes `configs/LB_recovery_targets.json`. That file is committed **before** any X analysis
  and never changed afterwards.

### B.3 Criteria (LB1–LB3; `l_b_analyze.py`)

As in K §4, with corrections:

| | criterion |
|---|---|
| P1 | change_B: KM median pushy → calm revision ≤ 6 B1 meetings |
| P2-task | change_B: KM median ≤ 8 B1 meetings until the B1 stopping distance is ≤ 0.35 m (handoff 0.30 + 0.05) for 2 consecutive meetings |
| P2-V | change_B: KM median ≤ 8 B1 meetings until it is within 0.05 m of the V-calibrated target for 2 consecutive meetings |
| P3 | late_B: KM median calm → pushy revision ≤ 4 B1 meetings |
| P4 | 0 falls; trips ≤ 1% of encounters; repeated-push encounters not above DT's (paired upper bound ≤ +0.02, change_B and late_B); **0 executed approaches with predicted risk above the cap** (recomputed from the logged belief) |
| P5 | navigation arrival (at the commanded stop, a hold included) ≥ 0.99 |

- **The recovery milestone is met** by a candidate only if P1, P2-task, P2-V, P3, P4 and P5 all pass.
- **Candidates:**
  - IMU only: DT_VOI on all bodies;
  - camera extension: DT_CAM_VOI on all bodies.
- DT and DT_CAM are judged the same way, for reference.
- **Task outcomes are reported separately, not as criteria:**
  - interaction reached: stopping distance ≤ 0.35 m;
  - interaction hold completed: within 0.35 m throughout the first sensing window;
  - hold (no safe action) rate.

**LB4** (K §5, with measured success):
- Per encounter, task cost = approach time + 4 s × (windows − 1) + 20 s × strong perturbations
  + 12.6 s × (1 − success).
- Success is measured:
  - inspect and handoff: hold completed;
  - separation: stopping distance ≥ 0.5 m;
  - waypoint: arrival.
- DT is promoted for a task if:
  - cost difference: 95% upper bound < 0;
  - contacts: upper bound ≤ +0.05;
  - 0 falls.
- DT replaces L0 if it is promoted in ≥ 3 of 4 tasks.

### B.4 Verdicts on programme K's claims (rules fixed now)

| claim (K) | survives | weakens | fails |
|---|---|---|---|
| C1 NERVA camera + VOI meets the milestone | DT_CAM_VOI milestone met | P1, P3, P4, P5 met and exactly one P2 variant met | otherwise |
| C2 NERVA IMU only meets it (at the margin) | DT_VOI milestone met | as C1 | otherwise |
| C3 wheeled: camera + VOI meets, IMU only does not | DT_CAM_VOI met and DT_VOI not met | DT_CAM_VOI weakens (as C1) | DT_CAM_VOI not met, or DT_VOI met |
| C4 humanoid: same as C3 | as C3 | as C3 | as C3 |
| C5 safe information gathering cuts contacts per pushy meeting (NERVA ~3×) | DT_CAM_VOI / DT ratio ≤ 0.5, bootstrap upper bound < 1 | upper bound < 1, ratio > 0.5 | upper bound ≥ 1 |
| C6 the decision-theoretic mapping beats H's lookup (4 of 4 tasks) | promoted in 4 of 4 | 3 of 4 | ≤ 2 |
| C7 caution returns within 1–2 meetings (calm → pushy) | candidates' P3 median ≤ 2 | ≤ 4 | > 4 |

- **Reported next to C5, never as a verdict.** The pilot showed that a holding robot avoids contact by abandoning
  the task. So the realised task cost per B1 meeting (LB4's cost with measured success; a hold costs its time plus
  the 12.6 s retry) is reported for DT and DT_CAM_VOI, with paired bootstrap CIs.
- C5 is judged on LB1 (NERVA) and reported for LB2 and LB3. Its ratio is pooled across histories, over lifetimes,
  with bootstrap CIs.
- C1–C4 are judged per body.
- The verdicts are mechanical. Explanations, such as "IMU silence at the hold distance is not evidence", are
  reported separately from the verdicts.

### B.5 Compute

- Cloud spot VMs (`jcloud.py`), under programme L's US$60 cap. A preempted job is rerun with identical seeds, and
  the rerun is logged.
- Raw lifetimes are git-ignored under `runs/L/` and hashed in the analysis output.
