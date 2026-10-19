# APM v1.0 deviations from PROTOCOL.md

Every departure from the frozen protocol, with its reason. Entries are appended; none is ever removed.

## D1 (2026-10-19): partition T exploration ran short (fitting only; no V or X data affected)

- **Planned (§2).** 48 RANDINTENT lifetimes per robot: T indices 0–11 (warehouse) and 12–23 (home), histories
  risky_B and safe.
- **What happened.** The jobs ran three Isaac processes per VM. Each process ran about 2.5× slower than alone,
  and the jobs hit their timeouts (100–120 min) with these lifetimes complete:
  - Nova Carter: 37 (18 warehouse, 19 home);
  - T1: 30 (15 + 15);
  - NERVA: recorded when its jobs finish.
- **What was done.**
  - The models are fitted on the completed lifetimes, by the same rule; no lifetime is selected or dropped.
  - The fitted files record their lifetime count (`configs/M5_*`).
  - Later stages use more VMs instead of more processes per VM, and progress heartbeats.

## D2 (2026-10-19): NERVA had no body colliders; its exploration is redone (no V or X data affected)

- **Found in NERVA's partition-T data:** 0 lunge contacts in 556 encounters, calm and pushy alike, and a detector
  rate of 0.
- **Cause.** The Open Duck MJCF gives collision geometry only to the two feet. In MuJoCo (programmes K/L), pushes
  were velocity impulses, so this never mattered. With physical people in PhysX, a person passed through NERVA's
  body.
- **Fix** (before any NERVA V or X data). `apm_isaac.nerva.add_body_colliders` gives the trunk, head, neck and upper
  legs convex-hull colliders from NERVA's own visual meshes:
  - mass and inertia stay MuJoCo's;
  - articulation self-collision is off;
  - shins and feet are unchanged, so the gait's ground contacts are as before;
  - the gate battery is re-run with the colliders.
- **Consequence.** The NERVA models from the first exploration (`configs/M5_nerva_*`, 40 lifetimes) were fitted on
  contact-free data. They are superseded by a new NERVA exploration on the same T indices, by the same rule. The
  first fit is kept in git history for the record.
- **Separate finding in the same data:** 19 NERVA falls, all in the home scene, early in an approach, identical in
  paired lifetimes, so deterministic. They are diagnosed separately (D3).

## D3 (2026-10-19): NERVA's comparative data come from its validated MuJoCo backend (no V or X data affected)

- **What happened in Isaac Sim.** With body colliders (D2), NERVA's frozen controller still passes its gate
  battery unchanged: no falls in nominal conditions or with pushes ≤ 0.15 m/s. It fails physical interaction:
  - in the first encounter of an interaction check, a light contact from a person's body (impulse 0.12 N·s) toppled
    it during its approach;
  - in the home scene it cannot make progress on the floor, stepping in place with 0.01 straightness, and falls.

  These are honest limits of NERVA's controller in PhysX with physical people. They are not tuned away; the
  environment is not changed to make NERVA pass.
- **Decision** (the brief's fallback: "keep the MuJoCo version and clearly label the backend").
  - NERVA's comparative experiments (A, B, C, E) run on its validated MuJoCo backend: programme L's world, rendered
    MuJoCo vision and L's NERVA models.
  - They use the same runtime, systems, M seeds, partitions, lifetime design and analysis
    (`scripts/m6_nerva_mujoco.py`; records labelled engine "mujoco 3.3.0").
  - MONOTONE* for NERVA is calibrated on partition V by the same rule.
  - Experiment D (realistic scenes) is not demonstrated for NERVA.
- **NERVA in Isaac Sim** remains in the release for what it does demonstrate (import fixes, gate battery,
  cross-physics agreement, perception, walking in the warehouse and hospital, videos). Every NERVA result names its
  engine.

### D3 correction (2026-10-19, after NERVA's Isaac demonstration lifetimes)

D3 overstated NERVA's interaction failure; it rested on one observed fall. The complete Isaac evidence, all
warehouse unless noted:

| run | encounters | lunge contacts | falls |
|---|---|---|---|
| encounter check (one person) | 1 | 1 | 1 (after a light contact during the approach) |
| demonstration lifetimes D0–D1, DT, change_B and safe | 96 | 4 | 0 |
| partition T exploration, home scene (before body colliders) | 556 (both scenes) | 0 | 19, all home, deterministic |

**Corrected statement.**
- NERVA cannot work in the home scene: deterministic falls, and stepping in place on that floor.
- In the warehouse it walks, perceives and survives most contacts. Its contact fall rate there is not yet
  estimated (1 fall in 5 contacts observed).

The decision to use the MuJoCo backend for NERVA's comparative data stands, for these reasons:
- there is no valid Isaac NERVA model (its exploration data were contact-free, D2);
- the home scene is impassable;
- redoing NERVA's exploration, calibration and test in Isaac was not done in this sprint. It is listed as open work
  in the report.

## D4 (2026-10-19): completion of the X and F runs (after the X data; no threshold, seed or criterion changed)

- **Two T1 LOOKUP lifetimes re-run.** The `X_humanoid_main_LOOKUP` job hit its timeout (LOOKUP lifetimes take
  about 20 min each) before X late_B indices 3 and 7. They were run afterwards with the identical command, seeds
  and system (`X_humanoid_main_LOOKUP_fill3`, `_fill7`). No other lifetime was re-run or dropped. With 18 of 20
  pairs C3 was already decided (p < 10⁻⁴); with 20 it is unchanged.
- **F, identity uncertainty.** The first `--anon-b1` runs failed before any lifetime: the runtime's anonymous track
  id had no station in the Isaac body (KeyError). The harness now maps the track to the physical person at B1's
  station (7ee3d8a); the runs were repeated with the same seeds.
- **F sizes.** §8 asks for 4 lifetimes in the missing-sensor and camera-dropout checks; 2 per robot were run (as
  for the other F checks), for time. F was also run on T1, not only Nova Carter. F was not run for NERVA (D3).
- **F, unexpected contact.** The world does not write its fault log into the rows. IMU detection is therefore
  assessed by comparing each scripted-contact approach with the paired reference approach (same seeds, no fault),
  on approaches without a lunge contact in either (`m6_robustness.py`, `contact_imu`).
