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
