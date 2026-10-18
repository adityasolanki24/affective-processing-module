# APM v1.0 release: milestone tracker

This is the only tracker for the final sprint. The final account is `REPORT.md`. Confirmatory tests are frozen in
`PROTOCOL.md` before their data exist.

**Branch:** `release/apm-v1.0`, from `research/apm-v0.8-audit` at 84aeef9. Nothing is merged without approval.

**Cloud cap:** US$500, approved 2026-10-18. Spending is logged in the cost section of `REPORT.md`.

**Pinned stack:**
- Isaac Sim 5.1.0 (pip wheel, Python 3.11, Ubuntu 22.04);
- Isaac Lab v2.3.2;
- GCP L4 (Ada, RT cores, 24 GB);
- NVIDIA driver 580 (Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580`).

Not chosen:
- Isaac Sim 6.x: Isaac Lab's stable branch still targets 5.1.
- Isaac Lab 3.0: early access.
- A100 and H100: they lack RT cores.

| # | milestone | gate (must pass before the next safety-critical step) | status |
|---|---|---|---|
| M1 | freeze and stabilise v0.8 | clean integration test (runtime → sensors → memory → decision → bounded command → safety veto → motion → persist/restart → replay → AffectLink) passes from a clean checkout; L's open defects fixed (task-scoring artefact, falls characterised) | in progress |
| M2 | Isaac Sim infrastructure | pinned env restores on a fresh VM; headless render of a stock scene; physics/sensor smoke test | |
| M3 | three embodiments in Isaac | each robot imported, controller validated by a battery (stand, walk, stop, turn, pushes); MuJoCo vs Isaac characterised | |
| M4 | perception and interaction | camera/IMU/odometry → perception → `Observation`, no simulator truth; moving agents with physical presence; ≥ 2 functional scenes | |
| M5 | integrated APM | the same runtime drives all three robots in Isaac; body-specific expressive realisation | |
| M6 | final evaluation | `PROTOCOL.md` frozen; experiments A–F on fresh seeds; held-out scene | |
| M7 | videos and packaging | 4 videos from recorded traces; trace files, seeds, commands | |
| M8 | QA and release | checklist verified; cloud audit; docs; release report | |
