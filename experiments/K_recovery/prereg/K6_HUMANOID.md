# K6 addendum: the same loop on a humanoid (fixed before any humanoid data, 2026-10-18)

**Robot:** Booster T1 (MuJoCo). Its walking controller is a joystick policy trained with mujoco_playground's published
PPO recipe on one L4 GPU (`humanoid/train_t1.py`), with that recipe's random pushes of 0.1–1.0 m/s. The trained
policy is frozen. `src/apm_transfer/humanoid_world.py` deploys it in plain MuJoCo, with the same adapter, supervisor,
observation model and camera extension as the wheeled robot. APM never commands joints.

## Gate G0: the controller is validated before any APM data

Over 50 encounters (calm and pushy entities, stop distances 0.2–0.6 m, 1–3 windows, seeds K6/P), all of the
following must hold:
- 0 falls;
- arrival in ≥ 98% of encounters;
- a standing stop within 0.1 m of the requested distance in ≥ 90%.

If G0 fails, the humanoid result is "controller not validated". No APM claims are made for it, and a motor-control
milestone is reported.

## If G0 passes

1. **Exploration (K6-T):** 100 lifetimes × 4 histories, RANDINTENT. `k_fit.py --robot humanoid` fits the robot's own
   acc8 edges, time model and type model.
2. **Test (K6-X):**
   - indices 0–29;
   - systems L0, DT, DT_CAM, DT_CAM_VOI, ORACLE × histories risky_B, safe, change_B, late_B;
   - task handoff; `--transition-fix` (D1).
3. **Criteria:** P1–P5 of `PROTOCOL.md` §4, judged per candidate (DT with IMU only; DT_CAM_VOI). A fall is the pelvis
   below 0.45 m or tilt above 50°.
