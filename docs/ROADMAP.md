# Roadmap

Only the milestones still needed to demonstrate autonomous expressive development. History:
[RESEARCH_HISTORY.md](RESEARCH_HISTORY.md).

| # | milestone | done when | status |
|---|---|---|---|
| K1 | **Diagnose the recovery bottleneck** | it is measured what a distanced NERVA can observe about a changed entity: now, with a plausible extra sensor, and only through contact | **done**: IMU at 0.6 m carries 0.14 nats per calm meeting; a camera, 0.74 at any distance |
| K2 | **Safe information gathering** | the smallest mechanism that lets beliefs recover, using only signals the robot can sense, within the safety envelope | **done**: KG* value of information + OBSERVE from beyond reach (camera); memory fix D1 |
| K3 | **Independent confirmation** | recovery (both directions) and the decision-theoretic mapping with task values frozen in advance, on fresh seeds, vs L0 and baselines | **done**: K3-X/X2 milestone met (camera; IMU at the margin); K3b 4/4 tasks |
| K4 | **Continuous demonstration** | two identical NERVA robots, different histories, one continuous lifetime each, change → revision → recovery, with recorded-trace video and plots | **done**: `experiments/K_recovery/videos/` |
| K5 | **Second embodiment** | the same memory, mapping and information gathering, unchanged, on a physically different robot with its own adapter and safety layer | **done** for a MuJoCo wheeled robot (K5) and a Booster T1 humanoid (K6) |
| S1 | **Real exteroceptive sensing on NERVA** | a camera (or equivalent) that measures nearby agents' motion, validated against the modelled extension | open; makes recovery reliable rather than marginal |
| M1 | NERVA motor-control milestone | stable under repeated sub-envelope pushes; richer validated motion primitives (speed range, turning dynamics) | needs NERVA controller work (separate) |
| H2 | human legibility | preregistered study run after an ethics determination | blocked on ethics |
| R1 | production runtime | the validated architecture (memory with D1 fix, decision-theoretic mapping, information gathering) behind AffectLink, replacing the PAD core | next |
