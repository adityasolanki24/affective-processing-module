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
| S1 | **Real exteroceptive sensing** | rendered-image perception validated against the abstract sensor, then a real camera on NERVA | **done in Isaac Sim (M)** for three robots with physically present people. Open: T1's camera false positives (21% with calm people), real hardware |
| M1 | NERVA motor-control milestone | stable under repeated sub-envelope pushes; richer validated motion primitives (speed range, turning dynamics) | needs NERVA controller work (separate) |
| H2 | human legibility | preregistered study run after an ethics determination | blocked on ethics |
| R1 | integrated runtime | the validated architecture behind AffectLink, on more than one body, with persistence, replay and an independent supervisor | **done in simulation (L F, M F)**: `apm_transfer.runtime` on three bodies, Isaac and MuJoCo. Open: replacing `src/apm`'s PAD core as the shipped runtime |
| L1 | memory under stress | an alternative that improves task cost under drift, overlap and identity errors | GRU question **closed (M)**: the explicit memory predicts held-out contacts better on Nova Carter and T1 |
| N1 | NERVA in Isaac | NERVA exploration, calibration and confirmatory test in Isaac scenes; the home floor | open (M D3) |
| A2 | identity-specific memory | DT beats a shuffled-identity memory with adequate power | open: not demonstrated on any robot (M) |
| T1S | T1 push robustness | T1 falls with pushing people reduced without APM bypassing its controller | open: 16 falls in 5,363 X encounters (M) |
