# Project state

*The single authoritative account of what APM is and does today. Updated 2026-10-19, after the APM v1.0 release
programme (M; branch `release/apm-v1.0`, not merged; report: `experiments/M_release/REPORT.md`). Where M differs from
programme L, M governs.*

Everything below is **simulation only**. No hardware has been tested, and no human has evaluated the robots'
expression.

## Current architecture (one sentence)

1. The robot senses encounters with its **IMU and a rendered depth eye**.
2. An **explicit, revisable memory** turns them into a belief about each person: a learned latent type, with
   change-aware forgetting.
3. A **decision-theoretic mapping** (no VOI by default) with an **enforced risk cap** turns beliefs into a bounded
   **expressive intent**.
4. An **embodiment adapter** realises the intent on the robot's own validated controller.
5. **Two independent safety layers** can override it: APM's risk supervisor and the robot's physical supervisor.

One **integrated runtime** runs all of it, in Isaac Sim (NVIDIA Isaac Sim 5.1, PhysX) and in MuJoCo. Details:
[ARCHITECTURE.md](ARCHITECTURE.md).

## What works (simulation; evidence in the report)

| | Nova Carter | Booster T1 | NERVA |
|---|---|---|---|
| controller | stock wheel drives | Isaac Lab PPO policy + velocity map | frozen neutral ONNX + SafetyMonitor |
| Isaac Sim gate battery | pass | pass | pass |
| Isaac scenes (warehouse / home / hospital) | all three | all three | warehouse and hospital; **fails in the home** |
| comparative experiments | Isaac | Isaac | **MuJoCo** (DEVIATIONS D3) |

Preregistered tests on 624 fresh-seed lifetimes, verdicts across the three robots:

| verdict | hypotheses |
|---|---|
| **Demonstrated** | B1 (revises after a calm person turns pushy); C1–C4 (lower task cost than fixed style, neutral motion, fixed lookup and a calibrated simple rule) |
| **Partially demonstrated** | A1 (experience changes behaviour: T1 only after Holm); B2 (pushy → calm: Nova Carter, T1); D1–D3 (held-out hospital and home: Nova Carter, T1) |
| **Not demonstrated** | A2 (identity-specific memory, vs shuffled identities) |

Other results:
- **Expression** is physically measured and graded:
  - 114–345 distinct executed intents;
  - proximity vs stopping distance ρ ≈ −0.75 to −0.80;
  - tempo vs speed ρ ≈ +0.65 to +0.70.
- **Safety.** APM never executed an approach above its risk cap. T1 fell 16 times in 5,363 encounters, across all
  systems, always with the pushing person. Nova Carter fell once; NERVA (MuJoCo) never.
- **Robustness.** Restart, replay, camera loss, dropout, interruption, veto, anonymous identity and 100-encounter
  operation all pass. IMU detection of a light unexpected contact does not.

## What is not supported

- People's interpretation of the movement (study prepared, not run).
- Emotion, feeling or consciousness.
- Hardware.
- Production safety.
- Identity-specific memory.
- NERVA in Isaac experiments.
- Independent reproduction.

Full list: report §13–§14.
