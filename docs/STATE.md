# Project state

*The single authoritative account of what APM is and does today. Updated 2026-10-20 for the v1.0 simulation research
snapshot (tag `v1.0-sim-snapshot`). Current results: `experiments/M_release/REMEDIATION_REPORT.md`, which governs
where it differs from the original v1.0 report (`experiments/M_release/REPORT.md`).*

Everything below is **simulation only**. No hardware has been tested, and no human has evaluated the robots'
expression.

## Current results (remediation, October 2026)

These are the corrected results. They supersede the comparisons further down where the two differ, because those
were affected by the person-contact artefact, DEVIATIONS D6.

| question | Nova Carter (Isaac) | Booster T1 (Isaac) | NERVA (MuJoCo) |
|---|---|---|---|
| identity-specific memory (beats pooled, swapped and shuffled memory) | supported | supported | supported |
| beats the calibrated rule, lookup, fixed style and no memory | yes | yes | yes |
| beats neutral motion with the same supervisor | no (identical) | no (n.s.) | no (worse) |
| falls in the corrected comparisons | 0 / 240 lifetimes | 0 / 240 | 0 / 240 |
| experience-dependent movement (one matched present-task seed) | timing, backing away | guarded arms, side-on torso, watching, backing away | head and neck pitch, watching, backing away |
| recovery within 7 calm meetings | not visible | partial | not visible |

- **T1 controller:** the original Isaac Lab PPO policy. The official Booster controller failed the frozen stability
  gate.
- **Perception:** rendered depth and motion heuristics. In the identity experiments the memory receives the
  person's identity.

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

## Original v1.0 results (preserved; partly superseded by the remediation)

| | Nova Carter | Booster T1 | NERVA |
|---|---|---|---|
| controller | stock wheel drives | Isaac Lab PPO policy + velocity map | frozen neutral ONNX + SafetyMonitor |
| Isaac Sim gate battery | pass | pass | pass |
| Isaac scenes (warehouse / home / hospital) | all three | all three | warehouse only; **falls in the home and in its hospital lifetime** |
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
- Identity-specific memory with visual re-identification (identities are provided; the corrected tests support
  identity-specific memory given identities).
- NERVA in Isaac experiments.
- Independent reproduction.

Full list: report §13–§14.
