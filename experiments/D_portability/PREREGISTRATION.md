# Milestone D preregistration: second embodiment, portability and cross-embodiment transfer

Fixed before any wheeled-robot learning result was computed. The Milestone C results were also not yet available.

## Embodiment 2

The wheeled robot (`apm_testbeds.wheeled`, adapter `apm.adapters.wheeled`): a differential drive with
- obstacles and pedestrians with persistent identities;
- low-friction patches causing wheel slip;
- a battery;
- waypoint navigation.

It supplies its own capabilities: a traction-based stability margin, energy margin, actuation stress, entities, and
outcome kinds collision, stability_loss (wheel slip), recovery, task_completed, task_interrupted and
resource_critical. It does not imitate the biped's joints, gait, tilt or stability calculation.

## Portability (engineering claim, not learned transfer)

- **P1.** The APM core runs on both embodiments without modification. The SHA-256 of every file under
  `src/apm/core`, `src/apm/memory` and `src/apm/protocol` is identical at the start and end of Milestone D, and no
  embodiment-specific field is added to AffectLink.
- **P2.** Every message from both adapters validates; the core rejects none for protocol reasons in the
  demonstration runs.

## Data

- **Wheeled episodes:** 400 × 60 s, seeds 10,000–10,399.
- **Labels:** observed outcomes from the wheeled adapter's detectors only.
- **Sampling:** every 0.1 s, 2 s history, armed samples only, with the same canonical masked features as
  Milestone C.
- **Split:** a seeded permutation (seed 0) into a training pool (60%), validation (20%) and test (20%).
- **NERVA:** the Milestone C NERVA experience, splits and GRU procedure.

## Tasks

- **T1, compatible semantics: `stability_loss` within 0.5 s.** On both bodies this is "the calibrated stability
  margin reaches its critical value", which is tilt for the biped and traction for the wheeled robot.
  - [stated limitation] The causes differ: pushes on the biped, turning on slippery floor on the wheeled robot.
  - The biped's losses always ended in falls in the Milestone C smoke data; the wheeled robot usually recovers.
- **T2, wheeled only: `collision` within 1.0 s.** NERVA has no collision signal, so no transfer is attempted.

## Arms for T1

Nested wheeled training subsets of N ∈ {5, 20, 80, 240} episodes from the pool (all 240 = full pool), seeds
{0, 1, 2}:

| arm | model |
|---|---|
| A | GRU from scratch (64 hidden, the same architecture as Milestone C) on N wheeled episodes, early stopping on wheeled validation |
| B | the NERVA-trained GRU applied zero-shot |
| C | the NERVA-trained GRU fine-tuned on N wheeled episodes (lr 1e-3, same early stopping) |
| D | logistic regression on N wheeled episodes |

Also reported: the robot-specific models on their own test sets (NERVA M4 on NERVA test; arm A at N = 240 on
wheeled test).

## Criteria (fixed now)

- **TR1, zero-shot transfer is informative.** Arm B's AUROC on the wheeled test set is ≥ 0.75.
- **TR2, sample efficiency.** Arm C's Brier score is lower than arm A's at N = 5 and at N = 20, in at least 2 of 3
  seeds each, and the mean improvement at each of those N is ≥ 10% relative.
- **TR3, no harm with full data.** Arm C's Brier score is ≤ 1.05 × arm A's at N = 240 (mean over seeds).
- **T2.** The wheeled GRU (240 episodes) is compared with logistic regression on collision prediction (Brier, AUROC);
  reported, with no transfer claim.

Learned cross-embodiment transfer is claimed only if TR1 and TR2 both hold. Portability (P1, P2) is reported
separately and is not evidence of learned transfer.
