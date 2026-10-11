# Milestone D: second embodiment, portability and cross-embodiment transfer

Protocol: [PREREGISTRATION.md](PREREGISTRATION.md) (committed `ece9723` before any wheeled learning result).
Runner `run_portability.py` (`0976d34`). Results: `results/results.json`. Local CPU, 72 min. The NERVA model retrained here reproduces Milestone C M4 exactly (Brier 0.016600).

## Second embodiment

A simulated differential-drive wheeled robot (`apm_testbeds.wheeled`), materially different from the NERVA biped:
- no joints, gait or tilt;
- wheel-traction stability margin;
- battery, obstacles and pedestrians with identities;
- waypoint navigation.

It has its own adapter (`apm.adapters.wheeled`) with its own capability profile and outcome detectors. Example: `examples/wheeled_robot/run.py`.

## Portability (engineering result) [measured]

| check | result |
|---|---|
| P1: core unchanged | **pass**: the SHA-256 of all 14 core/memory/protocol files is identical at the start and end of Milestone D; no AffectLink field was added |
| P2: protocol | **pass**: the 60 s wheeled demo sent 10,899 messages, all valid and none rejected; every outcome kind was observed, including battery `resource_critical` |
| same core, two bodies | the NERVA adapter (Milestone B) and the wheeled adapter drive the same `APM` class with no embodiment-specific code in the core |

The wheeled dataset (400 × 60 s episodes) contains:

| outcome | count |
|---|---|
| collision | 1,124 |
| wheel slip (stability_loss) | 733 |
| recovery | 711 |
| task_completed | 2,791 |
| task_interrupted | 676 |
| resource_critical | 56 |

In NERVA, stability losses almost never recovered (1 of 276); on the wheeled robot, 711 of 733 did.

## T1: transfer of `stability_loss` prediction (wheeled test set; mean over 3 seeds)

| arm | N = 5 | N = 20 | N = 80 | N = 240 |
|---|---|---|---|---|
| A: GRU from scratch, Brier / AUROC | 0.00892 / 0.908 | 0.00773 / 0.930 | 0.00712 / 0.949 | **0.00660 / 0.962** |
| C: NERVA GRU fine-tuned | 0.00920 / 0.875 | 0.00818 / 0.909 | 0.00736 / 0.949 | 0.00672 / 0.965 |
| D: logistic regression | **0.00827 / 0.931** | 0.00802 / 0.939 | 0.00795 / 0.939 | 0.00797 / 0.940 |
| B: NERVA GRU zero-shot | 0.0317 / 0.826 (any N) | | | |

Reference: the same GRU recipe on NERVA scores Brier 0.0166, AUROC 0.79 on NERVA's own test set (Milestone C).

| criterion | result |
|---|---|
| TR1: zero-shot AUROC ≥ 0.75 | **pass** (0.826): the NERVA-trained margin dynamics rank wheeled slip risk informatively, but are badly calibrated for the new body (Brier 0.032, ECE 0.045, about 5× the scratch model) |
| TR2: fine-tuned beats scratch at N = 5 and 20 by ≥ 10% | **FAIL**: fine-tuning is *worse* than scratch at N = 5 (+3%; better in 1 of 3 seeds) and N = 20 (+6%; better in 0 of 3) |
| TR3: no harm at N = 240 | pass (+1.8%) |

**Learned cross-embodiment transfer: not demonstrated** (TR2 failed). Pretraining on the biped did not improve sample
efficiency on the wheeled robot, despite compatible outcome semantics. Zero-shot ranking carried over; calibration
did not.

## Learning on the second robot [measured]

- With 240 episodes, the wheeled GRU beats logistic regression on slip prediction: Brier 0.0066 vs 0.0080 (−17%),
  AUROC 0.96 vs 0.94.
- With 5 episodes, logistic regression is best.
- **T2, collision within 1 s (wheeled only, no transfer attempted):**

  | model | Brier | AUROC | recall @ 5% FPR | ECE |
  |---|---|---|---|---|
  | GRU | **0.0185** | **0.946** | 0.80 | 0.006 |
  | logistic | 0.0299 | 0.916 | 0.72 | 0.015 |

  The temporal model clearly helps when entity dynamics matter.

## Observed behaviour of the transparent baseline on this body (negative finding)

In a 10-minute continuous wheeled run (`examples/wheeled_robot/run.py`):
- affect stayed chronically negative-valenced and aroused (V ≈ −0.4 to −0.8, A ≈ +0.3 to +1.0, briefly at the
  arousal bound);
- the adapter's time-to-contact estimator raises collision hypotheses almost continuously in a busy arena, and the
  baseline appraises routine, controllable approaches as threat.

The hand-set appraisal constants do not transfer gracefully to a robot whose estimator reports risk this often. This
is a calibration problem of the baseline; it was not tuned.

Per-identity memory differed meaningfully across pedestrians after 10 minutes: per-encounter collision probability
0.01–0.16, with confidence 0.57–0.82.
