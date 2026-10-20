# APM v1.0 remediation: final report

Branch `codex/apm-v1-remediation`. The release candidate `release/apm-v1.0` @ `c150288` is unchanged and nothing is
merged. Everything here is simulation (Isaac Sim 5.1 / PhysX; NERVA's comparisons on its MuJoCo backend). Beliefs
are estimates of expected physical perturbation from specific people, not emotions. No human has rated anything.

## 1. Outcome by request

| request | outcome |
|---|---|
| NERVA expression | **Works on a fresh seed, after fixes.** Learned wariness 0.73 after the pushy history vs 0.37 after the calm one, learned from the camera (near-miss lunges). Head and neck pitch, watching time and backing away differ visibly. Speed and stopping gap are matched. No falls. It took three corrections to NERVA's camera evidence, each found from pilot data (D8). |
| T1 walking | **The original controller (v1) is kept.** The official Booster controller fails the frozen battery (3 falls at 0.5 m/s pushes; settle time +42%), and so does every push-reflex variant, even on development seeds. Its gait looks better (slip −69%, jerk −57%, tilt −48%), but it is less robust. |
| Person-specific memory | **Supported on all three robots** (fresh seeds; DT beats POOLED, SWAPPED and SHUFFLED, with fewer contacts from the pushy person than POOLED): NERVA (MuJoCo), Nova Carter and T1 (Isaac). |
| Fair comparisons | Same supervisor for every system; falls, incomplete lifetimes, task success and ejections reported. 0 falls in 720 lifetimes. DT beats MONOTONE*, FIXED, LOOKUP and NOMEM on every robot, but **does not beat neutral motion + the same supervisor on any robot** (Carter: identical; T1: +0.38 s, n.s.; NERVA: +0.78 s, worse). |
| Videos | Three unlabeled side-by-side comparisons (neutral / after calm / after pushy) and three labelled learning demonstrations, all real physics. §5. |
| Release quality | Three simulator defects found and fixed (§6); CI green; fresh install verified; reproduction commands (§7). |

## 2. Expression: different histories, different movement

Protocol (`isaac/expression_v2.py`):
1. A quiet-period calibration with a still person sets the robot's own sensor thresholds.
2. Eight meetings with person B1, who is either pushy or calm. Memory learns from the robot's own IMU (unintended
   displacement while standing) and depth camera (looming). No labels.
3. The present task: B1 is calm, the physics seeds are identical, and approach speed and stopping distance are
   command-matched. It runs with the learned memory and with neutral behaviour.

Measured from the recorded traces (`scripts/expression_measures.py`):

| robot (seed) | learned wariness pushy / calm | after the pushy history | neutral | after the calm history | falls |
|---|---|---|---|---|---|
| T1, v1 controller (73301) | 0.94 / 0.28 | watches 3.5 s; arms guarded 0.91 rad; torso side-on 0.34 rad; backs away 0.19 m; turns away over 12.0 s | 0.5 s; 0; 0; 0; 7.2 s | 0.9 s; 0; 0; 0; 4.7 s | 0 of 24 meetings |
| NERVA (73302) | 0.73 / 0.37 | head pitched down −0.25 rad, neck −0.21 rad; watches 3.1 s; backs away 0.29 m | 0, 0; 0.5 s; 0 | head up +0.21 rad, neck +0.11 rad, head tilt 0.13 rad; 1.0 s; 0 | 0 of 24 |
| Nova Carter (73301) | 0.82 / 0.55 | watches 3.5 s; backs away 0.35 m; turns away over 10.9 s | 0.5 s; 0; 7.5 s | 1.9 s; 0.12 m; 8.3 s; looks around 0.06 rad | 0 of 24 |

- Matching: approach speed within 0.005 m/s of neutral for all robots. Stopping gaps are command-matched; realised
  gaps differ by up to 9 cm (T1 after pushy) and 5 cm (Carter after pushy).
- **NERVA's learning is near-miss learning.** Its pushy person lunges and stops 3 cm short of its body, because
  NERVA topples on contact in Isaac (D3). Two light contacts still happened in 8 pushy meetings. NERVA learned
  from what its camera saw (movement evidence 0.65 vs 0.14), not from being pushed.
- **Carter separates less.** Its calibrated disturbance threshold is 0.0008 m/s, since a standing wheeled robot
  barely moves, so even the calm person's occasional light touch counts as a disturbing meeting.
- Results: [expression10_humanoid.json](results/remediation/expression10_humanoid.json),
  [expression11_nerva.json](results/remediation/expression11_nerva.json),
  [expression7_nova_carter.json](results/remediation/expression7_nova_carter.json).
- Failed and superseded attempts are kept: [expression7_nerva.json](results/remediation/expression7_nerva.json)
  (camera threshold 0.54 m, nothing learned) and [invalid_D6/](results/remediation/invalid_D6/).

**Learning and recovery** (one lifetime: pushy for 6 meetings, then calm for 7; seed 73401):

| robot | wariness over the pushy phase | at the end of the calm phase | behaviour |
|---|---|---|---|
| T1 | 0.93 → 0.97 | 0.61 | learns, then recovers part-way: guarded posture visibly weaker at the end; no falls |
| Nova Carter | 0.89 → 0.98 | 0.79 | learns; recovery too small to change behaviour (the plan saturates above 0.8) |
| NERVA | 0.87 → 0.92, drifting to 0.77 | 0.71 | learns; recovery not demonstrated in 7 calm meetings |

The first T1 demo attempts fell, and every attempt is listed in D8.

## 3. T1 walking

- **Frozen battery:** 260 matched pairs, seeds 72001–72005; 4 environments × 13 scenarios
  ([gait_battery_v1_vs_candidate.json](results/remediation/gait_battery_v1_vs_candidate.json)).
- **Candidate quality:** stance slip 0.044 → 0.013 m/s; joint acceleration RMS 59 → 26 rad/s²; tilt p95 7.5° →
  3.9°; push peak tilt 7.8° → 4.6°.
- **Candidate failures:**
  - rule 1: 3 falls at 0.5 m/s walking pushes (one each in home, hospital and warehouse; v1: 0);
  - rule 4: settle time 0.94 s vs 0.66 s.
  - **Not promoted.**
- **Push reflex** (development seeds 72101–72105; [t1_push_reflex_dev.json](results/remediation/t1_push_reflex_dev.json)).
  None of R1–R4 meets the rule:
  - R1–R3 hijack normal walking (speed error about 0.27 m/s);
  - R4 still falls once and triples the tracking error;
  - settle time stays above 1.2 × v1.
  - No fresh battery was run, and PPO fine-tuning was stopped (two runs diverged).
- **Decision (D7):** v1 stays. Expression on v1 adds the posture to the targets and to the observed rest pose
  (`LabPolicyBody.upper_offset`). The posture check passes with 0 falls standing and walking, and the posture
  applies only while T1 stands still.

## 4. Corrected comparisons (frozen protocol, fresh R/Y seeds, the same supervisor for every system)

Summary:

| | Nova Carter | T1 | NERVA (MuJoCo) |
|---|---|---|---|
| DT vs MONOTONE*, FIXED, LOOKUP, NOMEM | better on all four (Holm), no extra falls | better on all four (Holm), no extra falls | better on all four (Holm), no extra falls |
| **DT vs neutral motion + the same supervisor** | **identical behaviour** (diff 0.00 s): DT never departs from the neutral approach | **not better** (+0.38 s, p = 0.89) | **worse** (+0.78 s) |
| identity-specific memory (beats POOLED and SWAPPED, Holm; B1 contacts ≤ POOLED) | **supported** (−4.1 s, −7.5 s; −0.35 B1 contacts per encounter) | **supported** (−5.0 s, −4.8 s; −0.76) | **supported** (−2.5 s, −15.0 s; −0.19) |
| falls / incomplete lifetimes / ejections | 0 / 0 / 0 of 240 | 0 / 0 / 0 of 240 | 0 / 0 / – of 240 |

**What this means.** The gains over the other baselines come from APM's per-person memory acting through the
shared risk supervisor: it holds back with the person who pushes and approaches the others. The decision-theoretic
choice of *expressive style* adds nothing measurable to task cost in this task, on any robot. APM is not claimed
to outperform neutral behaviour. Person-specific memory, by contrast, now holds on all three robots with fresh
seeds and controlled baselines.

### Nova Carter (Isaac, after D6; [corrected_nova_carter.json](results/remediation/corrected_nova_carter.json))

| system | lifetimes | lifetimes with a fall | complete | cost, all planned (s) | cost, survived (s) | contacts / encounter | task success | vetoes | ejections |
|---|---|---|---|---|---|---|---|---|---|
| main/DT | 30 | 0 | 1.00 | 16.78 | 16.78 | 0.282 | 0.71 | 0 | 0 |
| identity/DT | 15 | 0 | 1.00 | 18.79 | 18.79 | 0.300 | 0.56 | 0 | 0 |
| main/FIXED | 30 | 0 | 1.00 | 24.95 | 24.95 | 0.122 | 0.00 | 197 | 0 |
| main/LOOKUP | 30 | 0 | 1.00 | 26.72 | 26.72 | 0.117 | 0.00 | 192 | 0 |
| main/MONOTONE | 30 | 0 | 1.00 | 18.13 | 18.13 | 0.292 | 0.63 | 192 | 0 |
| main/NEUTRAL | 30 | 0 | 1.00 | 16.78 | 16.78 | 0.282 | 0.71 | 118 | 0 |
| main/NOMEM | 30 | 0 | 1.00 | 19.52 | 19.52 | 0.449 | 0.80 | 0 | 0 |
| identity/POOLED | 15 | 0 | 1.00 | 22.85 | 22.85 | 0.383 | 0.34 | 0 | 0 |
| identity/SHUFFLED | 15 | 0 | 1.00 | 25.21 | 25.21 | 0.581 | 0.52 | 0 | 0 |
| identity/SWAPPED | 15 | 0 | 1.00 | 26.30 | 26.30 | 0.642 | 0.54 | 0 | 0 |

| DT vs | cost diff (s) | 95% CI | p (one-sided) | Holm | falls DT/other | better overall |
|---|---|---|---|---|---|---|
| MONOTONE | -1.35 | [-2.25, -0.49] | 0.0029 | pass | 0/0 | yes |
| FIXED | -8.17 | [-9.27, -7.05] | 0.0000 | pass | 0/0 | yes |
| NEUTRAL | +0.00 | [+0.00, +0.00] | 1.0000 | fail | 0/0 | no |
| LOOKUP | -9.94 | [-11.02, -8.82] | 0.0000 | pass | 0/0 | yes |
| NOMEM | -2.74 | [-3.82, -1.71] | 0.0000 | pass | 0/0 | yes |

| identity: DT vs | cost diff (s) | 95% CI | p | Holm | B1 contacts diff | calm-person cost diff |
|---|---|---|---|---|---|---|
| POOLED | -4.06 | [-6.18, -1.94] | 0.0009 | pass | -0.345 | -3.47 |
| SWAPPED | -7.50 | [-10.13, -4.69] | 0.0003 | pass | -0.937 | -1.54 |
| SHUFFLED | -6.42 | [-9.19, -3.43] | 0.0007 | pass | -0.693 | -2.00 |

identity-specific memory supported: True

### T1, v1 controller (Isaac, after D6; [corrected_humanoid.json](results/remediation/corrected_humanoid.json))

| system | lifetimes | lifetimes with a fall | complete | cost, all planned (s) | cost, survived (s) | contacts / encounter | task success | vetoes | ejections |
|---|---|---|---|---|---|---|---|---|---|
| main/DT | 30 | 0 | 1.00 | 16.83 | 16.83 | 0.303 | 0.76 | 0 | 0 |
| identity/DT | 15 | 0 | 1.00 | 18.16 | 18.16 | 0.294 | 0.62 | 0 | 0 |
| main/FIXED | 30 | 0 | 1.00 | 27.67 | 27.67 | 0.275 | 0.14 | 164 | 0 |
| main/LOOKUP | 30 | 0 | 1.00 | 32.71 | 32.71 | 0.053 | 0.00 | 179 | 0 |
| main/MONOTONE | 30 | 0 | 1.00 | 21.01 | 21.01 | 0.222 | 0.32 | 143 | 0 |
| main/NEUTRAL | 30 | 0 | 1.00 | 16.45 | 16.45 | 0.269 | 0.73 | 189 | 0 |
| main/NOMEM | 30 | 0 | 1.00 | 18.06 | 18.06 | 0.487 | 0.99 | 0 | 0 |
| identity/POOLED | 15 | 0 | 1.00 | 23.16 | 23.16 | 0.578 | 0.69 | 0 | 0 |
| identity/SHUFFLED | 15 | 0 | 1.00 | 25.37 | 25.37 | 0.767 | 0.84 | 0 | 0 |
| identity/SWAPPED | 15 | 0 | 1.00 | 22.93 | 22.93 | 0.561 | 0.69 | 0 | 0 |

| DT vs | cost diff (s) | 95% CI | p (one-sided) | Holm | falls DT/other | better overall |
|---|---|---|---|---|---|---|
| MONOTONE | -4.18 | [-5.24, -3.18] | 0.0000 | pass | 0/0 | yes |
| FIXED | -10.84 | [-12.09, -9.60] | 0.0000 | pass | 0/0 | yes |
| NEUTRAL | +0.38 | [-0.19, +0.98] | 0.8884 | fail | 0/0 | no |
| LOOKUP | -15.88 | [-17.36, -14.36] | 0.0000 | pass | 0/0 | yes |
| NOMEM | -1.23 | [-2.42, -0.17] | 0.0192 | pass | 0/0 | yes |

| identity: DT vs | cost diff (s) | 95% CI | p | Holm | B1 contacts diff | calm-person cost diff |
|---|---|---|---|---|---|---|
| POOLED | -5.00 | [-7.66, -2.33] | 0.0022 | pass | -0.758 | -1.93 |
| SWAPPED | -4.77 | [-7.40, -2.01] | 0.0036 | pass | -0.643 | -1.46 |
| SHUFFLED | -7.21 | [-9.85, -4.31] | 0.0001 | pass | -1.039 | -1.17 |

identity-specific memory supported: True

### NERVA (MuJoCo backend, labelled; [corrected_nerva_mujoco.json](results/remediation/corrected_nerva_mujoco.json))

| system | lifetimes | lifetimes with a fall | complete | cost, all planned (s) | cost, survived (s) | contacts / encounter | task success | vetoes | ejections |
|---|---|---|---|---|---|---|---|---|---|
| main/DT | 30 | 0 | 1.00 | 15.74 | 15.74 | 0.272 | 0.72 | 0 | None |
| identity/DT | 15 | 0 | 1.00 | 17.46 | 17.46 | 0.333 | 0.58 | 0 | None |
| main/FIXED | 30 | 0 | 1.00 | 27.03 | 27.03 | 0.261 | 0.00 | 162 | None |
| main/LOOKUP | 30 | 0 | 1.00 | 25.25 | 25.25 | 0.158 | 0.00 | 76 | None |
| main/MONOTONE | 30 | 0 | 1.00 | 16.41 | 16.41 | 0.262 | 0.63 | 154 | None |
| main/NEUTRAL | 30 | 0 | 1.00 | 14.96 | 14.96 | 0.250 | 0.70 | 218 | None |
| main/NOMEM | 30 | 0 | 1.00 | 21.61 | 21.61 | 0.640 | 1.00 | 0 | None |
| identity/POOLED | 15 | 0 | 1.00 | 19.99 | 19.99 | 0.364 | 0.23 | 0 | None |
| identity/SHUFFLED | 15 | 0 | 1.00 | 25.38 | 25.38 | 0.703 | 0.59 | 0 | None |
| identity/SWAPPED | 15 | 0 | 1.00 | 32.42 | 32.42 | 1.119 | 0.77 | 0 | None |

| DT vs | cost diff (s) | 95% CI | p (one-sided) | Holm | falls DT/other | better overall |
|---|---|---|---|---|---|---|
| MONOTONE | -0.67 | [-1.17, -0.15] | 0.0082 | pass | 0/0 | yes |
| FIXED | -11.29 | [-13.10, -9.55] | 0.0000 | pass | 0/0 | yes |
| NEUTRAL | +0.78 | [+0.55, +1.02] | 1.0000 | fail | 0/0 | no |
| LOOKUP | -9.51 | [-10.55, -8.46] | 0.0000 | pass | 0/0 | yes |
| NOMEM | -5.87 | [-7.53, -4.32] | 0.0000 | pass | 0/0 | yes |

| identity: DT vs | cost diff (s) | 95% CI | p | Holm | B1 contacts diff | calm-person cost diff |
|---|---|---|---|---|---|---|
| POOLED | -2.52 | [-3.85, -1.37] | 0.0007 | pass | -0.194 | -2.27 |
| SWAPPED | -14.96 | [-17.63, -12.36] | 0.0001 | pass | -1.841 | -0.82 |
| SHUFFLED | -7.91 | [-10.61, -5.58] | 0.0001 | pass | -0.879 | -1.49 |

identity-specific memory supported: True

Reading the tables:
- "Better overall" requires a lower cost over all planned encounters (Holm) **and** no higher fall rate.
- Falls count as failed encounters in the all-planned cost.
- Ejections are encounters where the standing robot was thrown faster than 3 m/s (D6). After the fix there
  should be none.

## 5. Videos

All six are real physics, recorded in Isaac Sim, with no retiming or interpolation. Local copies are in
`runs/remediation/final_videos/`, with a backup in the project bucket (`apm-m/remediation_final/videos/`). All are
H.264 / yuv420p / faststart and play on web and mobile.

| file | what | length |
|---|---|---|
| `T1_unlabeled.mp4` | T1: neutral, after calm, after pushy; panels A/B/C shuffled; key in `.key.json` | 64 s |
| `NERVA_unlabeled.mp4` | NERVA, the same | 45 s |
| `Carter_unlabeled.mp4` | Nova Carter, the same | 43 s |
| `T1_learning_recovery_demo.mp4` | labelled: one T1 lifetime, 6 pushy then 7 calm meetings, learned state shown; 2× speed | 7.6 min |
| `Carter_learning_recovery_demo.mp4` | the same for Nova Carter (continuous lifetime) | 5.4 min |
| `NERVA_learning_demo.mp4` | the same for NERVA | 5.2 min |

The v1-vs-candidate walking comparison from the gait gate is `runs/remediation/review/01_T1_walking_comparison.mp4`.

## 6. Simulator defects found and fixed

- **D6, the person model.**
  - A lunge's contact limit was recomputed every step. A robot pushed back was chased at up to 10 m/s, which threw
    Nova Carter 1–4 m in about 9–12% of encounters, in programme M and in the first corrected runs.
  - Two more defects surfaced while verifying the fix: Carter's caster colliders report 100 m boxes, and a
    limit frozen at onset overran an approaching robot.
  - Final rule: the limit is the smallest since onset, and Carter's gap is measured to its footprint.
  - Verified: 16/16 encounters with real contact and 0 ejections, against 12/12 ejections before
    ([before](results/remediation/carter_contact_before_D6.json), [after](results/remediation/carter_contact_after_D6.json)).
  - Programme M's Isaac results contain the artefact; they are preserved, with this note.
- **The Isaac environment snapshot** in the bucket had been overwritten with a truncated copy. The good version
  was restored from soft delete, and the setup script now re-restores incomplete installs and never overwrites the
  snapshot.
- **D8:** NERVA's camera evidence and T1's posture timing (§2).

## 7. Reproduction

```bash
pip install -e ".[dev,sim]" && pytest -q && ruff check . && mypy src
```

Isaac jobs need `experiments/M_release/cloud/isaac_env.sh` (it exports `$ISAAC_PY`); NERVA also needs
`setup_nerva_inputs.sh`.

| what | command |
|---|---|
| expression run (one robot, both histories) | `HISTORIES="pushy calm" SEED=73302 bash experiments/M_release/cloud/expression_v2.sh nerva` (also `humanoid`, `nova_carter`) |
| learning-and-recovery demo | `SEED=73401 bash experiments/M_release/cloud/expression_demo.sh humanoid` |
| expression measures | `python experiments/M_release/scripts/expression_measures.py --records <run>.json ... --out <json>` |
| unlabeled video | `python experiments/M_release/scripts/compose_unlabeled.py --stems <neutral> <calm> <pushy> --out X.mp4` |
| demo video | `python experiments/M_release/scripts/compose_demo.py --stem <..._demo> --out D.mp4 --speed 2` |
| gait battery | `$ISAAC_PY experiments/M_release/isaac/gait_battery.py --controller v1 --env flat --seeds 72001-72005 --out out/gb`, then `scripts/gait_battery_analyze.py` |
| corrected comparisons | `python experiments/M_release/cloud/m6_plan.py --stage RY --vms <vm> --tuned M6_tuned_R6.json` prints the jobs; `scripts/r_analyze.py` analyses them |
| Carter contact check | `$ISAAC_PY experiments/M_release/isaac/carter_contact_check.py --out out/cc --footprint` |

## 8. Limitations

- Simulation only. One scene (the warehouse) for the corrected comparisons. Two present-task meetings per
  condition for the expression measures, which are descriptive, not a statistical test.
- NERVA cannot take contact in Isaac. Its expression memory is near-miss learning, and its comparative results
  come from MuJoCo.
- Recovery after a calm phase is partial (T1) or not visible (Carter, NERVA) within 7 meetings. The memory's 0.85
  retention is slow, and Carter's disturbance threshold counts light touches.
- The expressive mapping is designed. What is learned is the per-person state that drives it.
- No human perception study. The user's visual approval is not perception evidence.
- The security note on commit `75c3b86` (a public instance-list dump: project ID, service-account email, bucket
  name; no secrets) still needs the user's decision.

## 9. Cloud cost

| item | VM-hours | estimate (list prices, from audit logs) |
|---|---|---|
| v1.0 sprint (programme M) | 124 h, g2-standard-16 + L4 | about US$142 (about US$150 with disks and network) |
| this remediation | 102 h on-demand L4 (plus about 1 h spot and CPU) | about US$112 (about US$120 with disks, 4 temporary NAT gateways, storage and cross-region restores) |
| **total against the US$500 cap** | | **about US$270** |

- About US$25 of the remediation went on jobs that were stopped and discarded after the three D6 defects were
  found. They are listed in D6.
- On-demand VMs only after spot preemptions: 6–10 h runtime caps and 45 min idle self-deletion.
- **Cleanup verified on 2026-10-20 02:05 UTC:** no VMs, disks, routers or addresses remain (`jcloud.py audit`).
  The bucket keeps the job archives, the environment snapshot (plus the broken one, kept as a record) and the
  final videos.
- The billing console is authoritative. These are list-price estimates.
