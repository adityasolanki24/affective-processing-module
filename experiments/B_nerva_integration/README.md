# Milestone B: NERVA embodiment through AffectLink (observe-only)

Embodiment: NERVA's Open Duck Mini v2 biped in MuJoCo (backlash scene, training observation noise), driven by
NERVA's validated neutral locomotion candidate (the policy that passed NERVA's long motor gate). Adapter:
`apm.adapters.nerva`, which reads arrays only and never imports NERVA. APM was observe-only: the policy, its
actions, rewards, checkpoints, NERVA's safety logic and its expressive inputs were untouched (NERVA and upstream
git trees verified clean afterwards).

## 1. Stored telemetry replay [measured]

`examples/nerva/replay_gate.py`: all 160 raw rollouts of that candidate's neutral gate.
- 7 steady commands × 5 seeds, 5 transition sequences, 40 pushes of 0.30 m/s; each with and without action latency.
- 150,000 control steps through one persistent APM instance, with a 30 s pause between trials.

| item | result |
|---|---|
| observed outcomes | **none**: no stability loss (25°), fall, recovery or task interruption, consistent with NERVA's gate (no falls; max tilt ≈ 15.7°) |
| lowest stability margin | 0.651 (≈ 15.7° tilt), in a push trial |
| affect | bounded: peak \|V\| 0.25 (steady), 0.27 (push); peak A +0.35; zero time near the PAD bounds |
| messages rejected | 0 |
| memory updates | 423 context windows (no entity encounters: the motor telemetry has no entities) |
| adapter predictor | tilt extrapolation reached **p = 0.98 for a stability loss that never happened**: 14 of 160 trials had at least one p ≥ 0.5 prediction (50 of 150,000 control steps). **Negative finding:** this simple estimator raises false alarms on real data, and those false alarms feed appraisal as threat |

## 2. One bounded live rollout [measured]

`examples/nerva/live_rollout.py`: forward walk (0.074 m/s), seed 0, one 1.0 m/s sideways push at 8.0 s. NERVA's
training pushes span 0.1–1.0 m/s; this push is larger than the gate's 0.30 m/s.

| t (s) | event / APM output |
|---|---|
| 8.00 | push; tilt 2.4°; V −0.07, A +0.08 |
| 8.04–8.16 | adapter predicts stability_loss, p 0.99–1.00 (≈ 140 ms before it occurs) |
| 8.18 | **stability_loss** observed (tilt ≥ 25°) |
| 8.28 | **fall** observed (tilt > 45°): the validated policy did not recover from this push |
| 9.0 | V −0.39, A +0.56, D −0.27; self: disengage 0.82, seek_assistance 0.75 |
| 12.0 | V −0.85, A +0.97, D −0.62; disengage decays (outcome frame expired), seek_assistance 0.98 (margin still 0) |
| 20.0 | V −0.74, A +0.74, D −0.61; seek_assistance 1.00 |

A first run of this rollout exposed a baseline flaw. With no live adverse frame, controllability defaulted to
"fully in control", so a robot lying on the ground stopped seeking assistance after 4 s. It was fixed: the
stability margin is used as the fallback, with a regression test. The table shows the run after the fix.

## Files

`gate_replay_summary.json` (per trial), `gate_replay_console.txt`, `live_rollout_events.json`,
`live_rollout_console.txt`. Full JSONL output logs are reproducible locally under `runs/` (ignored).

## Reproduce

```bash
python examples/nerva/replay_gate.py --gate-dir <NERVA>/experiments/cloud_runs/turn_translation-20261011-000711/gate
PYTHONPATH=src python examples/nerva/live_rollout.py --nerva <NERVA> \
    --policy <NERVA>/experiments/cloud_runs/turn_translation-20261011-000711/candidate.onnx --push 1.0
```

The second command must run in a Python environment with NERVA's simulator dependencies.
