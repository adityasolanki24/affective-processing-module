"""Replay stored NERVA simulator telemetry through the NERVA adapter into ONE persistent APM (observe-only).

Input: the raw rollouts of NERVA's neutral motor gate for its validated candidate (160 trials: steady commands,
transition sequences, 0.30 m/s pushes; with and without action latency), written by NERVA's
`experiments/locomotion_curriculum/neutral_gate.py`. Only arrays are read; NERVA is not imported.
The command schedule is reconstructed from NERVA's documented gate protocol (trial names, 5 s transition phases).

    python examples/nerva/replay_gate.py --gate-dir <NERVA>/experiments/cloud_runs/turn_translation-20261011-000711/gate
        [--out runs/nerva_gate_replay]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.nerva import NervaAdapter
from apm.protocol import to_dict
from apm.protocol.messages import OutcomeHypothesis, OutcomeSignal

COMMANDS = {"rest": (0.0, 0.0, 0.0), "forward": (0.074, 0.0, 0.0), "backward": (-0.074, 0.0, 0.0),
            "left": (0.0, 0.074, 0.0), "right": (0.0, -0.074, 0.0), "turn_left": (0.0, 0.0, 0.6),
            "turn_right": (0.0, 0.0, -0.6)}
PHASES = ("rest", "forward", "backward", "left", "right", "turn_left", "turn_right", "rest")


def schedule(name: str, steps: int):
    """Trial file stem, e.g. "no_latency_push_turn_left_90_3", → one command per control step."""
    trial = name.removeprefix("no_latency_").removeprefix("latency_")
    parts = trial.split("_")
    if parts[0] == "transition":
        return [COMMANDS[PHASES[min(k // 250, 7)]] for k in range(steps)]
    command = "_".join(parts[1:-1]) if parts[0] == "steady" else "_".join(parts[1:-2])
    return [COMMANDS[command]] * steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/nerva_gate_replay"))
    args = ap.parse_args()
    files = sorted(args.gate_dir.glob("*.npz"))
    if not files:
        raise SystemExit("no NERVA gate rollouts found")
    args.out.mkdir(parents=True, exist_ok=True)
    emitter = Emitter("nerva-open-duck")
    adapter = NervaAdapter(emitter)
    apm = APM(profile=adapter.profile())
    offset, summary = 0.0, []
    with (args.out / "outputs.jsonl").open("w", encoding="utf-8") as log:
        for f in files:
            a = dict(np.load(f))
            trial = f.stem  # e.g. latency_push_forward_90_3
            commands = schedule(trial, len(a["t"]))
            counts: collections.Counter = collections.Counter()
            peak = np.zeros(3)
            p_max, max_tilt_margin, alarms = 0.0, 1.0, 0
            for i in range(len(a["t"])):
                t = offset + float(a["t"][i])
                for m in adapter.step(t, a["base_quat"][i], a["base_linvel"][i], a["base_angvel"][i],
                                      float(a["tau_sq"][i]), commands[i]):
                    reason = apm.ingest(m)
                    if reason is None and isinstance(m, OutcomeSignal):
                        counts[m.kind] += 1
                    if reason is None and isinstance(m, OutcomeHypothesis):
                        p_max = max(p_max, m.probability)
                        alarms += m.probability >= 0.5
                if i % 5 == 4:  # APM steps at 10 Hz
                    out = apm.step(t)
                    peak = np.where(np.abs(out.affect_vector()) > np.abs(peak), out.affect_vector(), peak)
                    max_tilt_margin = min(max_tilt_margin, apm.context.margin("stability_margin", t) or 1.0)
                    if i % 50 == 49:
                        log.write(json.dumps({"trial": trial, "t": t, "affect": to_dict(out.affect),
                                              "tendencies": [to_dict(x) for x in out.tendencies],
                                              "diagnostics": to_dict(out.diagnostics)}) + "\n")
            summary.append({"trial": trial, "outcomes": dict(counts), "max_stability_loss_p": p_max,
                            "control_steps_with_p_over_half": int(alarms), "control_steps": len(a["t"]),
                            "min_stability_margin": max_tilt_margin, "peak_affect": peak.round(3).tolist()})
            offset = t + 30.0  # 30 s pause between trials: separate episodes of one continuous robot life
            adapter = NervaAdapter(emitter)  # trial boundary: detectors re-arm (the robot was reset)
            apm.step(offset)
    apm.save(args.out / "apm_state.json")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    totals: collections.Counter = collections.Counter()
    for s in summary:
        totals.update(s["outcomes"])
    print(f"trials {len(summary)}; outcomes {dict(totals)}; "
          f"min stability margin {min(s['min_stability_margin'] for s in summary):.3f}; "
          f"max predicted stability-loss p {max(s['max_stability_loss_p'] for s in summary):.3f}; "
          f"trials with a p >= 0.5 prediction {sum(s['control_steps_with_p_over_half'] > 0 for s in summary)}, "
          f"control steps {sum(s['control_steps_with_p_over_half'] for s in summary)} of "
          f"{sum(s['control_steps'] for s in summary)}")
    for kind in ("steady", "transition", "push"):
        sel = [s for s in summary if f"_{kind}_" in s["trial"] or s["trial"].endswith(kind)]
        if sel:
            peaks = np.array([s["peak_affect"] for s in sel])
            print(f"{kind:10s} n={len(sel):3d} peak |V| {np.abs(peaks[:, 0]).max():.3f} "
                  f"peak A {peaks[:, 1].max():+.3f} min margin {min(s['min_stability_margin'] for s in sel):.3f}")
    d = apm.step(offset + 0.1).diagnostics
    print(f"diagnostics: health {d.health}, rejected {d.rejected_messages}, memory updates {d.memory_updates}, "
          f"missing {d.missing_capabilities}, saturation {d.saturation}")


if __name__ == "__main__":
    main()
