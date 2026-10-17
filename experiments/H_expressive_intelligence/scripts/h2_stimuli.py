"""H2 stimulus clips (prereg/H2_PROTOCOL.md §3): anonymised, label-free renderings of expressive approaches.

Scene:
- NERVA's flat backlash scene plus one orange ball (visual only, no collision) at the target;
- one fixed free camera (azimuth 120°, elevation −20°) looking at the midpoint of start and target;
- 480 × 360 px, 25 fps, H.264 through the system ffmpeg.

Each clip runs from the start until 5 s after arrival (at most 45 s). File names are opaque hashes; the
style-to-file key goes into a separate manifest, which is never shown to participants.

The closed loop is identical to `apm_express.episode.run_episode` (same adapter, monitor and fallback), with a frame
grabbed every second control step.

    NERVA_ROOT=... PYTHONPATH="src;<NERVA>" <nerva-venv>/python scripts/h2_stimuli.py --set h1 --out videos/h1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from h0_interface import interface  # noqa: E402
from h1_repertoire import GEOMETRIES, S  # noqa: E402

from apm_express.adapter import NEUTRAL_REQUEST, BipedExpressionAdapter  # noqa: E402
from apm_express.intent import ExpressionIntent  # noqa: E402
from apm_express.safety import SafetyMonitor  # noqa: E402
from apm_express.styles import DESIGNED, NEUTRAL, with_tempo  # noqa: E402

W, H, FPS = 480, 360, 25
MAX_S = 45.0
TAIL_S = 5.0


def ball_extender(spec) -> None:
    import mujoco

    body = spec.worldbody.add_body(name="target_ball", mocap=True, pos=[5.0, 0.0, 0.06])
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.06, 0, 0], rgba=[0.95, 0.45, 0.10, 1], contype=0, conaffinity=0
    )


def render_clip(intent: ExpressionIntent, target: tuple[float, float], seed: int, iface: dict, path: Path) -> dict:
    import mujoco

    from apm_express.biped import NervaBiped

    ch, limits = interface(iface)
    ctrl = NervaBiped(
        policy=iface["policy"],
        seed=seed,
        head_channel=ch.head,
        scene_extender=ball_extender,
        max_head_rate=1.5,
        max_pitch_rate=2.0,
    )
    model, data = ctrl.sim.model, ctrl.sim.data
    mocap = model.body_mocapid[model.body("target_ball").id]
    ox, oy = ctrl._origin
    data.mocap_pos[mocap] = [ox + target[0], oy + target[1], 0.06]
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [ox + target[0] / 2, oy + target[1] / 2, 0.15]
    cam.distance, cam.azimuth, cam.elevation = 1.8, 120.0, -20.0
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    adapter, monitor = BipedExpressionAdapter(ch, seed=seed), SafetyMonitor(limits)
    adapter.reset(ctrl.t, ctrl.pose(), target, intent)
    arrived_at, k, fell = None, 0, False
    d_stop = None
    while ctrl.t < MAX_S:
        tripped = monitor.update(ctrl.t, **ctrl.physical())
        if tripped:
            req = NEUTRAL_REQUEST
            adapter.notify_fallback(ctrl.t)
        else:
            req = adapter.request(ctrl.t, ctrl.pose(), target, intent)
        ctrl.apply(req)
        row = ctrl.advance()
        fell = fell or row["tilt"] > 45 or not row["finite"]
        if k % 2 == 0:
            renderer.update_scene(data, cam)
            assert ff.stdin is not None
            ff.stdin.write(renderer.render().tobytes())
        k += 1
        x, y = ctrl.pose()[:2]
        d = math.hypot(target[0] - x, target[1] - y)
        d_stop = 0.6 - 0.4 * intent.sanitized().proximity
        if arrived_at is None and d <= d_stop + 0.02:
            arrived_at = ctrl.t
        if arrived_at is not None and ctrl.t >= arrived_at + TAIL_S:
            break
    assert ff.stdin is not None
    ff.stdin.close()
    ff.wait()
    renderer.close()
    return {"seconds": round(ctrl.t, 2), "arrived_at": arrived_at, "fell": fell, "d_stop": d_stop}


def h1_set(tempos: dict) -> list[dict]:
    """Protocol §3 clip set: 6 styles × 2 geometries × 2 seeds, the same at matched speed, neutral × 2 × 2,
    20 random intents. Seeds are partition X index 0 and 1 (fixed rule, not chosen by eye)."""
    clips = []
    geoms = (0, 6)  # 0.9 m straight ahead; 1.3 m at bearing −0.7 rad
    for g in geoms:
        for s in (0, 1):
            clips.append(
                {"style": "neutral", "speed": "native", "geometry": g, "seed_index": s, "intent": NEUTRAL.as_dict()}
            )
            for name, it in DESIGNED.items():
                clips.append({"style": name, "speed": "native", "geometry": g, "seed_index": s, "intent": it.as_dict()})
                clips.append(
                    {
                        "style": name,
                        "speed": "matched",
                        "geometry": g,
                        "seed_index": s,
                        "intent": with_tempo(it, tempos[name]).as_dict(),
                    }
                )
    rng = np.random.default_rng(S.world_seed("H2", "X", 0))
    from apm_express.styles import random_intent

    for i in range(20):
        clips.append(
            {
                "style": "random",
                "speed": "native",
                "geometry": geoms[i % 2],
                "seed_index": i % 2,
                "intent": random_intent(rng).as_dict(),
            }
        )
    return clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="h1", choices=["h1"])
    ap.add_argument("--interface", type=Path, default=HERE / "results/H0/X/admitted_interface.json")
    ap.add_argument("--tempos", type=Path, default=HERE / "results/H1/V/tempos.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N clips (smoke test)")
    a = ap.parse_args()
    iface = json.loads(a.interface.read_text(encoding="utf-8"))["interface"]
    clips = h1_set(json.loads(a.tempos.read_text(encoding="utf-8"))["tempos"])
    if a.limit:
        clips = clips[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in clips:
        dist, bearing = GEOMETRIES[c["geometry"]]
        target = (dist * math.cos(bearing), dist * math.sin(bearing))
        seed = S.sim_seed("H2", "X", c["seed_index"])
        tmp = a.out / "tmp.mp4"
        info = render_clip(ExpressionIntent(**c["intent"]), target, seed, iface, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        final = a.out / f"clip_{digest[:12]}.mp4"
        tmp.replace(final)
        manifest.append({**c, **info, "file": final.name, "sha256": digest})
        print(final.name, c["style"], c["speed"], info, flush=True)
    (a.out / "MANIFEST_KEY_DO_NOT_SHOW.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
