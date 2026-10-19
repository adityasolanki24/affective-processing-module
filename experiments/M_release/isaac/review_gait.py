"""Short real-time T1 pilot recording and synchronized physical trace (no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--controller", choices=("baseline", "booster"), required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seed", type=int, default=71001)
ap.add_argument("--seconds", type=float, default=24)
ap.add_argument("--forward-scale", type=float, default=1.0,
                help="pilot command calibration; external requested speed remains unchanged")
ap.add_argument("--push", type=float, default=0.0, help="pilot velocity disturbance at 12 s and 16 s, m/s")
ap.add_argument("--target-filter", type=float, default=0.2, help="candidate target smoothing coefficient")
ap.add_argument("--training-gains", action="store_true", help="candidate: official simulation gains instead of deploy")
ap.add_argument("--gait-hz", type=float, default=1.0)
a = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": True}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for ext in ("isaacsim.sensors.camera", "isaacsim.storage.native"):
    enable_extension(ext)

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import UsdLux, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from apm_isaac import exit_on_error  # noqa: E402
from apm_isaac.booster import BoosterPolicyBody  # noqa: E402
from apm_isaac.robots import LabPolicyBody  # noqa: E402
from apm_isaac.world import _look_at  # noqa: E402

exit_on_error()
a.out.mkdir(parents=True, exist_ok=True)
policy = ROOT / "experiments/M_release/policies" / (
    "t1_isaaclab_v1" if a.controller == "baseline" else "t1_booster_candidate")
world = World(stage_units_in_meters=1.0, physics_dt=0.005 if a.controller == "baseline" else 0.002,
              rendering_dt=0.02)
world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
light = UsdLux.DomeLight.Define(world.stage, "/World/Light")
light.CreateIntensityAttr(1500.)
if a.controller == "baseline":
    body = LabPolicyBody.t1(world, "/World/T1", (0., 0.), 0., a.seed, policy)
else:
    body = BoosterPolicyBody(world, "/World/T1", (0., 0.), 0., a.seed, policy)
    if not 0 < a.target_filter <= 1:
        raise ValueError("target filter must be in (0, 1]")
    body.target_filter = a.target_filter
    body.gait_frequency = a.gait_hz
    if a.training_gains:
        body.kp[11:] = [200., 200., 200., 200., 50., 50.] * 2
        body.kd[11:] = [5., 5., 5., 5., 1., 1.] * 2
eye, aim = np.array([3., -4.5, 1.7]), np.array([1.0, 0., 0.55])
camera = Camera("/World/Review", position=eye, orientation=_look_at(eye, aim), resolution=(960, 540))
world.reset()
body.post_reset()
noise = np.random.default_rng(a.seed).normal(0., .002, len(body.art.dof_names))
body.art.set_joint_positions(np.asarray(body.art.get_joint_positions()) + noise)
camera.initialize()
camera.set_focal_length(2.8)
feet = []
for side in ("left", "right"):
    prim = next(p for p in world.stage.Traverse()
                if p.GetName() == f"{side}_foot_link" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    foot = RigidPrim(str(prim.GetPath()), name=f"measure_{side}", track_contact_forces=True)
    foot.initialize()
    feet.append(foot)
for _ in range(12):
    world.render()


def command(t):
    if t < 2:
        return 0., 0.
    if t < 8:
        return 0.2, 0.
    if t < 10:
        return 0., 0.
    if t < 14:
        return 0.2, 0.5
    if t < 20:
        return 0.2, 0.
    return 0., 0.


fell = False
trace = []
video = imageio.get_writer(a.out / "motion.mp4", fps=25, codec="libx264", quality=7, macro_block_size=2)
try:
    for k in range(round(a.seconds / 0.02)):
        t = k * 0.02
        cmd = command(t)
        tripped = body.apply(cmd[0] * a.forward_scale, cmd[1])
        if a.push and k in (600, 800):
            angle = np.random.default_rng([a.seed, k]).uniform(0., 2 * np.pi)
            body.push(a.push * np.cos(angle), a.push * np.sin(angle))
        row = body.advance()
        foot_positions = [np.asarray(f.get_world_poses()[0])[0].tolist() for f in feet]
        forces = [np.asarray(f.get_net_contact_forces(dt=world.get_physics_dt()))[0].tolist() for f in feet]
        record = {"t": round(body.t, 6), "command": cmd, "tripped": tripped,
                  "root_position": np.asarray(body.art.get_world_pose()[0]).tolist(),
                  "root_quaternion": np.asarray(body.art.get_world_pose()[1]).tolist(),
                  "linear_velocity": np.asarray(body.art.get_linear_velocity()).tolist(),
                  "angular_velocity": np.asarray(body.art.get_angular_velocity()).tolist(),
                  "q": np.asarray(body.art.get_joint_positions()).tolist(),
                  "dq": np.asarray(body.art.get_joint_velocities()).tolist(),
                  "feet": foot_positions, "contact_force": forces,
                  "tilt": row["tilt"], "z": row["z"], "finite": row["finite"]}
        trace.append(record)
        if k % 2 == 0:
            world.render()
            frame = np.asarray(camera.get_rgba())
            if frame.size == 0:
                raise RuntimeError("review camera returned no frame")
            video.append_data(frame[..., :3].astype(np.uint8))
        if body.fallen(row):
            fell = True
            print("FALL", a.controller, body.t, row, flush=True)
            break
finally:
    video.close()
    (a.out / "trace.jsonl").write_text("\n".join(json.dumps(x) for x in trace) + "\n")
    metadata = {"controller": a.controller, "seed": a.seed, "fell": fell, "seconds": body.t,
                "joint_names": list(body.art.dof_names), "camera_eye": eye.tolist(),
                "camera_aim": aim.tolist(), "fps": 25, "trace_hz": 50,
                "engine": "Isaac Sim 5.1 / PhysX", "playback_speed": 1,
                "physics_dt": world.get_physics_dt(), "policy_sha256": hashlib.sha256(
                    (policy / "policy.pt").read_bytes()).hexdigest()}
    metadata["forward_command_scale"] = a.forward_scale
    metadata["initial_joint_noise_std_rad"] = .002
    metadata["push_m_s"] = a.push
    metadata["target_filter"] = a.target_filter if a.controller == "booster" else None
    metadata["training_gains"] = a.training_gains
    metadata["gait_hz"] = a.gait_hz
    (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("REVIEW", json.dumps(metadata), flush=True)

from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
