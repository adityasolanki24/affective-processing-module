"""M3: the Booster T1's MuJoCo-trained walking policy on Isaac Sim's stock T1 USD (PhysX), with no retraining.

    $ISAAC_PY experiments/M_release/isaac/m3_t1.py --out out/m3_t1 [--video]

**What is matched to the MuJoCo training model** (`configs/t1_mujoco_ref.json`):
- joint order, by name;
- PD gains: drive stiffness = kp, drive damping = MuJoCo joint damping;
- armature;
- home pose;
- control period: 0.02 s (10 physics steps of 0.002 s);
- the policy's observation: local linear velocity, gyro, gravity in the trunk frame, command, joint offsets, joint
  velocities, last action, gait phase.

The policy is the exact NumPy copy validated in L A7.

**Battery** (each scenario from a fresh reset):
- stand 4 s;
- walk forward at 0.2 / 0.4 m/s for 8 s;
- turn at 0.5 rad/s;
- stop/start;
- pushes of 0.3 m/s while walking.

Falls are base height < 0.45 m or tilt > 50°. Recorded: tracking error, tilt, base height, falls. `--video`
writes a chase-camera MP4 of the walk scenario.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--video", action="store_true")
ap.add_argument("--friction", type=float, default=1.0)
ap.add_argument("--physx-lab", action="store_true", help="Isaac Lab articulation settings (see robots.py)")
a, _ = ap.parse_known_args()
launcher = AppLauncher({"headless": True, "enable_cameras": True})
app = launcher.app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for _ext in ("isaacsim.sensors.camera", "isaacsim.sensors.physics", "isaacsim.storage.native"):
    enable_extension(_ext)

import numpy as np  # noqa: E402
import omni.client  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
REF = json.loads((REPO / "experiments" / "M_release" / "configs" / "t1_mujoco_ref.json").read_text())
POL = np.load(REPO / "experiments" / "K_recovery" / "humanoid" / "trained" / "t1_policy.npz")
out = a.out
out.mkdir(parents=True, exist_ok=True)
log: dict = {"ref": REF["source"]}

root = get_assets_root_path()
d = f"{root}/Isaac/Robots/BoosterRobotics/BoosterT1"
_, entries = omni.client.list(d)
usds = [e.relative_path for e in entries if e.relative_path.endswith(".usd")]
log["usd_candidates"] = usds
usd = f"{d}/{sorted(usds, key=len)[0]}"
log["usd"] = usd
print("T1 usd", usd, usds, flush=True)

mean, std = POL["mean"], POL["std"]
N_LAYERS = len([k for k in POL.files if k.endswith("_kernel")])
W = [(POL[f"hidden_{i}_kernel"], POL[f"hidden_{i}_bias"]) for i in range(N_LAYERS)]


def policy(obs: np.ndarray) -> np.ndarray:
    x = (obs - mean) / std
    for i, (w, b) in enumerate(W):
        x = x @ w + b
        if i < len(W) - 1:
            x = x / (1.0 + np.exp(-x))
    return np.tanh(x[: x.shape[0] // 2])


def quat_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


world = World(stage_units_in_meters=1.0, physics_dt=REF["timestep"], rendering_dt=REF["ctrl_dt"])
world.scene.add_default_ground_plane(static_friction=a.friction, dynamic_friction=a.friction, restitution=0.0)
add_reference_to_stage(usd, "/World/T1")
stage = world.stage
art_root = None
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith("/World/T1") and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_root = str(prim.GetPath())
        break
log["articulation_root"] = art_root
if a.physx_lab:
    sys.path.insert(0, str(REPO / "src"))
    from apm_isaac.robots import physx_like_isaac_lab
    physx_like_isaac_lab(world.stage, "/World/T1")
robot = world.scene.add(SingleArticulation(art_root or "/World/T1", name="t1"))
world.reset()
names = list(robot.dof_names)
log["isaac_dofs"] = names
mj_joints = [x["joint"] for x in REF["actuators"]]
missing = [j for j in mj_joints if j not in names]
log["missing_in_isaac"] = missing
extra = [j for j in names if j not in mj_joints]
log["extra_in_isaac"] = extra
print("dofs", len(names), "missing", missing, "extra", extra, flush=True)
if missing:
    (out / "report.json").write_text(json.dumps(log, indent=1))
    app.close()
    sys.exit(0)
idx = np.array([names.index(j) for j in mj_joints])
kp = np.array([x["kp"] for x in REF["actuators"]])
kd = np.array([x["damping"] for x in REF["actuators"]])
home = np.array([x["home"] for x in REF["actuators"]])
# drives: stiffness/damping per joint; armature and no extra friction (MuJoCo frictionloss has no PhysX equivalent)
for x in REF["actuators"]:
    path = None
    for prim in stage.Traverse():
        if prim.GetName() == x["joint"] and prim.IsA(UsdPhysics.Joint):
            path = prim
            break
    if path is not None:
        PhysxSchema.PhysxJointAPI.Apply(path).CreateArmatureAttr(x["armature"])
full_kp, full_kd = np.zeros(len(names)), np.zeros(len(names))
full_kp[idx], full_kd[idx] = kp, kd
robot.get_articulation_controller().set_gains(kps=full_kp, kds=full_kd)
log["masses_total"] = float(sum(robot.get_articulation_body_masses() or [0])) if hasattr(
    robot, "get_articulation_body_masses") else None


def reset() -> None:
    world.reset()
    robot.get_articulation_controller().set_gains(kps=full_kp, kds=full_kd)
    q = np.zeros(len(names))
    q[idx] = home
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros(len(names)))
    robot.set_world_pose(position=np.array(REF["home_base_pos"]), orientation=np.array(REF["home_base_quat"]))
    robot.apply_action(ArticulationAction(joint_positions=q))


def run(name: str, script: list[tuple[float, float, float]], pushes: list[tuple[float, float, float]] | None = None,
        frames: list | None = None, cam=None) -> dict:
    reset()
    last = np.zeros(23)
    phase = np.array([0.0, math.pi])
    dphase = 2 * math.pi * REF["ctrl_dt"] * 1.5
    t, fell, track, tilts, zs = 0.0, False, [], [], []
    for v, w_, dur in script:
        for _ in range(int(dur / REF["ctrl_dt"])):
            pos, quat = robot.get_world_pose()
            R = quat_rot(np.asarray(quat, float))
            lin = R.T @ np.asarray(robot.get_linear_velocity(), float)
            gyro = R.T @ np.asarray(robot.get_angular_velocity(), float)
            grav = R.T @ np.array([0.0, 0.0, -1.0])
            q = np.asarray(robot.get_joint_positions(), float)[idx]
            dq = np.asarray(robot.get_joint_velocities(), float)[idx]
            cmd = np.array([v, 0.0, w_])
            ph = phase if np.linalg.norm(cmd) > 0.01 else np.ones(2) * math.pi
            obs = np.concatenate([lin, gyro, grav, cmd, q - home, dq, last, np.cos(ph), np.sin(ph)])
            act = policy(obs)
            target = np.zeros(len(names))
            target[idx] = home + act
            robot.apply_action(ArticulationAction(joint_positions=target))
            for p in pushes or []:
                if abs(t - p[0]) < REF["ctrl_dt"] / 2:
                    vel = np.asarray(robot.get_linear_velocity(), float)
                    robot.set_linear_velocity(vel + np.array([p[1], p[2], 0.0]))
            for _k in range(int(round(REF["ctrl_dt"] / REF["timestep"]))):
                world.step(render=False)
            if frames is not None and cam is not None:
                cam.set_world_pose(position=np.asarray(pos) + np.array([-2.2, -1.2, 0.9]))
                world.render()
                frames.append(cam.get_rgba()[..., :3].copy())
            last = act
            phase = np.fmod(phase + dphase + math.pi, 2 * math.pi) - math.pi
            t += REF["ctrl_dt"]
            tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
            tilts.append(tilt)
            zs.append(float(pos[2]))
            if pos[2] < 0.45 or tilt > 50:
                fell = True
                break
            if t > 2.0 and abs(v) + abs(w_) > 0:
                yaw = math.atan2(R[1, 0], R[0, 0])
                vw = np.asarray(robot.get_linear_velocity(), float)
                track.append((vw[0] * math.cos(yaw) + vw[1] * math.sin(yaw) - v,
                              float(np.asarray(robot.get_angular_velocity())[2]) - w_))
        if fell:
            break
    tr = np.array(track) if track else np.zeros((0, 2))
    res = {"fell": fell, "t": round(t, 2), "max_tilt": round(max(tilts), 1), "min_z": round(min(zs), 3),
           "v_err_mean": round(float(tr[:, 0].mean()), 3) if len(tr) else None,
           "w_err_mean": round(float(tr[:, 1].mean()), 3) if len(tr) else None}
    print(name, res, flush=True)
    return res


battery = {"stand": [(0.0, 0.0, 4.0)], "walk_0.2": [(0.2, 0.0, 8.0)], "walk_0.4": [(0.4, 0.0, 8.0)],
           "turn_0.5": [(0.2, 0.5, 8.0)], "stop_start": [(0.3, 0.0, 2.0), (0.0, 0.0, 2.0), (0.3, 0.0, 2.0),
                                                        (0.0, 0.0, 2.0)]}
log["battery"] = {}
t0 = time.time()
for name, sc in battery.items():
    log["battery"][name] = run(name, sc)
log["battery"]["push_walk_0.3"] = run("push_walk_0.3", [(0.2, 0.0, 10.0)],
                                      pushes=[(3.0, 0.3, 0.0), (5.0, 0.0, 0.3), (7.0, -0.3, 0.0)])
log["wall_s"] = round(time.time() - t0, 1)
if a.video:
    from isaacsim.sensors.camera import Camera
    cam = Camera(prim_path="/World/chase", resolution=(960, 540))
    cam.initialize()
    frames: list = []
    run("video_walk", [(0.0, 0.0, 1.0), (0.3, 0.0, 6.0), (0.2, 0.5, 4.0), (0.0, 0.0, 2.0)], frames=frames, cam=cam)
    import imageio
    imageio.mimsave(out / "t1_walk.mp4", [np.asarray(f, np.uint8) for f in frames if f is not None and f.size],
                    fps=50)
(out / "report.json").write_text(json.dumps(log, indent=1))
app.close()
