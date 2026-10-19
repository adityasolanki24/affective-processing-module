"""T1 gait validation battery (REMEDIATION_PROTOCOL.md, "T1 gait validation battery"; frozen before data).

    $ISAAC_PY experiments/M_release/isaac/gait_battery.py --controller v1|candidate --env flat|warehouse|home|hospital \
        --seeds 72001-72005 --out out/gait_battery [--video-seed 72001]

One process runs every scenario and seed for one complete controller stack in one environment. Each run records
the real PhysX state at 50 Hz: root pose and velocity, joint positions, velocities and measured efforts, foot poses
and foot contact forces. Nothing is played back, scripted at the joints or stabilised by external forces. The only
external forces are the protocol's velocity pushes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--controller", choices=("v1", "candidate"), required=True)
ap.add_argument("--env", choices=("flat", "warehouse", "home", "hospital"), required=True)
ap.add_argument("--seeds", default="72001-72005")
ap.add_argument("--scenarios", default="all", help="comma-separated names or 'all'")
ap.add_argument("--video-seed", type=int, default=None, help="record side-view video for this seed")
ap.add_argument("--video-scenarios", default="speed_0.2,turn_walk_+,startstop,push_walk_0.3")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def _scenarios() -> dict[str, dict]:
    """name -> {seconds, command(t) -> (v, w), pushes [(t, m/s)]}; stand still for the first 2 s."""
    s: dict[str, dict] = {}
    for v in (0.1, 0.2, 0.3, 0.4):
        s[f"speed_{v}"] = {"seconds": 18.0, "segments": [(2.0, 14.0, v, 0.0)], "pushes": []}
    for sign, tag in ((1.0, "+"), (-1.0, "-")):
        s[f"turn_walk_{tag}"] = {"seconds": 16.0, "segments": [(2.0, 4.0, 0.2, 0.0), (4.0, 12.0, 0.2, 0.5 * sign)],
                                 "pushes": []}
        s[f"turn_place_{tag}"] = {"seconds": 13.0, "segments": [(2.0, 10.0, 0.0, 0.5 * sign)], "pushes": []}
    s["startstop"] = {"seconds": 24.0, "segments": [(2.0 + 5 * i, 5.0 + 5 * i, 0.2, 0.0) for i in range(4)],
                      "pushes": []}
    for m in (0.2, 0.3, 0.5):
        s[f"push_walk_{m}"] = {"seconds": 18.0, "segments": [(2.0, 18.0, 0.2, 0.0)], "pushes": [(8.0, m), (12.0, m)]}
    s["push_stand_0.3"] = {"seconds": 12.0, "segments": [], "pushes": [(4.0, 0.3), (8.0, 0.3)]}
    return s


SCENARIOS = _scenarios()


def command(spec: dict, t: float) -> tuple[float, float]:
    for t0, t1, v, w in spec["segments"]:
        if t0 <= t < t1:
            return v, w
    return 0.0, 0.0


from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"headless": True, "enable_cameras": a.video_seed is not None}).app
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for ext in ("isaacsim.sensors.camera", "isaacsim.storage.native"):
    enable_extension(ext)

import numpy as np  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from apm_isaac import exit_on_error, hard_exit  # noqa: E402
from apm_isaac.world import _look_at  # noqa: E402

exit_on_error()
M = ROOT / "experiments" / "M_release"
policy = M / "policies" / ("t1_isaaclab_v1" if a.controller == "v1" else "t1_booster_candidate")
physics_dt = 0.005 if a.controller == "v1" else 0.002
lo, hi = (int(x) for x in a.seeds.split("-"))
seeds = list(range(lo, hi + 1))
names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
video_names = set(a.video_scenarios.split(","))

if a.env == "flat":
    from isaacsim.core.api import World
    from pxr import UsdLux
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=0.02)
    world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0)
    UsdLux.DomeLight.Define(world.stage, "/World/Light").CreateIntensityAttr(1500.0)
    if a.controller == "v1":
        from apm_isaac.robots import LabPolicyBody
        body = LabPolicyBody.t1(world, "/World/T1", (0.0, 0.0), 0.0, seeds[0], policy)
    else:
        from apm_isaac.booster import BoosterPolicyBody
        body = BoosterPolicyBody(world, "/World/T1", (0.0, 0.0), 0.0, seeds[0], policy)
    world.reset()
    body.post_reset()
    start = (0.0, 0.0, 0.0)
    stage = world.stage
    free_path = None
else:
    from apm_isaac.world import IsaacWorld, SceneSpec
    cfg = json.loads((M / "configs" / "scenes.json").read_text())[a.env]
    scene = SceneSpec(a.env, cfg["usd"], home=tuple(cfg["home"]),
                      stations={k: tuple(v) for k, v in cfg["stations"].items()}, floor_z=cfg["floor_z"],
                      scale=cfg.get("scale", 1.0), held_out=cfg.get("role", "").startswith("held"))
    iw = IsaacWorld(scene, "humanoid", seed=seeds[0], video=False, characters=None, physics_dt=physics_dt,
                    policy_dir=str(policy), walkers=0)
    world, body, stage = iw.world, iw.robot, iw.world.stage
    hx, hy, _ = scene.home
    stations = [np.array(v) for v in scene.stations.values()]
    free_path = None
    for deg in range(0, 360, 15):  # the protocol's start heading rule
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        pts = [np.array([hx, hy]) + f * u for f in np.linspace(0.0, 4.0, 17)]
        if all(iw._free(float(p[0]), float(p[1]), r=0.35) for p in pts) and \
                all(min(np.linalg.norm(p - s) for s in stations) >= 1.0 for p in pts[2:]):
            free_path = deg
            break
    if free_path is None:
        raise RuntimeError(f"no collider-free 4 m heading from the {a.env} hub")
    start = (hx, hy, math.radians(free_path))

feet = []
for side in ("left", "right"):
    prim = next(p for p in stage.Traverse()
                if p.GetName() == f"{side}_foot_link" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    foot = RigidPrim(str(prim.GetPath()), name=f"battery_{side}", track_contact_forces=True)
    foot.initialize()
    feet.append(foot)

camera = None
if a.video_seed is not None:
    from isaacsim.sensors.camera import Camera
    camera = Camera("/World/BatteryCam", resolution=(960, 540))
    camera.initialize()
    camera.set_focal_length(2.4)

a.out.mkdir(parents=True, exist_ok=True)
policy_sha = hashlib.sha256((policy / "policy.pt").read_bytes()).hexdigest()
summary = []
for name in names:
    spec = SCENARIOS[name]
    for seed in seeds:
        world.reset()
        body.post_reset()
        body.teleport(*start)
        body.rng = np.random.default_rng(seed)
        noise = np.random.default_rng([seed, 1]).normal(0.0, 0.002, len(body.art.dof_names))
        body.art.set_joint_positions(np.asarray(body.art.get_joint_positions()) + noise)
        push_rng = np.random.default_rng([seed, 2])
        pushes = {round(t / 0.02): (m, float(push_rng.uniform(0.0, 2 * math.pi))) for t, m in spec["pushes"]}
        writer = None
        if camera is not None and seed == a.video_seed and name in video_names:
            import imageio.v2 as imageio
            writer = imageio.get_writer(a.out / f"{a.env}_{a.controller}_{name}_{seed}.mp4", fps=25,
                                        codec="libx264", quality=7, macro_block_size=2)
        rows, fell, fall_t = [], False, None
        for k in range(round(spec["seconds"] / 0.02)):
            t = k * 0.02
            v, w = command(spec, t)
            tripped = body.apply(v, w)
            if k in pushes:
                m, ang = pushes[k]
                body.push(m * math.cos(ang), m * math.sin(ang))
            row = body.advance()
            pos, quat = body.art.get_world_pose()
            try:
                effort = np.asarray(body.art.get_measured_joint_efforts(), float).reshape(-1)
            except Exception:  # noqa: BLE001 - recorded as missing, not invented
                effort = np.full(len(body.art.dof_names), np.nan)
            rows.append({
                "t": round(t, 3), "cmd": [v, w], "push": pushes.get(k), "tripped": bool(tripped),
                "root": np.asarray(pos, float).round(5).tolist(), "quat": np.asarray(quat, float).round(6).tolist(),
                "lin": np.asarray(body.art.get_linear_velocity(), float).round(5).tolist(),
                "ang": np.asarray(body.art.get_angular_velocity(), float).round(5).tolist(),
                "q": np.asarray(body.art.get_joint_positions(), float).round(5).tolist(),
                "dq": np.asarray(body.art.get_joint_velocities(), float).round(4).tolist(),
                "effort": effort.round(3).tolist(),
                "feet": [np.asarray(f.get_world_poses()[0], float)[0].round(5).tolist() for f in feet],
                "force": [np.asarray(f.get_net_contact_forces(dt=world.get_physics_dt()), float)[0].round(3).tolist()
                          for f in feet],
                "tilt": row["tilt"], "z": row["z"]})
            if writer is not None and k % 2 == 0:
                x, y, yaw = body.pose()
                side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
                eye = np.array([x, y, 1.3]) - 3.2 * side
                camera.set_world_pose(position=eye, orientation=_look_at(eye, np.array([x, y, 0.6])))
                world.render()
                frame = np.asarray(camera.get_rgba())
                if frame.size:
                    writer.append_data(frame[..., :3].astype(np.uint8))
            if body.fallen(row):
                fell, fall_t = True, t
                break
        if writer is not None:
            writer.close()
        run = f"{a.env}_{a.controller}_{name}_{seed}"
        meta = {"run": run, "env": a.env, "controller": a.controller, "scenario": name, "seed": seed,
                "fell": fell, "fall_t": fall_t, "physics_dt": physics_dt, "trace_hz": 50,
                "start": [round(float(x), 4) for x in start], "spec": spec, "policy_sha256": policy_sha,
                "joint_names": list(body.art.dof_names), "engine": "Isaac Sim 5.1 / PhysX"}
        with open(a.out / f"{run}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": meta}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary.append({"run": run, "fell": fell, "fall_t": fall_t})
        print("RUN", run, "fell" if fell else "ok", f"{rows[-1]['t']:.2f}s", flush=True)

(a.out / f"summary_{a.env}_{a.controller}.json").write_text(json.dumps(summary, indent=1))
print("BATTERY DONE", a.env, a.controller, len(summary), "runs,", sum(s["fell"] for s in summary), "falls", flush=True)
hard_exit(0)
