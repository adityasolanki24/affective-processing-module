"""K4: two identical NERVA robots, one continuous lifetime each, different early experience.

    python k_demo.py --system DT_CAM_VOI --seed-index 0 --out runs/K/demo/ [--render]
    python k_demo.py --controls --n 10 --out runs/K/demo_controls.jsonl      (control conditions, no video)

**Both robots are identical.** Same controller, initial internal state, physical parameters, goal (handoff at the
entity), simulation seed and entity position. Each lives one continuous lifetime in its own never-reset simulation
(`nerva_continuous.ContinuousNerva`). Only the entity's behaviour during the first stage differs.

| stage | encounters | Robot A meets | Robot B meets |
|---|---|---|---|
| 1 initial experience | 1–10 | a disruptive entity (pushy) | a reliable entity (calm) |
| 2–3 same present situation; consequences | 11–16 | pushy | pushy |
| 4–6 change; updating; recovery | 17–36 | calm | calm |

At every encounter the robot:
1. reads its memory;
2. chooses an intent and a number of sensing windows (system policy, `infoseek.choose`);
3. walks, senses and returns in the continuous world;
4. updates its memory from what it observed.

Every number in the video overlay and the plots is read from this recorded trace.

**Controls** (`--controls`), each run for both robots:
- `FULL`: the chosen system;
- `NO_MEMORY_UPDATE`: memory frozen at the prior;
- `NO_EXPRESSION`: memory learns, but the intent is always neutral (one window);
- `SHUFFLED`: each robot's stage-1 observations are applied to the *other* robot's memory, so its history is not its
  own;
- `NO_INFO`: the same mapping without active information gathering (DT or DT_CAM).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
J = REPO / "experiments" / "J_validation_transfer"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "scripts"))
STAGE1, STAGE2 = 10, 16
N_ENC = 36
CAM_OFFSET = 24
W, H, FRAME_EVERY, FPS = 480, 360, 8, 50  # one frame per 8 control steps (0.16 s), played at 50 fps = 8x speed


def _seeds():
    spec = importlib.util.spec_from_file_location("k_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def schedule(robot: str, k: int) -> str:
    if k < STAGE1:
        return "pushy" if robot == "A" else "calm"
    if k < STAGE2:
        return "pushy"
    return "calm"


def stage_of(k: int) -> str:
    return "1 initial experience" if k < STAGE1 else ("2-3 same situation" if k < STAGE2 else "4-6 change, recovery")


class Overlay:
    """Renders one robot's frames with measured quantities from its own trace (visual only)."""

    def __init__(self, path: Path, robot: str):
        self.path, self.robot = path, robot
        self.ff = None
        self.step = 0
        self.text: list[str] = []
        self.arm_until = -1.0

    def extender(self, spec) -> None:
        import mujoco
        body = spec.worldbody.add_body(name="entity", mocap=True, pos=[5.0, 0.0, 0.0])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.07, 0.22, 0], pos=[0, 0, 0.3],
                      rgba=[0.55, 0.55, 0.6, 1], contype=0, conaffinity=0)
        arm = spec.worldbody.add_body(name="arm", mocap=True, pos=[5.0, 5.0, 0.0])
        arm.add_geom(name="arm_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.025, 0.5, 0],
                     rgba=[0.6, 0.6, 0.6, 1], contype=0, conaffinity=0)
        home = spec.worldbody.add_body(name="home", pos=[0, 0, 0])
        home.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.08, 0.002, 0], rgba=[0.3, 0.6, 0.9, 0.6],
                      contype=0, conaffinity=0)

    def _open(self, ctrl) -> None:
        import math

        import mujoco

        from apm_transfer.nerva_world import TARGET
        m, d = ctrl.sim.model, ctrl.sim.data
        self.ox, self.oy = ctrl._origin
        d.mocap_pos[m.body_mocapid[m.body("entity").id]] = [self.ox + TARGET[0], self.oy + TARGET[1], 0.0]
        self.arm_mocap, self.arm_geom = m.body_mocapid[m.body("arm").id], m.geom("arm_geom").id
        self.renderer = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [self.ox + 0.3, self.oy, 0.15]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.0, 90.0, -22.0
        self.math = math
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                    f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-crf", "22", str(self.path)], stdin=subprocess.PIPE)

    def hook(self, ctrl, info: dict) -> None:
        from PIL import Image, ImageDraw

        from apm_transfer.nerva_world import TARGET
        if self.ff is None:
            self._open(ctrl)
        m, d = ctrl.sim.model, ctrl.sim.data
        if info["lunge"] is not None:
            reach, push = info["lunge"]
            x, y, _ = ctrl.pose()
            ang = self.math.atan2(y - TARGET[1], x - TARGET[0])
            half = reach / 2
            d.mocap_pos[self.arm_mocap] = [self.ox + TARGET[0] + half * self.math.cos(ang),
                                           self.oy + TARGET[1] + half * self.math.sin(ang), 0.3]
            c, sn, r = self.math.cos(ang / 2), self.math.sin(ang / 2), self.math.sqrt(0.5)
            d.mocap_quat[self.arm_mocap] = [c * r, -sn * r, c * r, sn * r]
            m.geom_size[self.arm_geom][1] = half
            m.geom_rgba[self.arm_geom] = [0.85, 0.15, 0.1, 1] if push > 0 else [0.6, 0.6, 0.6, 1]
            self.arm_until = ctrl.t + 0.5
        elif ctrl.t > self.arm_until:
            d.mocap_pos[self.arm_mocap] = [self.ox + 5.0, self.oy + 5.0, 0.0]
        self.step += 1
        if self.step % FRAME_EVERY:
            return
        self.renderer.update_scene(d, self.cam)
        img = Image.fromarray(self.renderer.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 16 + 14 * len(self.text)], fill=(255, 255, 255))
        dr.text((6, 2), f"Robot {self.robot}   t {ctrl.t:6.1f} s   {info['phase']}   distance {info['dist']:.2f} m",
                fill=(0, 0, 0))
        for j, line in enumerate(self.text):
            dr.text((6, 16 + 14 * j), line, fill=(0, 0, 0))
        assert self.ff is not None and self.ff.stdin is not None
        self.ff.stdin.write(img.tobytes())

    def close(self) -> None:
        if self.ff is not None and self.ff.stdin is not None:
            self.ff.stdin.close()
            self.ff.wait()
            self.renderer.close()


def lifetime(args: tuple) -> dict:
    robot, system, condition, seed_index, cfg, render_dir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from k_run import SYSTEMS

    from apm_express.intent import ExpressionIntent
    from apm_express.states import Probe
    from apm_express.styles import NEUTRAL
    from apm_revise.belief import TypeModel
    from apm_transfer.expression_map import GRID, TimeModel
    from apm_transfer.infoseek import Emissions, Task, choose
    from apm_transfer.nerva_continuous import ContinuousNerva
    from apm_transfer.nerva_memory import TypeStateModel
    from apm_transfer.nerva_world import MIN_QUIET_FRAMES, Camera

    spec = SYSTEMS[system]
    S = _seeds()
    base = int(S.world_seed("K4", "D" if condition == "FULL" and render_dir else "X", seed_index)
               .generate_state(1)[0])
    mcfg = json.loads(Path(cfg["types_model"]).read_text())
    model = TypeModel.from_dict(mcfg["model"])
    em = Emissions.from_model(model.E, camera_block=True)
    em_used = em if spec["camera"] else Emissions(em.imu, None)
    edges = json.loads(Path(cfg["edges"]).read_text())["acc8"]
    tm = TimeModel(json.loads(Path(cfg["time_model"]).read_text())["time_coef"])
    task = Task(cfg["task"], **{k: v for k, v in cfg["task_spec"].items() if k != "why"})
    channels = json.loads(Path(cfg["channels"]).read_text())["interface"]
    overlay = Overlay(Path(render_dir) / f"robot_{robot}.mp4", robot) if render_dir else None
    world = ContinuousNerva(base, channels, camera=Camera() if spec["camera"] else None,
                            scene_extender=overlay.extender if overlay else None,
                            frame_hook=overlay.hook if overlay else None)
    mem = TypeStateModel(model, spec["lam"], transition_each_meeting=cfg.get("transition_fix", True))
    rng = np.random.default_rng(base + 3)
    pol = spec["policy"] if condition != "NO_INFO" else "greedy"
    rows = []
    other = cfg.get("other_stage1", [])
    for k in range(cfg.get("n_enc", N_ENC)):
        etype = schedule(robot, k)
        b = mem.belief("E1", "E")
        rd = mem.readout(Probe("E1", "E", "plain"))
        if condition == "NO_EXPRESSION":
            grid_i, w, diag = None, 1, {}
            intent = ExpressionIntent(**{**NEUTRAL.as_dict(), "proximity": 0.5}).sanitized()
        else:
            grid_i, w, diag = choose(pol, b, model.pi, tm, em_used, task, lam=spec["lam"],
                                     horizon=cfg["voi"]["horizon"], gap=1.0, rng=rng,
                                     legacy=cfg.get("legacy_cap", False))
            intent = GRID[2] if grid_i < 0 else GRID[grid_i]  # OBSERVE / HOLD stand off beyond reach
        standoff = diag.get("standoff", 0.0)
        if overlay:
            overlay.text = [f"encounter {k + 1}/{N_ENC}   stage {stage_of(k)}   entity now: {etype} (sim label)",
                            f"belief P(disruptive) {b[-1]:.2f}   uncertainty {rd.u:.2f}",
                            f"intent: stop at {0.6 - 0.4 * intent.proximity + standoff:.2f} m"
                            + (" (observe, out of reach)" if standoff else "") + f", tempo {intent.tempo:.1f}, "
                            f"sensing windows {w}"
                            + (f", info value {diag.get('voi', 0):.1f}" if "voi" in diag else ""),
                            f"last: stopped {rows[-1]['d_stop']:.2f} m, contacts {rows[-1]['contacts']}, "
                            f"informative obs {rows[-1]['informative']}" if rows else "last: -"]
        out = world.encounter(etype, intent, w, standoff=standoff)
        d = out["d_stop"]
        band = int(np.searchsorted((0.3, 0.5), d))
        imu = [int(np.searchsorted(edges, e[1], side="right")) for e in out["windows"] if e[10] >= MIN_QUIET_FRAMES]
        if standoff > 0:
            imu = []  # out of reach: IMU silence is not evidence
        cats = [band * 8 + c for c in imu] + ([CAM_OFFSET + c for c in out["camera"]] if spec["camera"] else [])
        if condition == "SHUFFLED" and k < STAGE1 and other:
            cats = other[k]
        if condition != "NO_MEMORY_UPDATE":
            mem.observe_many("E1", "E", cats, float(any(e[9] for e in out["windows"])))
        rows.append({"k": k, "stage": stage_of(k), "type": etype, "b_before": [round(float(v), 4) for v in b],
                     "p_disruptive": round(float(b[-1]), 4), "u": round(rd.u, 3), "grid_i": grid_i, "windows": w,
                     "proximity": round(intent.proximity, 3), "voi": round(diag.get("voi", 0.0), 2),
                     "standoff": standoff, "no_safe_action": bool(diag.get("no_safe_action", False)),
                     "d_stop": d, "t": out["t"], "t_arrive": out["t_arrive"], "aborted": out["aborted"],
                     "imu_cats": imu, "cam_codes": out["camera"], "informative": len(cats), "cats": cats,
                     "contacts": out["truth"]["contacts"], "max_push": out["truth"]["max_push"],
                     "fell": out["fell"], "trips": out["trips"], "task_success": task.p_success(d)})
        if out["fell"]:
            break
    if overlay:
        overlay.close()
    return {"robot": robot, "system": system, "condition": condition, "seed_index": seed_index, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="DT_CAM_VOI")
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--types-model", default=str(HERE / "configs" / "K_types.json"))
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-enc", type=int, default=N_ENC, help="shorter lifetimes, for smoke tests only")
    a = ap.parse_args()
    tasks = json.loads((HERE / "configs" / "K_tasks.json").read_text())
    cfg = {"types_model": a.types_model, "edges": str(J / "configs" / "J4_edges.json"),
           "time_model": str(J / "configs" / "J6_model.json"), "channels": str(J / "configs" /
                                                                                 "H_admitted_interface.json"),
           "task": tasks["recovery_study_task"], "task_spec": tasks["tasks"][tasks["recovery_study_task"]],
           "voi": tasks["voi"], "n_enc": a.n_enc}
    t = time.time()
    if not a.controls:
        a.out.mkdir(parents=True, exist_ok=True)
        jobs = [(r, a.system, "FULL", a.seed_index, cfg, str(a.out) if a.render else "") for r in ("A", "B")]
        with Pool(2) as pool:
            res = pool.map(lifetime, jobs)
        (a.out / "trace.json").write_text(json.dumps(res), encoding="utf-8")
        if a.render:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a.out / "robot_A.mp4"), "-i",
                            str(a.out / "robot_B.mp4"), "-filter_complex", "[0:v][1:v]hstack=shortest=0",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                            str(a.out / "demo_side_by_side.mp4")],
                           check=True)
        print("done in", round(time.time() - t, 1), "s ->", a.out)
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    nocond = "DT_CAM" if SYSTEMS_CAM(a.system) else "DT"
    with open(a.out, "w", encoding="utf-8") as fh, Pool(a.workers) as pool:
        # pass 1: FULL for both robots (their stage-1 observations feed the SHUFFLED control)
        full = pool.map(lifetime, [(r, a.system, "FULL", i, cfg, "") for i in range(a.n) for r in ("A", "B")])
        for r in full:
            fh.write(json.dumps(r) + "\n")
        st1 = {(r["robot"], r["seed_index"]): [x["cats"] for x in r["rows"][:STAGE1]] for r in full}
        jobs = []
        for i in range(a.n):
            for r in ("A", "B"):
                for cond in ("NO_MEMORY_UPDATE", "NO_EXPRESSION", "SHUFFLED"):
                    c2 = dict(cfg, other_stage1=st1[("B" if r == "A" else "A", i)])
                    jobs.append((r, a.system, cond, i, c2, ""))
                jobs.append((r, nocond, "NO_INFO", i, cfg, ""))
        for r in pool.imap_unordered(lifetime, jobs, chunksize=1):
            fh.write(json.dumps(r) + "\n")
    print("done", "in", round(time.time() - t, 1), "s ->", a.out)


def SYSTEMS_CAM(system: str) -> bool:  # noqa: N802
    from k_run import SYSTEMS
    return bool(SYSTEMS[system]["camera"])


if __name__ == "__main__":
    main()
