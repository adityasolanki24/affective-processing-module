"""L protocol §G: a continuous demonstration on the corrected runtime, with ablations.

    python g_demo.py run --n 10 --workers 80 --out <jsonl>                    (conditions × robots × seeds; NERVA venv)
    python g_demo.py render --index 0 --out runs/L/G_demo/                    (the pre-declared demo lifetime, video)
    python g_demo.py analyse --runs <jsonl> --out results/G_demo.json

**Two identical NERVA robots**, each living one never-reset continuous simulation (`ContinuousNerva`, vision mode).
- **Shared:** controller, initial state, physics, goal (the hand-over distance), simulation seed and entity position.
- **Sensing:** IMU + rendered vision, so each robot sees the entity column through its own depth eye.
- **Decisions:** the integrated runtime (`apm_transfer.runtime`), all §B corrections.
- Only the entity's behaviour in stage 1 differs.

| stage | encounters | Robot A meets | Robot B meets |
|---|---|---|---|
| 1 experience | 1–10 | disruptive (pushy) | reliable (calm) |
| 2 same present situation | 11–16 | pushy | pushy |
| 3 change | 17–36 | calm | calm |

**Conditions:**
- FULL;
- NO_MEMORY: the runtime never learns, so its belief stays the prior;
- NO_EXPRESSION: memory learns, but the robot always approaches with a neutral intent and one window;
- NO_INFO: the same runtime without information gathering (greedy policy).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parents[1]
K = REPO / "experiments" / "K_recovery"
J = REPO / "experiments" / "J_validation_transfer"
STAGE1, STAGE2, N_ENC = 10, 16, 36
CONDITIONS = ("FULL", "NO_MEMORY", "NO_EXPRESSION", "NO_INFO")
W, H, FRAME_EVERY, FPS = 640, 360, 8, 50


def _seeds():
    spec = importlib.util.spec_from_file_location("l_seeds", HERE / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def schedule(robot: str, k: int) -> str:
    if k < STAGE1:
        return "pushy" if robot == "A" else "calm"
    return "pushy" if k < STAGE2 else "calm"


def stage_of(k: int) -> int:
    return 1 if k < STAGE1 else 2 if k < STAGE2 else 3


class Overlay:
    """Video only: a third-person view with the entity column where the world puts it, the measured trace as text,
    and an inset of the robot's own depth image. Nothing here feeds back into the simulation."""

    def __init__(self, path: Path, robot: str):
        self.path, self.robot, self.ff, self.step = path, robot, None, 0
        self.text: list[str] = []
        self.world = None

    def _open(self, ctrl) -> None:
        import mujoco
        m = ctrl.sim.model
        self.mj = mujoco
        self.renderer = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        ox, oy = ctrl._origin
        self.cam.lookat[:] = [ox + 0.25, oy, 0.25]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.3, 90.0, -20.0
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                    f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-crf", "22", str(self.path)], stdin=subprocess.PIPE)

    def hook(self, ctrl, info: dict) -> None:
        from PIL import Image, ImageDraw

        from apm_transfer.vision import ENTITY_H, ENTITY_R
        if self.ff is None:
            self._open(ctrl)
        self.step += 1
        if self.step % FRAME_EVERY:
            return
        r = self.renderer
        r.update_scene(ctrl.sim.data, self.cam)
        scn = r.scene
        if self.world is not None and scn.ngeom < scn.maxgeom:
            self.mj.mjv_initGeom(scn.geoms[scn.ngeom], int(self.mj.mjtGeom.mjGEOM_CYLINDER),
                                 np.array([ENTITY_R, ENTITY_H / 2, 0.0]), np.array(self.world.entity_centre()),
                                 np.eye(3).flatten(), np.array([0.85, 0.35, 0.2, 1.0], np.float32))
            scn.ngeom += 1
        img = Image.fromarray(r.render())
        vis = getattr(self.world, "vis", None)
        depth = getattr(vis, "last_depth", None) if vis is not None else None
        if depth is not None:  # the robot's own depth view (near = bright)
            g = np.clip(1.0 - np.asarray(depth, float) / 1.5, 0, 1)
            g[np.asarray(depth) <= 0.02] = 0
            inset = Image.fromarray((g * 255).astype(np.uint8)).resize((192, 144), Image.NEAREST).convert("RGB")
            img.paste(inset, (W - 196, H - 148))
            ImageDraw.Draw(img).text((W - 196, H - 162), "robot's depth view", fill=(0, 0, 0))
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
    robot, condition, partition, index, render_dir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from apm_express.intent import ExpressionIntent
    from apm_express.styles import NEUTRAL
    from apm_transfer.infoseek import Task
    from apm_transfer.nerva_continuous import ContinuousNerva
    from apm_transfer.nerva_world import MIN_QUIET_FRAMES
    from apm_transfer.runtime import DIST_EDGES, Observation, Runtime
    from apm_transfer.vision import VisionCamera

    base = int(_seeds().world_seed("LG", partition, index).generate_state(1)[0])
    model = json.loads((HERE / "configs" / "LC_nerva_vision_types.json").read_text())
    time_coef = json.loads((J / "configs" / "J6_model.json").read_text())["time_coef"]
    edges = json.loads((J / "configs" / "J4_edges.json").read_text())["acc8"]
    channels = json.loads((J / "configs" / "H_admitted_interface.json").read_text())["interface"]
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())
    task = Task("handoff", **{k: v for k, v in tasks["tasks"]["handoff"].items() if k != "why"})
    rt = Runtime(model, time_coef, task, tasks["voi"], seed=base, policy="greedy" if condition == "NO_INFO" else "voi")
    overlay = Overlay(Path(render_dir) / f"robot_{robot}.mp4", robot) if render_dir else None
    world = ContinuousNerva(base, channels, camera=VisionCamera(), frame_hook=overlay.hook if overlay else None)
    if overlay is not None:
        overlay.world = world
    r_vec = np.asarray(model["model"]["r"], float)
    mid = float((np.asarray(model["oracle"]["calm"]) @ r_vec + np.asarray(model["oracle"]["pushy"]) @ r_vec) / 2)
    rows = []
    for k in range(N_ENC):
        etype = schedule(robot, k)
        d = rt.decide("E1", "E")
        if condition == "NO_EXPRESSION":
            intent, w, standoff = ExpressionIntent(**{**NEUTRAL.as_dict(), "proximity": 0.5}).sanitized(), 1, 0.0
        else:
            intent, w, standoff = d.intent(), d.windows, d.standoff
        if overlay is not None:
            stop = 0.6 - 0.4 * intent.proximity + standoff
            overlay.text = [f"encounter {k + 1}/{N_ENC}   stage {stage_of(k)}   entity now: {etype} (simulator label)",
                            f"belief: expected disruption {d.p_risk:.2f}   uncertainty {d.uncertainty:.2f}",
                            f"decision: {d.reason}; stop at {stop:.2f} m, {w} sensing window(s)",
                            (f"last: stopped {rows[-1]['d_stop']:.2f} m, contacts {rows[-1]['contacts']}, "
                             f"camera codes {rows[-1]['camera']}") if rows else "last: -"]
        out = world.encounter(etype, intent, w, standoff=standoff)
        dd = out["d_stop"]
        band = int(np.searchsorted(DIST_EDGES, dd))
        imu = () if standoff > 0 else tuple(int(np.searchsorted(edges, e[1], side="right")) for e in out["windows"]
                                            if e[10] >= MIN_QUIET_FRAMES)
        obs = Observation("E1", "E", band, imu, tuple(out["camera"]), any(bool(e[9]) for e in out["windows"]))
        if condition != "NO_MEMORY":
            rt.observe(obs)
        rows.append({"k": k, "stage": stage_of(k), "type": etype, "p_risk": round(d.p_risk, 4),
                     "risky_belief": d.p_risk > mid, "action": d.action, "reason": d.reason, "windows": w,
                     "standoff": standoff, "d_stop": dd, "imu": list(imu), "camera": out["camera"],
                     "contacts": out["truth"]["contacts"], "max_push": out["truth"]["max_push"],
                     "fell": out["fell"], "trips": out["trips"], "t": out["t"]})
        if out["fell"]:
            break
    if overlay is not None:
        overlay.close()
    return {"robot": robot, "condition": condition, "partition": partition, "index": index, "rows": rows}


def onset(ok: list[bool]) -> int | None:
    for j in range(len(ok) - 1):
        if ok[j] and ok[j + 1]:
            return j + 1
    return None


def analyse(paths: list[Path], out: Path) -> None:
    import k_analyze as KA
    lts = [json.loads(x) for p in paths for x in p.read_text(encoding="utf-8").splitlines()]
    by: dict = {}
    for lt in lts:
        by.setdefault(lt["condition"], {}).setdefault(lt["index"], {})[lt["robot"]] = lt["rows"]
    res: dict = {}
    for cond, seeds in by.items():
        st2_stop, st2_con, rec, rev, falls = [], [], {"A": [], "B": []}, {"A": [], "B": []}, 0
        for _, rob in seeds.items():
            if "A" not in rob or "B" not in rob:
                continue
            s2 = {r: [x for x in rob[r] if x["stage"] == 2] for r in ("A", "B")}
            st2_stop.append(np.mean([x["d_stop"] for x in s2["A"]]) - np.mean([x["d_stop"] for x in s2["B"]]))
            st2_con.append(np.mean([x["contacts"] for x in s2["A"]]) - np.mean([x["contacts"] for x in s2["B"]]))
            for r in ("A", "B"):
                s3 = [x for x in rob[r] if x["stage"] == 3]
                rec[r].append(onset([x["d_stop"] <= 0.35 for x in s3]))
                rev[r].append(onset([not x["risky_belief"] for x in s3]))
                falls += sum(x["fell"] for x in rob[r])

        def summ(v: list) -> dict:
            got = [x for x in v if x is not None]
            return {"recovered": len(got), "of": len(v), "median": float(np.median(got)) if got else None}
        res[cond] = {"stage2_stop_A_minus_B": KA.boot(st2_stop), "stage2_contacts_A_minus_B": KA.boot(st2_con),
                     "stage3_recovery": {r: summ(rec[r]) for r in ("A", "B")},
                     "stage3_revision": {r: summ(rev[r]) for r in ("A", "B")}, "falls": falls}
    f = res.get("FULL", {})
    checks = {}
    if f:
        checks["G1_matched_difference"] = (f["stage2_stop_A_minus_B"]["ci95"][0] > 0
                                           and f["stage2_contacts_A_minus_B"]["ci95"][1] < 0)
        checks["G2_both_recover"] = all(f["stage3_recovery"][r]["recovered"] >= 0.8 * f["stage3_recovery"][r]["of"]
                                        for r in ("A", "B"))
    if "NO_MEMORY" in res:
        lo, hi = res["NO_MEMORY"]["stage2_stop_A_minus_B"]["ci95"]
        checks["G3_memory_free_no_difference"] = lo <= 0 <= hi
    if "NO_EXPRESSION" in res:
        checks["G4_expression_free_no_protection"] = res["NO_EXPRESSION"]["stage2_contacts_A_minus_B"]["ci95"][1] >= 0
    if "NO_INFO" in res and f:
        fr = sum(f["stage3_recovery"][r]["recovered"] for r in ("A", "B"))
        nr = sum(res["NO_INFO"]["stage3_recovery"][r]["recovered"] for r in ("A", "B"))
        checks["G5_info_seeking_helps_recovery"] = nr < fr
    res["checks"] = checks
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for cond, o in res.items():
        if cond != "checks":
            print(cond, "stage2 stop A-B", round(o["stage2_stop_A_minus_B"]["estimate"], 3),
                  np.round(o["stage2_stop_A_minus_B"]["ci95"], 3), "contacts A-B",
                  round(o["stage2_contacts_A_minus_B"]["estimate"], 2),
                  np.round(o["stage2_contacts_A_minus_B"]["ci95"], 2),
                  "recovery", o["stage3_recovery"], "falls", o["falls"])
    print(checks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "render", "analyse"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--runs", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.mode == "analyse":
        sys.path.insert(0, str(K / "scripts"))
        analyse(a.runs, a.out)
        return
    if a.mode == "render":
        a.out.mkdir(parents=True, exist_ok=True)
        with Pool(2) as p:
            res = p.map(lifetime, [(r, "FULL", "D", a.index, str(a.out)) for r in ("A", "B")])
        (a.out / "trace.json").write_text(json.dumps(res), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a.out / "robot_A.mp4"), "-i",
                        str(a.out / "robot_B.mp4"), "-filter_complex", "[0:v][1:v]vstack=shortest=0", "-c:v",
                        "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(a.out / "G_paired.mp4")], check=True)
        print("rendered ->", a.out)
        return
    jobs = [(r, c, "X", i, "") for i in range(a.n) for c in a.conditions.split(",") for r in ("A", "B")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p, open(a.out, "w", encoding="utf-8") as f:
        for r in p.imap_unordered(lifetime, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
    print("done", len(jobs), "->", a.out)


if __name__ == "__main__":
    main()
