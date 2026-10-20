"""Paired NERVA videos: the same robot, the same seeds, two different experiences, and what it does afterwards.

    NERVA_ROOT=... PYTHONPATH="src;<NERVA>" <nerva-venv>/python j_videos.py --runs <J6-X lifetimes.jsonl>
        --systems CL,LEARNED_V --out experiments/J_validation_transfer/videos

**Choosing the lifetimes.** For each system, the J6-X index whose late-B1 stopping-distance difference (risky_B − safe,
encounters k ≥ 10) is the median across indices. This picks a typical pair, not the most striking one.

**Re-simulation.** Each chosen lifetime is re-run through `j5_closed_loop.job` with the same arguments as J6-X. Two B1
encounters are rendered:
- **first**: B1's first meeting. Same seed, same prior belief and same intent in both histories; only the entity
  differs (pushy vs calm). This is the *experience*.
- **last**: B1's last meeting. Same seed; the intent now differs because the beliefs differ. This is the *subsequent
  expression*.

Rendering is visual only: one grey entity column at the entity's position, plus an arm that extends by the lunge's
reach (red if it reaches the robot, grey if it falls short). Every re-simulated row is compared with the J6-X log, and
the manifest records whether they match exactly.

**Output:**
- per encounter, `<system>_<index>_<history>_<first|last>.mp4`;
- a side-by-side `<system>_<index>_<first|last>_pair.mp4` (left risky_B, right safe);
- `manifest.json`.

Captions show the history, encounter, entity type, the robot's belief p and the stopping distance. Entity types are
simulator labels for the viewer only. They are not emotions and are never shown to the robot.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import j5_closed_loop as cl  # noqa: E402

W, H, FPS = 480, 360, 25
ARM_S = 0.4
OUT: Path = Path(".")
WANT: dict = {}  # (system, history, index) -> {k: tag}


def _extender(spec) -> None:
    import mujoco

    body = spec.worldbody.add_body(name="entity", mocap=True, pos=[5.0, 0.0, 0.0])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.07, 0.22, 0], pos=[0, 0, 0.3],
                  rgba=[0.55, 0.55, 0.6, 1], contype=0, conaffinity=0)
    arm = spec.worldbody.add_body(name="arm", mocap=True, pos=[5.0, 5.0, 0.0])
    arm.add_geom(name="arm_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.025, 0.5, 0], rgba=[0.6, 0.6, 0.6, 1],
                 contype=0, conaffinity=0)


class Clip:
    def __init__(self, system: str, history: str, index: int, k: int, tag: str, etype: str, p: float, intent) -> None:
        self.name = f"{system}_{index}_{history}_{tag}"
        self.caption = [f"{system} | history {history} | encounter {k + 1}/30: {tag} meeting with B1",
                        f"B1 is {etype} (simulator label, unseen by the robot) | robot's belief p {p:.2f} "
                        f"| intent proximity {intent.proximity:.2f}"]
        self.meta = {"system": system, "history": history, "index": index, "k": k, "tag": tag, "type": etype,
                     "p_before": round(p, 4), "proximity": round(intent.proximity, 3), "tempo": round(intent.tempo, 3),
                     "hesitation": round(intent.hesitation, 3)}
        self.ff = None
        self.step = 0
        self.frames = 0
        self.arm_until = -1.0
        self.contacts = 0

    def kwargs(self) -> dict:
        return {"scene_extender": _extender, "frame_hook": self.hook}

    def _open(self, ctrl) -> None:
        import mujoco

        from apm_transfer.nerva_world import TARGET

        m, d = ctrl.sim.model, ctrl.sim.data
        self.ox, self.oy = ctrl._origin
        d.mocap_pos[m.body_mocapid[m.body("entity").id]] = [self.ox + TARGET[0], self.oy + TARGET[1], 0.0]
        self.arm_mocap = m.body_mocapid[m.body("arm").id]
        self.arm_geom = m.geom("arm_geom").id
        self.renderer = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [self.ox + 0.45, self.oy, 0.15]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.7, 90.0, -18.0
        self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                    f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-crf", "20", str(OUT / f"{self.name}.mp4")], stdin=subprocess.PIPE)

    def hook(self, ctrl, info: dict) -> None:
        from PIL import Image, ImageDraw

        from apm_transfer.nerva_world import TARGET

        if self.ff is None:
            self._open(ctrl)
        m, d = ctrl.sim.model, ctrl.sim.data
        if info["lunge"] is not None:
            reach, push = info["lunge"]
            x, y, _ = ctrl.pose()
            ang = math.atan2(y - TARGET[1], x - TARGET[0])
            half = reach / 2
            d.mocap_pos[self.arm_mocap] = [self.ox + TARGET[0] + half * math.cos(ang),
                                           self.oy + TARGET[1] + half * math.sin(ang), 0.3]
            # capsule axis is z: q = yaw(ang) ⊗ pitch(90°) lays it horizontally along the direction ang
            c, sn, r = math.cos(ang / 2), math.sin(ang / 2), math.sqrt(0.5)
            d.mocap_quat[self.arm_mocap] = [c * r, -sn * r, c * r, sn * r]
            m.geom_size[self.arm_geom][1] = half
            m.geom_rgba[self.arm_geom] = [0.85, 0.15, 0.1, 1] if push > 0 else [0.6, 0.6, 0.6, 1]
            self.arm_until = ctrl.t + ARM_S
            self.contacts += int(push > 0)
        elif ctrl.t > self.arm_until:
            d.mocap_pos[self.arm_mocap] = [self.ox + 5.0, self.oy + 5.0, 0.0]
        self.step += 1
        if self.step % 2:
            return
        self.renderer.update_scene(d, self.cam)
        img = Image.fromarray(self.renderer.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 48], fill=(255, 255, 255))
        dr.text((6, 2), self.caption[0], fill=(0, 0, 0))
        dr.text((6, 17), self.caption[1], fill=(0, 0, 0))
        status = "approaching" if not info["arrived"] else "standing (sensing)"
        dr.text((6, 32), f"t {ctrl.t:5.1f} s | distance to entity {info['dist']:.2f} m | {status} | "
                         f"contacts {self.contacts}", fill=(0, 0, 0))
        assert self.ff is not None and self.ff.stdin is not None
        self.ff.stdin.write(img.tobytes())
        self.frames += 1

    def close(self, row: dict) -> None:
        if self.ff is not None and self.ff.stdin is not None:
            self.ff.stdin.close()
            self.ff.wait()
            self.renderer.close()
        self.meta.update({"frames": self.frames, "d_stop": row["d_stop"], "contacts": row["contacts"],
                          "max_push": row["max_push"], "t_arrive": row["t_arrive"]})
        (OUT / f"{self.name}.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")


def render(system, history, index, k, ident, etype, rd, intent):
    tag = WANT.get((system, history, index), {}).get(k)
    return Clip(system, history, index, k, tag, etype, rd.p, intent) if tag else None


cl.RENDER = render


def _init(out: str, want: dict) -> None:
    global OUT, WANT
    OUT, WANT = Path(out), want
    cl.RENDER = render


def median_index(lts: dict, system: str) -> int:
    late = lambda r: np.mean([x["d_stop"] for x in r["rows"] if x["entity"] == "B1" and x["k"] >= 10])  # noqa: E731
    diffs = {i: late(lts[(system, "risky_B", i)]) - late(lts[(system, "safe", i)]) for (s, h, i) in lts
             if s == system and h == "risky_B" and (system, "safe", i) in lts}
    med = float(np.median(list(diffs.values())))
    return min(diffs, key=lambda i: (abs(diffs[i] - med), i))


def compose(a: Path, b: Path, out: Path, fa: int, fb: int) -> None:
    n = max(fa, fb)
    pad = lambda f: f"tpad=stop_mode=clone:stop={n - f}"  # noqa: E731
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a), "-i", str(b), "-filter_complex",
                    f"[0:v]{pad(fa)}[l];[1:v]{pad(fb)}[r];[l][r]hstack", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--systems", default="CL,LEARNED_V")
    ap.add_argument("--first", type=int, default=40)
    ap.add_argument("--types-model", default=str(HERE.parent / "configs" / "J5_types_distance.json"))
    ap.add_argument("--edges", default=str(HERE.parent / "configs" / "J4_edges.json"))
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--j6-model", default=str(HERE.parent / "configs" / "J6_model.json"))
    ap.add_argument("--j6-model-v", default=str(HERE.parent / "configs" / "J6_model_v.json"))
    ap.add_argument("--gru-v", default=str(HERE.parent / "configs" / "J6_gru_v.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lts = {}
    for line in a.runs.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        lts[(r["system"], r["history"], r["index"])] = r
    channels = json.loads((HERE.parent / "configs" / "H_admitted_interface.json").read_text())["interface"]
    j6 = json.loads(Path(a.j6_model).read_text())
    j6["v"] = {**json.loads(Path(a.j6_model_v).read_text()), "gru": a.gru_v}
    want: dict = {}
    jobs = []
    chosen = {}
    for s in a.systems.split(","):
        i = median_index(lts, s)
        chosen[s] = i
        for h in ("risky_B", "safe"):
            ks = [x["k"] for x in lts[(s, h, i)]["rows"] if x["entity"] == "B1"]
            want[(s, h, i)] = {ks[0]: "first", ks[-1]: "last"}
            jobs.append(("X", i, h, s, a.types_model, a.edges, a.lam, channels, j6))
    with Pool(len(jobs), initializer=_init, initargs=(str(a.out), want)) as pool:
        results = pool.map(cl.job, jobs)
    manifest: dict = {"chosen_index": chosen, "rule": "median late-B1 stopping-distance difference", "lifetimes": []}
    for r in results:
        logged = lts[(r["system"], r["history"], r["index"])]["rows"]
        keys = ("entity", "type", "p_before", "d_stop", "category", "contacts", "max_push", "t_arrive")
        same = all(all(x[k] == y[k] for k in keys) for x, y in zip(r["rows"], logged, strict=True))
        manifest["lifetimes"].append({"system": r["system"], "history": r["history"], "index": r["index"],
                                      "matches_J6X_log": same})
    for s, i in chosen.items():
        for tag in ("first", "last"):
            m = {h: json.loads((a.out / f"{s}_{i}_{h}_{tag}.json").read_text()) for h in ("risky_B", "safe")}
            compose(a.out / f"{s}_{i}_risky_B_{tag}.mp4", a.out / f"{s}_{i}_safe_{tag}.mp4",
                    a.out / f"{s}_{i}_{tag}_pair.mp4", m["risky_B"]["frames"], m["safe"]["frames"])
            manifest[f"{s}_{i}_{tag}"] = m
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"chosen": chosen, "match": [x["matches_J6X_log"] for x in manifest["lifetimes"]]}))


if __name__ == "__main__":
    main()
