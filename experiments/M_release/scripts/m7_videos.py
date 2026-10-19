"""M7: build the four release videos from the demonstration lifetimes (partition D, recorded with --video).

    python experiments/M_release/scripts/m7_videos.py --d <dir with D_* fetched outputs> --out <videos dir>

1. `1_showcase.mp4` (3-5 min): title, each robot in the warehouse (DT, change_B: the first encounters with the pushy
   person and the first after the switch), each robot in the held-out hospital, closing card. Encounters are shown
   back to back. Between encounters the robot is reset to its approach point, as in the experiments, and the cuts
   are marked.
2. `2_three_robots.mp4`: NERVA | Booster T1 | Nova Carter side by side, the same history and seed index, aligned by
   encounter, with overlays.
3. `3_lifetime.mp4`: one complete 24-encounter DT lifetime (Nova Carter, warehouse, change_B), sped up 2x, with
   the belief and decision overlay. It shows memory forming and then revising after the switch.
4. `4_<robot>.mp4` per robot (NERVA's includes its depth-eye inset).

All footage is Isaac Sim 5.1 / PhysX; every frame is labelled with the engine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPOSE = [sys.executable, str(HERE / "m7_compose.py")]
LABEL = {"nerva": "NERVA (Open Duck Mini v2)", "humanoid": "Booster T1", "nova_carter": "NVIDIA Nova Carter"}


FAILED: list[str] = []


def run(*args: str) -> None:
    """One composer step; a failure is reported and the remaining videos are still built."""
    r = subprocess.run([*COMPOSE, *args])
    if r.returncode:
        FAILED.append(f"{args[0]} {args[-1]} (exit {r.returncode})")
        print("FAILED:", FAILED[-1], flush=True)


def stem(d: Path, robot: str, scene: str, history: str, index: int = 0) -> str | None:
    hits = sorted(d.rglob(f"{robot}_{scene}_DT_{history}_D{index}_meta.json"))
    return str(hits[0])[: -len("_meta.json")] if hits else None


def cut(stem_: str, encounters: list[int], out: Path, speed: int = 2, eye: bool = False) -> Path | None:
    """A clip of selected encounters of one lifetime (frames whose metadata k is listed)."""
    import imageio.v2 as imageio
    sys.path.insert(0, str(HERE))
    from m7_compose import overlay
    meta = json.loads(Path(f"{stem_}_meta.json").read_text())["frames"]
    rd = imageio.get_reader(f"{stem_}.mp4")
    eyes = None
    if eye and Path(f"{stem_}_eye.npz").exists():
        import numpy as np
        z = np.load(f"{stem_}_eye.npz")
        eyes = (z["t"], z["depth"])
    from m7_compose import eye_at
    wr = imageio.get_writer(out, fps=25, codec="libx264", quality=8, macro_block_size=8)
    n = 0
    for i, fr in enumerate(rd):
        if i >= len(meta) or meta[i].get("k") not in encounters or i % speed:
            continue
        wr.append_data(overlay(fr, meta[i], eye=eye_at(eyes, meta[i].get("t", -1)) if eyes else None))
        n += 1
    wr.close()
    return out if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    o = a.out
    o.mkdir(parents=True, exist_ok=True)
    tmp = o / "_parts"
    tmp.mkdir(exist_ok=True)
    parts: list[str] = []

    def card(name: str, text: str, s: float = 3.5) -> str:
        p = tmp / f"{name}.mp4"
        run("card", "--text", text, "--seconds", str(s), "--out", str(p))
        return str(p)

    # 1. showcase
    parts.append(card("title", "APM v1.0: experience-dependent expressive approach\\nThree robots in Isaac Sim 5.1 "
                               "(PhysX) | simulation only\\nNERVA  |  Booster T1  |  NVIDIA Nova Carter", 5))
    for robot in ("nova_carter", "humanoid", "nerva"):
        s = stem(a.d, robot, "warehouse", "change_B")
        if s is None:
            continue
        parts.append(card(f"t_{robot}", f"{LABEL[robot]}\\nwarehouse: meets a person who pushes (B1), "
                                        "then the same person after they change"))
        c = cut(s, [0, 1, 2, 3, 10, 11, 12], tmp / f"wh_{robot}.mp4", speed=3, eye=robot == "nerva")
        if c:
            parts.append(str(c))
    parts.append(card("t_hosp", "Held-out scene: hospital\\nnever used for fitting or tuning"))
    for robot in ("nova_carter", "humanoid", "nerva"):
        s = stem(a.d, robot, "hospital", "late_B")
        if s:
            c = cut(s, [0, 1, 12, 13], tmp / f"hosp_{robot}.mp4", speed=3)
            if c:
                parts.append(str(c))
    parts.append(card("end", "Simulation evidence only. No hardware, no human-participant data.\\n"
                             "Beliefs are expected-perturbation estimates, not emotions.", 5))
    run("concat", "--parts", *parts, "--out", str(o / "1_showcase.mp4"))
    # 2. three robots side by side
    stems = [stem(a.d, r, "warehouse", "change_B") for r in ("nerva", "humanoid", "nova_carter")]
    if all(stems):
        run("panel", "--stems", *[s for s in stems if s], "--labels", "NERVA", "Booster T1", "Nova Carter",
            "--encounters", "0-12", "--speed", "2", "--out", str(o / "2_three_robots.mp4"))
    # 3. one complete lifetime
    s = stem(a.d, "nova_carter", "warehouse", "change_B")
    if s:
        run("clip", "--stem", s, "--speed", "2", "--out", str(o / "3_lifetime.mp4"))
    # 4. per robot
    for robot in ("nova_carter", "humanoid", "nerva"):
        s = stem(a.d, robot, "warehouse", "late_B") or stem(a.d, robot, "warehouse", "change_B")
        if s:
            run("clip", "--stem", s, "--speed", "2", "--out", str(o / f"4_{robot}.mp4"),
                *(["--eye"] if robot == "nerva" else []))
    print("videos in", o, "| failed:", FAILED or "none")


if __name__ == "__main__":
    main()
