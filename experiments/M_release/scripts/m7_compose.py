"""M7: compose the release videos from recorded Isaac Sim footage. Overlays only; no frame is synthesised or altered
beyond labels, and no motion is added.

    python experiments/M_release/scripts/m7_compose.py clip  --stem <dir/name> --out clip.mp4 [--eye] [--speed 2]
    python experiments/M_release/scripts/m7_compose.py panel --stems A B C --labels NERVA "Booster T1" "Nova Carter"
        --out panel.mp4 [--encounters 0-5]
    python experiments/M_release/scripts/m7_compose.py card  --text "..." --seconds 3 --out card.mp4
    python experiments/M_release/scripts/m7_compose.py concat --parts a.mp4 b.mp4 ... --out showcase.mp4

A stem is a recorded lifetime: <stem>.mp4 (25 fps side view), <stem>_meta.json (per-frame: time, phase and the
runtime's decision), optionally <stem>_eye.npz (the robot's own depth eye, 10 Hz during sensing).

Overlay panel, per frame:
- engine, robot, scene, system, history;
- encounter number and phase;
- the person's true type, labelled as evaluation truth the robot never receives;
- the runtime's belief readout (expected perturbation) and uncertainty, as bars;
- the decision (approach intent or OBSERVE/HOLD, sensing windows, supervisor veto);
- the previous encounter's observation (IMU categories, camera codes, detector, lunge contacts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 25
W, H = 1280, 720
NAMES = {"nova_carter": "NVIDIA Nova Carter", "humanoid": "Booster T1", "nerva": "NERVA (Open Duck Mini v2)"}
SCENES = {"warehouse": "warehouse", "home": "home (residential room)", "hospital": "hospital (held-out scene)"}


def font(size: int):
    for f in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_S, F_M, F_L = font(19), font(22), font(36)


def action_text(m: dict) -> str:
    a = m.get("action")
    if a is None:
        return ""
    if a == -1:
        return "OBSERVE from 0.3 m beyond reach"
    if a == -2:
        return "HOLD (no approach within the risk cap)"
    return f"approach  proximity {m['proximity']:.2f}  tempo {m['tempo']:.2f}  hesitation {m['hesitation']:.2f}"


def bar(d: ImageDraw.ImageDraw, x: int, y: int, w: int, v: float, label: str, col) -> None:
    d.rectangle([x, y, x + w, y + 14], outline=(220, 220, 220), width=1)
    d.rectangle([x + 1, y + 1, x + 1 + int((w - 2) * float(np.clip(v, 0, 1))), y + 13], fill=col)
    d.text((x + w + 8, y - 2), f"{label} {v:.2f}", font=F_S, fill=(240, 240, 240))


def overlay(frame: np.ndarray, m: dict, title: str | None = None, eye: np.ndarray | None = None) -> np.ndarray:
    im = Image.fromarray(frame).convert("RGBA")
    if im.size != (W, H):
        im = im.resize((W, H))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0, 150))
    head = title or (f"Isaac Sim 5.1 (PhysX)  |  {NAMES.get(m.get('robot', ''), m.get('robot', ''))}  |  "
                     f"{SCENES.get(m.get('scene', ''), m.get('scene', ''))}")
    d.text((12, 7), head, font=F_M, fill=(255, 255, 255))
    if m.get("k") is not None:
        x0, y0 = 12, H - 178
        d.rectangle([0, y0 - 10, 800, H], fill=(0, 0, 0, 150))
        d.text((x0, y0), f"encounter {m['k'] + 1}   {m.get('phase', '')}   APM system: {m.get('system', '')}"
                         f"   history: {m.get('history', '')}", font=F_S, fill=(255, 255, 255))
        d.text((x0, y0 + 24), f"person {m.get('entity', '')}   true type: {m.get('true_type', '')} "
                              f"(evaluation truth, not given to the robot)", font=F_S, fill=(255, 210, 120))
        bar(d, x0, y0 + 52, 200, m.get("p_risk", 0.0), "expected perturbation (belief)", (230, 90, 70, 255))
        bar(d, x0, y0 + 76, 200, m.get("uncertainty", 0.0), "uncertainty", (110, 160, 240, 255))
        veto = "   VETOED by the risk supervisor" if m.get("vetoed") else ""
        d.text((x0, y0 + 100), f"decision: {action_text(m)}   windows {m.get('windows', '')}{veto}", font=F_S,
               fill=(200, 255, 200))
        last = m.get("last")
        if last:
            d.text((x0, y0 + 126), f"last observation ({last[0]}): IMU {last[1]}  camera {last[2]}  "
                                   f"detector {'yes' if last[3] else 'no'}  lunge contacts {last[4]}", font=F_S,
                   fill=(220, 220, 220))
    out = Image.alpha_composite(im, layer).convert("RGB")
    if eye is not None:
        e = np.clip(eye.astype(float) / 2.0, 0, 1)
        e = (255 * (1 - e)) * (eye > 0)
        ei = Image.fromarray(e.astype(np.uint8)).resize((256, 192), Image.NEAREST).convert("RGB")
        dd = ImageDraw.Draw(ei)
        dd.text((4, 2), "robot's depth eye (64x48)", font=F_S, fill=(255, 255, 0))
        out.paste(ei, (W - 266, 44))
    return np.asarray(out)


def load_stem(stem: str):
    meta = json.loads(Path(f"{stem}_meta.json").read_text())
    eye = None
    if Path(f"{stem}_eye.npz").exists():
        z = np.load(f"{stem}_eye.npz")
        eye = (z["t"], z["depth"])
    return imageio.get_reader(f"{stem}.mp4"), meta["frames"], eye


def eye_at(eye, t: float):
    if eye is None:
        return None
    ts, depth = eye
    i = int(np.searchsorted(ts, t, side="right")) - 1
    return depth[i] if i >= 0 and t - ts[i] < 0.3 else None


def cmd_clip(a) -> None:
    rd, meta, eye = load_stem(a.stem)
    wr = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for i, fr in enumerate(rd):
        if i % a.speed:
            continue
        m = meta[i] if i < len(meta) else {}
        wr.append_data(overlay(fr, m, eye=eye_at(eye, m.get("t", -1)) if a.eye else None))
    wr.close()


def cmd_panel(a) -> None:
    """Three lifetimes side by side, aligned by encounter: each panel shows its own robot's encounter k; a robot that
    finishes earlier holds its last frame (marked) until all have finished."""
    lo, hi = (int(x) for x in a.encounters.split("-"))
    streams = []
    for stem in a.stems:
        rd, meta, _ = load_stem(stem)
        frames = [f for f in rd]
        streams.append((frames, meta))
    wr = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for k in range(lo, hi + 1):
        per = []
        for frames, meta in streams:
            idx = [i for i, m in enumerate(meta) if m.get("k") == k and i < len(frames)]
            per.append(idx[:: a.speed])
        n = max((len(p) for p in per), default=0)
        for j in range(n):
            tiles = []
            for (frames, meta), idx, lab in zip(streams, per, a.labels, strict=True):
                if not idx:
                    tiles.append(np.zeros((360, 640, 3), np.uint8))
                    continue
                i = idx[min(j, len(idx) - 1)]
                t = overlay(frames[i], meta[i], title=f"Isaac Sim 5.1  |  {lab}")
                t = np.asarray(Image.fromarray(t).resize((640, 360)))
                if j >= len(idx):
                    t = (t * 0.6).astype(np.uint8)
                tiles.append(t)
            wr.append_data(np.concatenate(tiles, axis=1))
    wr.close()


def cmd_card(a) -> None:
    im = Image.new("RGB", (W, H), (12, 14, 18))
    d = ImageDraw.Draw(im)
    lines = a.text.split("\\n")
    y = H // 2 - 24 * len(lines)
    for i, line in enumerate(lines):
        f = F_L if i == 0 else F_M
        w = d.textlength(line, font=f)
        d.text(((W - w) / 2, y), line, font=f, fill=(255, 255, 255) if i == 0 else (200, 200, 200))
        y += 52 if i == 0 else 32
    wr = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    arr = np.asarray(im)
    for _ in range(int(a.seconds * FPS)):
        wr.append_data(arr)
    wr.close()


def cmd_concat(a) -> None:
    wr = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for p in a.parts:
        for fr in imageio.get_reader(p):
            if fr.shape[:2] != (H, W) and fr.shape[1] != 1920:
                fr = np.asarray(Image.fromarray(fr).resize((W, H)))
            wr.append_data(fr)
    wr.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("clip")
    p.add_argument("--stem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--eye", action="store_true")
    p.add_argument("--speed", type=int, default=1)
    p.set_defaults(func=cmd_clip)
    p = sub.add_parser("panel")
    p.add_argument("--stems", nargs=3, required=True)
    p.add_argument("--labels", nargs=3, required=True)
    p.add_argument("--encounters", default="0-5")
    p.add_argument("--speed", type=int, default=2)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_panel)
    p = sub.add_parser("card")
    p.add_argument("--text", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_card)
    p = sub.add_parser("concat")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_concat)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
