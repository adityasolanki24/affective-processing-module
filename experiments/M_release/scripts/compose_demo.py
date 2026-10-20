"""Learning-and-recovery demonstration video (labelled): the recorded lifetime with a state panel underneath.

    python experiments/M_release/scripts/compose_demo.py --stem <..._demo> --out D_carter.mp4 [--speed 2]

Input: `<stem>.mp4` and `<stem>_meta.json` from `isaac/expression_v2.py --demo`. The panel shows, per frame:
- the meeting number and how the person behaves in it (this is a labelled demonstration, unlike the unlabeled
  comparisons);
- the robot's learned wariness of that person (a bar), as read before the meeting from its own experience memory;
- the expressive plan derived from it: watching time, guarded posture, back-off distance.

Playback: `--speed` keeps every n-th recorded frame at 25 fps (2 = twice real time) and says so on screen. Frames
are never interpolated or synthesised. Output: H.264, yuv420p, faststart (plays on web and mobile).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(m: dict, w: int, h: int, speed: int, n_pushy: int, n_total: int) -> np.ndarray:
    im = Image.new("RGB", (w, h), (24, 26, 30))
    d = ImageDraw.Draw(im)
    big, small = _font(28), _font(22)
    k, person = int(m.get("k", 0)), str(m.get("person", ""))
    d.text((20, 14), f"Meeting {k + 1} of {n_total}   person: {person}", font=big, fill=(240, 240, 240))
    phase_txt = f"first {n_pushy} meetings pushy, then calm      {speed}x speed" if speed > 1 else \
        f"first {n_pushy} meetings pushy, then calm"
    d.text((20, 52), phase_txt, font=small, fill=(170, 170, 170))
    wv = float(m.get("wariness", 0.0))
    x0, y0, bw, bh = 20, 92, w // 2 - 120, 26
    d.text((x0, y0 - 2), "learned wariness of this person", font=small, fill=(220, 220, 220))
    y1 = y0 + 30
    d.rectangle([x0, y1, x0 + bw, y1 + bh], outline=(120, 120, 120))
    col = (int(80 + 170 * wv), int(200 - 120 * wv), 90)
    d.rectangle([x0, y1, x0 + int(bw * wv), y1 + bh], fill=col)
    d.text((x0 + bw + 10, y1), f"{wv:.2f}", font=small, fill=(240, 240, 240))
    xr = w // 2 + 10
    d.text((xr, y0 - 2), "plan from it", font=small, fill=(220, 220, 220))
    d.text((xr, y1), f"watch {m.get('observe_s', 0):.1f} s   guard {m.get('guard', 0):.2f}   "
                     f"back off {m.get('retreat_m', 0):.2f} m", font=small, fill=(240, 240, 240))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    meta = json.loads(Path(f"{a.stem}_meta.json").read_text())["frames"]
    rec = json.loads(Path(f"{a.stem}.json").read_text()).get("demo", {})
    n_pushy = int(rec.get("pushy_n", 6))
    n_total = n_pushy + int(rec.get("calm_n", 7))
    r = imageio.get_reader(f"{a.stem}.mp4")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(a.out), fps=25, codec="libx264", quality=None, macro_block_size=8,
                           ffmpeg_params=["-crf", "26", "-preset", "medium", "-pix_fmt", "yuv420p",
                                          "-movflags", "+faststart"])
    ph = 172  # video 540 + 172: divisible by 8 (no encoder resize)
    n = 0
    for i, fr in enumerate(r):
        if i % a.speed or i >= len(meta):
            continue
        img = Image.fromarray(np.asarray(fr)[..., :3])
        vh = int(round(img.height * a.width / img.width / 2)) * 2
        top = np.asarray(img.resize((a.width, vh)))
        w.append_data(np.concatenate([top, panel(meta[i], a.width, ph, a.speed, n_pushy, n_total)], axis=0))
        n += 1
    w.close()
    r.close()
    print("wrote", a.out, "frames", n, "speed", a.speed)


if __name__ == "__main__":
    main()
