"""Unlabeled side-by-side comparison of recorded encounters (final video B): no history names, no belief bars.

    python experiments/M_release/scripts/compose_unlabeled.py --stems <stem> <stem> <stem> --out B_nerva.mp4 \
        [--encounter 0] [--seed 1]

Each stem is a recorded present-task video (`<stem>.mp4` + `<stem>_meta.json`) from `isaac/expression_v2.py`.
- Panels are shuffled with a fixed seed and marked only A, B, C. The assignment key (which panel is neutral or
  which history) goes to `<out>.key.json`, kept apart from the video.
- Each panel plays the selected recorded encounter from its first frame, at real time: no retiming, no
  interpolation, no frame synthesis. A panel whose encounter ends earlier holds its last frame until all have
  ended (the key records this).
- Frames are streamed, never held in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def encounter_frames(meta: list[dict], k: int) -> tuple[int, int]:
    """[first, last) frame indices of the k-th encounter: encounters start where the phase becomes 'settle'."""
    starts = [i for i, m in enumerate(meta) if m.get("phase") == "settle"
              and (i == 0 or meta[i - 1].get("phase") != "settle")]
    starts.append(len(meta))
    if k + 1 >= len(starts):
        raise ValueError(f"encounter {k} not recorded")
    return starts[k], starts[k + 1]


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encounter", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--panel", default="640x360")
    a = ap.parse_args()
    pw, ph = (int(x) for x in a.panel.split("x"))
    order = list(np.random.default_rng(a.seed).permutation(len(a.stems)))
    stems = [a.stems[i] for i in order]
    letters = "ABCDEF"[:len(stems)]
    readers, spans = [], []
    for s in stems:
        meta = json.loads(Path(f"{s}_meta.json").read_text())["frames"]
        lo, hi = encounter_frames(meta, a.encounter)
        readers.append(imageio.get_reader(f"{s}.mp4"))
        spans.append((lo, hi))
    n = max(hi - lo for lo, hi in spans)
    font = _font(28)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(a.out), fps=25, codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    iters = [enumerate(r) for r in readers]
    last: list[np.ndarray | None] = [None] * len(readers)
    pos = [0] * len(readers)
    # advance each reader to its encounter start
    for j, (lo, _) in enumerate(spans):
        while pos[j] < lo:
            next(iters[j])
            pos[j] += 1
    held = [0] * len(readers)
    for _ in range(n):
        tiles = []
        for j, (_lo, hi) in enumerate(spans):
            if pos[j] < hi:
                _, fr = next(iters[j])
                pos[j] += 1
                last[j] = np.asarray(Image.fromarray(np.asarray(fr)[..., :3]).resize((pw, ph)))
            else:
                held[j] += 1
            im = Image.fromarray(last[j])
            ImageDraw.Draw(im).text((12, 8), letters[j], font=font, fill=(255, 255, 255), stroke_width=2,
                                    stroke_fill=(0, 0, 0))
            tiles.append(np.asarray(im))
        w.append_data(np.concatenate(tiles, axis=1))
    w.close()
    key = {"panels": {letters[j]: stems[j] for j in range(len(stems))}, "encounter": a.encounter,
           "frames_per_panel": [hi - lo for lo, hi in spans], "held_last_frame": dict(zip(letters, held, strict=True)),
           "playback": "real time from each encounter's first frame; no retiming or synthesis",
           "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}
    a.out.with_suffix(".key.json").write_text(json.dumps(key, indent=1))
    print("wrote", a.out, "frames", n, "held", held)


if __name__ == "__main__":
    main()
