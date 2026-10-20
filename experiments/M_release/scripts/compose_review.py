"""Side-by-side raw review footage. No history labels, overlays, retiming or looping."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def select(source: Path, meta: Path, dest: Path, after: int, phase: str):
    frames = json.loads(meta.read_text())["frames"]
    keep = [i for i, m in enumerate(frames) if m.get("k", -1) >= after and
            (phase == "all" or m.get("phase") == phase)]
    if not keep:
        raise ValueError(f"no requested frames in {source}")
    groups = []
    start = end = keep[0]
    for index in keep[1:]:
        if index == end + 1:
            end = index
        else:
            groups.append((start, end))
            start = end = index
    groups.append((start, end))
    expr = "+".join(f"between(n\\,{start}\\,{end})" for start, end in groups)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-vf", f"select={expr},setpts=N/(25*TB)", "-an", "-c:v", "libx264",
                    "-crf", "18", "-threads", "2", str(dest)], check=True)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--left-meta", type=Path)
    ap.add_argument("--right-meta", type=Path)
    ap.add_argument("--after", type=int, default=6)
    ap.add_argument("--phase", default="all", choices=("all", "sense", "approach"))
    ap.add_argument("--header-text", type=Path, help="optional gait-only header; omit for blinded expression")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    key = {"left": str(a.left), "right": str(a.right), "playback_speed": 1,
           "phase": a.phase, "cuts": "selected recorded encounters; no frame synthesis"}
    with tempfile.TemporaryDirectory() as tmp:
        inputs = [a.left, a.right]
        for side, meta in enumerate((a.left_meta, a.right_meta)):
            if meta:
                dest = Path(tmp) / f"{side}.mp4"
                key[f"frame_indices_{side}"] = select(inputs[side], meta, dest, a.after, a.phase)
                inputs[side] = dest
        # Compare the common recorded duration; the assignment key preserves every source.
        key["trimmed_to_common_duration"] = True
        extra, end = [], "[l][r]hstack=shortest=1[v]"
        if a.header_text:
            from PIL import Image, ImageDraw, ImageFont
            header = Image.new("RGB", (1920, 90), (15, 18, 22))
            draw = ImageDraw.Draw(header)
            try:
                font = ImageFont.truetype("arial.ttf", 26)
            except OSError:
                font = ImageFont.truetype("DejaVuSans.ttf", 26)
            draw.multiline_text((20, 10), a.header_text.read_text(), font=font, fill="white", spacing=8)
            header.save(Path(tmp) / "header.png")
            extra = ["-loop", "1", "-framerate", "25", "-i", str(Path(tmp) / "header.png")]
            end = "[l][r]hstack=shortest=1[p];[2:v][p]vstack=shortest=1[v]"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(inputs[0]), "-i", str(inputs[1]), *extra,
                        "-filter_complex", "[0:v]scale=960:540,setsar=1[l];"
                        "[1:v]scale=960:540,setsar=1[r];" + end,
                        "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-threads", "2",
                        str(a.out)], check=True)
    key["sha256"] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    a.out.with_suffix(".key.json").write_text(json.dumps(key, indent=2))


if __name__ == "__main__":
    main()
