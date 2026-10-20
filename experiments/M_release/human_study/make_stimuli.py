"""Build the blinded clip set for the APM v1.0 perception study (PROTOCOL.md §3). It does not recruit anyone and
creates no ratings.

    python experiments/M_release/human_study/make_stimuli.py --lifetimes <dir> --out <stimuli dir> --key <key.json>

Input: recorded lifetimes (<stem>.mp4 + <stem>_meta.json from m5_lifetime.py --video). Raw footage only: the clips
carry no overlay or label. Output:
- `<out>/<id>.mp4` clips;
- `<out>/stimuli_manifest.json` (IDs and SHA-256 only, shipped to the study platform);
- `--key` (ID -> robot, scene, history, encounter, condition), kept apart from the stimuli, opened only for analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import imageio.v2 as imageio

SWITCH, AFTER = 10, 12


def first_b1_after(meta: list[dict], k_min: int) -> tuple[int, int] | None:
    """Frame range of the first B1 approach at encounter >= k_min: 1 s before approach to 3 s after arrival."""
    ks = sorted({m["k"] for m in meta if m.get("entity") == "B1" and m.get("k", -1) >= k_min})
    if not ks:
        return None
    k = ks[0]
    idx = [i for i, m in enumerate(meta) if m.get("k") == k]
    app = [i for i in idx if meta[i].get("phase") in ("approach", "sense")]
    if not app:
        return None
    sense = [i for i in idx if meta[i].get("phase") == "sense"]
    lo = max(idx[0], app[0] - 25)
    hi = min(idx[-1], (sense[0] if sense else app[-1]) + 75, lo + 30 * 25)
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lifetimes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest, key = [], {}
    for meta_path in sorted(a.lifetimes.glob("*_meta.json")):
        stem = str(meta_path)[: -len("_meta.json")]
        rec = json.loads(meta_path.read_text())
        meta, info = rec["frames"], rec["record"]
        if info["system"] != "DT":
            continue
        cond = {"change_B": "pushed_before", "late_B": "pushed_after", "safe": "calm"}.get(info["history"])
        if cond is None:
            continue
        rng_ = first_b1_after(meta, AFTER)
        if rng_ is None:
            continue
        cid = secrets.token_hex(4)
        rd = imageio.get_reader(f"{stem}.mp4")
        wr = imageio.get_writer(a.out / f"{cid}.mp4", fps=25, codec="libx264", quality=8, macro_block_size=8)
        for i, fr in enumerate(rd):
            if rng_[0] <= i <= rng_[1]:
                wr.append_data(fr)
        wr.close()
        h = hashlib.sha256((a.out / f"{cid}.mp4").read_bytes()).hexdigest()
        manifest.append({"id": cid, "sha256": h})
        key[cid] = {"robot": info["robot"], "scene": info["scene"], "history": info["history"],
                    "index": info["index"], "condition": cond, "frames": list(rng_)}
    (a.out / "stimuli_manifest.json").write_text(json.dumps(sorted(manifest, key=lambda m: m["id"]), indent=1))
    a.key.parent.mkdir(parents=True, exist_ok=True)
    a.key.write_text(json.dumps(key, indent=1))
    print(f"{len(manifest)} clips; manifest {a.out / 'stimuli_manifest.json'}; key (keep apart) {a.key}")


if __name__ == "__main__":
    main()
