"""Generate the clip playlists ("forms") for the H2 survey page (survey/index.html).

python scripts/h2_survey_forms.py

Forms list file names only: the style key never reaches the page. Pilot forms draw from seed partition H2/P.
Per prereg/H2_PROTOCOL.md §4. Implementation choices the protocol leaves open (recorded in survey/README.md):
- training uses the two neutral clips with seed index 1, and the main block uses the two with seed index 0, so no
  clip is shown twice;
- full form (36): each designed style 2 native + 2 matched (one per geometry each), 2 neutral, 10 random styles;
- quick form (12, pilot only): each designed style once, 2 neutral, 4 random styles;
- attention checks use random-style clips that are not otherwise in that form (2 in full, 1 in quick);
- no two consecutive main clips share a category (a designed style, neutral, or random).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from seeds import world_seed  # noqa: E402

N_FORMS = 20
DESIGNED = ("cautious", "assertive", "curious", "hesitant", "relaxed", "energetic")


def _order(items: list[dict], rng: np.random.Generator) -> list[dict]:
    for _ in range(10_000):
        perm = [items[i] for i in rng.permutation(len(items))]
        if all(perm[i]["cat"] != perm[i + 1]["cat"] for i in range(len(perm) - 1)):
            return perm
    raise RuntimeError("no valid order")


def _pick(clips: list[dict], rng: np.random.Generator, **match: object) -> dict:
    pool = [c for c in clips if all(c[k] == v for k, v in match.items())]
    return pool[int(rng.integers(len(pool)))]


def make_form(clips: list[dict], kind: str, rng: np.random.Generator) -> list[dict]:
    geoms = sorted({c["geometry"] for c in clips if c["style"] in DESIGNED})
    main: list[dict] = []
    for s in DESIGNED:
        if kind == "full":
            for speed in ("native", "matched"):
                for g in geoms:
                    main.append(_pick(clips, rng, style=s, speed=speed, geometry=g))
        else:
            main.append(_pick(clips, rng, style=s, speed=str(rng.choice(["native", "matched"])),
                              geometry=geoms[int(rng.integers(len(geoms)))]))
    main += [c for c in clips if c["style"] == "neutral" and c["seed_index"] == 0]
    rand = [c for c in clips if c["style"] == "random"]
    rand = [rand[i] for i in rng.permutation(len(rand))]
    n_rand, n_att = (10, 2) if kind == "full" else (4, 1)
    main += rand[:n_rand]
    items = _order([{"file": c["file"], "cat": c["style"]} for c in main], rng)
    att = rand[n_rand:n_rand + n_att]
    for j, c in enumerate(att):  # spread attention checks evenly through the block
        pos = round((j + 1) * len(items) / (n_att + 1)) + j
        items.insert(pos, {"file": c["file"], "cat": "attention"})
    return [{"file": it["file"], "attention": it["cat"] == "attention"} for it in items]


def main() -> None:
    clips = json.loads((HERE / "videos" / "h1" / "MANIFEST_KEY_DO_NOT_SHOW.json").read_text(encoding="utf-8"))
    training = [c["file"] for c in sorted(clips, key=lambda c: c["geometry"])
                if c["style"] == "neutral" and c["seed_index"] == 1]
    forms: dict[str, list] = {"full": [], "quick": []}
    for kind, offset in (("full", 0), ("quick", 1000)):
        for i in range(N_FORMS):
            rng = np.random.default_rng(world_seed("H2", "P", offset + i))
            forms[kind].append(make_form(clips, kind, rng))
    out = {
        "instrument": "H2 survey v0.1 (pilot)",
        "seed_partition": "H2/P",
        "video_dir": "../videos/h1/",
        "training": training,
        "forms": forms,
    }
    path = HERE / "survey" / "forms.js"
    path.parent.mkdir(exist_ok=True)
    path.write_text("window.H2_FORMS = " + json.dumps(out, indent=1) + ";\n", encoding="utf-8")
    print("wrote", path, {k: (len(v), len(v[0])) for k, v in forms.items()})


if __name__ == "__main__":
    main()
