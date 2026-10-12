"""H1 analysis (prereg/H1_ADDENDUM.md). Numpy only.

    python scripts/h1_analyze.py calibrate --dir results/H1/V      -> tempos.json (speed matching, V partition)
    python scripts/h1_analyze.py analyze   --dir results/H1/X      -> h1_results.json

Classifier: shrinkage LDA (shrinkage 0.1 toward scaled identity) on standardised features. Cross-validation is
leave-one-geometry-out, so a style is recognised in a task geometry the classifier never saw.

Feature sets:
- "all";
- "non_speed": drops the speed group and the cost group (cost tracks the amount of walking);
- one set per group: temporal, posture, acceleration, spatial.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from apm_express.features import GROUPS

SHRINK = 0.1
NON_SPEED = tuple(f for g, fs in GROUPS.items() if g not in ("speed", "cost") for f in fs)
NON_SPEED_NON_POSTURE = tuple(f for g, fs in GROUPS.items() if g not in ("speed", "cost", "posture") for f in fs)
FEATURE_SETS = {
    "all": tuple(f for fs in GROUPS.values() for f in fs),
    "non_speed": NON_SPEED,
    "non_speed_non_posture": NON_SPEED_NON_POSTURE,
    **{g: GROUPS[g] for g in ("temporal", "posture", "acceleration", "spatial")},
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _matrix(rows, features):
    return np.array([[r["features"][f] for f in features] for r in rows], dtype=float)


class LDA:
    def fit(self, X, y):
        self.mu_x, self.sd_x = X.mean(0), X.std(0)
        self.keep = self.sd_x > 1e-9
        Z = (X[:, self.keep] - self.mu_x[self.keep]) / self.sd_x[self.keep]
        self.classes = sorted(set(y))
        y = np.asarray(y)
        self.means = np.array([Z[y == c].mean(0) for c in self.classes])
        resid = np.concatenate([Z[y == c] - Z[y == c].mean(0) for c in self.classes])
        cov = resid.T @ resid / max(len(Z) - len(self.classes), 1)
        p = cov.shape[0]
        cov = (1 - SHRINK) * cov + SHRINK * np.trace(cov) / p * np.eye(p)
        self.W = np.linalg.inv(cov)
        self.prior = np.log(np.array([np.mean(y == c) for c in self.classes]))
        return self

    def predict(self, X):
        Z = (X[:, self.keep] - self.mu_x[self.keep]) / self.sd_x[self.keep]
        scores = Z @ self.W @ self.means.T - 0.5 * np.sum(self.means @ self.W * self.means, axis=1) + self.prior
        return [self.classes[i] for i in np.argmax(scores, axis=1)]


def classify_logo(rows, features, label=lambda r: r["trial"]["style"]) -> dict:
    """Leave-one-geometry-out predictions -> balanced accuracy, recall per class, confusion."""
    X = _matrix(rows, features)
    y = [label(r) for r in rows]
    g = np.array([r["trial"]["geometry"] for r in rows])
    pred = [None] * len(rows)
    for geo in sorted(set(g)):
        test = g == geo
        model = LDA().fit(X[~test], [yy for yy, t in zip(y, test, strict=True) if not t])
        for i, p in zip(np.nonzero(test)[0], model.predict(X[test]), strict=True):
            pred[i] = p
    return _scores(y, pred)


def _scores(y, pred) -> dict:
    classes = sorted(set(y))
    recall = {c: float(np.mean([p == c for yy, p in zip(y, pred, strict=True) if yy == c])) for c in classes}
    conf = {c: dict(collections.Counter(p for yy, p in zip(y, pred, strict=True) if yy == c)) for c in classes}
    return {
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "chance": 1.0 / len(classes),
        "recall": recall,
        "confusion": conf,
        "n": len(y),
    }


def transfer(train_rows, test_rows, features) -> dict:
    model = LDA().fit(_matrix(train_rows, features), [r["trial"]["style"] for r in train_rows])
    y = [r["trial"]["style"] for r in test_rows]
    return _scores(y, model.predict(_matrix(test_rows, features)))


def safety_table(rows, key=lambda r: r["trial"]["style"]) -> dict:
    per = collections.defaultdict(list)
    for r in rows:
        per[key(r)].append(r)
    return {
        k: {
            "n": len(v),
            "falls": sum(r["fell"] for r in v),
            "trips": sum(r["trips"] > 0 for r in v),
            "arrival": float(np.mean([r["arrived"] for r in v])),
            "progress_speed": float(np.mean([r["features"]["progress_speed"] for r in v])),
            "sparc": float(np.mean([r["features"]["sparc"] for r in v])),
            "power": float(np.mean([r["features"]["mean_power"] for r in v])),
            "head_while_moving_max": float(max(r["head_while_moving_max"] for r in v)),
        }
        for k, v in sorted(per.items())
    }


def diversity(rows, features) -> float:
    """Mean pairwise Euclidean distance between trials in standardised feature space."""
    X = _matrix(rows, features)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(len(Z), 1)].mean())


def pca_dims(rows, features, var=0.9) -> dict:
    X = _matrix(rows, features)
    keep = X.std(0) > 1e-9
    Z = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    frac = np.cumsum(ev) / ev.sum()
    return {"components_for_90pct": int(np.searchsorted(frac, var) + 1), "explained": frac[:8].round(3).tolist()}


def _arm_summary(rows) -> dict:
    return {
        "n": len(rows),
        "falls": sum(x["fell"] for x in rows),
        "arrival": float(np.mean([x["arrived"] for x in rows])),
        "sparc": float(np.mean([x["features"]["sparc"] for x in rows])),
        "power": float(np.mean([x["features"]["mean_power"] for x in rows])),
        "diversity_non_speed": diversity(rows, NON_SPEED),
    }


def h1_gate(out: dict) -> dict:
    """prereg/H1_ADDENDUM.md gate G1-G5."""
    designed = ("cautious", "assertive", "curious", "hesitant", "relaxed", "energetic")
    arms = [out.get(k, {}) for k in ("core_safety", "matched_safety", "switch_safety")]
    g1 = all(v["falls"] == 0 and v["trips"] == 0 for a in arms for v in a.values()) and all(
        v["falls"] == 0 for v in out.get("push_safety", {}).values()
    )
    g2 = all(v["arrival"] >= 0.95 for k in ("core_safety", "matched_safety") for v in out.get(k, {}).values())
    core = out["core_classification"]["non_speed"]
    g3 = core["balanced_accuracy"] >= 0.80 and sum(core["recall"].get(s, 0) >= 0.80 for s in designed) >= 4
    m = out.get("matched_classification", {}).get("non_speed")
    spread_ok = all(abs(v) <= 0.15 for v in out.get("matched_speed_spread", {}).values())
    g4 = (
        bool(m)
        and spread_ok
        and m["balanced_accuracy"] >= 0.70
        and sum(m["recall"].get(s, 0) >= 0.70 for s in designed) >= 4
    )
    g5 = out.get("push_transfer_non_speed", {}).get("balanced_accuracy", 0.0) >= 0.70
    repertoire = [s for s in designed if core["recall"].get(s, 0) >= 0.80 and m and m["recall"].get(s, 0) >= 0.70]
    return {
        "G1_safety": g1,
        "G2_task": g2,
        "G3_distinct": g3,
        "G4_matched": g4,
        "matched_speed_ok": spread_ok,
        "G5_perturbation": g5,
        "passes": bool(g1 and g2 and g3 and g4 and g5),
        "repertoire": repertoire,
    }


def calibrate(d: Path) -> dict:
    rows = _load(d / "calibrate_primary.json")["rows"]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        style, tempo = r["trial"]["style"].split("@")
        per[style][float(tempo)].append(r["features"]["progress_speed"])
    curves = {s: {t: float(np.mean(v)) for t, v in sorted(c.items())} for s, c in per.items()}
    target = 0.9 * min(c[max(c)] for c in curves.values())  # 90% of the slowest style's full-tempo speed
    tempos = {}
    for s, c in curves.items():
        ts, vs = np.array(list(c)), np.array(list(c.values()))
        order = np.argsort(vs)
        tempos[s] = float(np.clip(np.interp(target, vs[order], ts[order]), 0.25, 1.0))
    return {"target_progress_speed": target, "curves": curves, "tempos": tempos}


def analyze(d: Path) -> dict:
    core = _load(d / "core_primary.json")["rows"]
    out: dict = {"core_safety": safety_table(core)}
    out["core_classification"] = {k: classify_logo(core, f) for k, f in FEATURE_SETS.items()}
    matched = _load(d / "matched_primary.json")
    if matched:
        m = matched["rows"]
        out["matched_safety"] = safety_table(m)
        out["matched_classification"] = {k: classify_logo(m, f) for k, f in FEATURE_SETS.items()}
        speeds = {k: v["progress_speed"] for k, v in out["matched_safety"].items()}
        mean = float(np.mean(list(speeds.values())))
        out["matched_speed_spread"] = {k: v / mean - 1 for k, v in speeds.items()}
    push = _load(d / "push_primary.json")
    if push:
        out["push_safety"] = safety_table(push["rows"])
        out["push_transfer_non_speed"] = transfer(core, push["rows"], NON_SPEED)
    switch = _load(d / "switch_primary.json")
    if switch:
        out["switch_safety"] = safety_table(switch["rows"])
    shift = _load(d.parent / "S" / "core_primary.json")
    if shift:
        out["shift_safety"] = safety_table(shift["rows"])
        out["shift_transfer_non_speed"] = transfer(core, shift["rows"], NON_SPEED)
    sec = _load(d / "core_secondary.json")
    if sec:
        out["secondary_safety"] = safety_table(sec["rows"])
        out["secondary_classification_non_speed"] = classify_logo(sec["rows"], NON_SPEED)
    out["gate"] = h1_gate(out)
    rnd = _load(d / "random_primary.json")
    if rnd:
        r = rnd["rows"]
        designed = [x for x in core if x["trial"]["style"] != "neutral"]
        out["x1_vs_x2"] = {"x1": _arm_summary(designed), "x2": _arm_summary(r), "x3_pca_on_x2": pca_dims(r, NON_SPEED)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "analyze"])
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    if a.mode == "calibrate":
        res, path = calibrate(a.dir), a.dir / "tempos.json"
    else:
        res, path = analyze(a.dir), a.dir / "h1_results.json"
    path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
