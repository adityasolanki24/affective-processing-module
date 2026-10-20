"""L audit A7 (part 1): the humanoid's NumPy policy is the trained Brax network, and its assets are reproducible.

Two steps, because the trained model needs MuJoCo 3.16 (`hum` venv) and the reference network needs brax/jax:

    # 1. hum venv: observations from real rollouts (commands, stop/start, turns, pushes), NumPy actions
    python l_a7_policy_equiv.py collect --out results/A7_obs.npz
    # 2. brax venv: the training pickle through brax's own inference function, compared with step 1
    python l_a7_policy_equiv.py compare --obs results/A7_obs.npz --params <t1_params.pkl>
        --out results/A7_policy_equiv.json

`compare` also rebuilds `t1_policy.npz` from the pickle and checks it array by array, and records the SHA-256 of the
model and weights against the stored hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
ASSETS = REPO / "experiments" / "K_recovery" / "humanoid" / "trained"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(out: Path) -> None:
    from apm_transfer.humanoid_world import HumanoidRobot, _act
    obs, acts = [], []
    script = [(0.0, 0.0, 1.0), (0.3, 0.0, 3.0), (0.0, 0.0, 1.0), (0.2, 0.8, 2.0), (0.5, -0.5, 2.0), (0.0, 0.0, 1.0)]
    for seed in range(4):
        r = HumanoidRobot(seed)
        rng = np.random.default_rng(seed)
        for v, w, dur in script:
            for k in range(int(dur / r.dt)):
                r.apply(v, w)
                if k % 40 == 20:
                    r.push(*rng.uniform(-0.3, 0.3, 2))
                o = r._obs(np.array([r.v_cmd, 0.0, r.w_cmd]))
                obs.append(o)
                acts.append(_act(o, r.mean, r.std, r.weights))
                r.advance()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, obs=np.array(obs), act=np.array(acts))
    print("collected", len(obs))


def compare(obs_path: Path, params: Path, out: Path) -> None:
    import pickle

    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    meta = json.loads((ASSETS / "t1_meta.json").read_text())
    prm = pickle.load(open(params, "rb"))
    z = np.load(obs_path)
    n_state, n_priv = prm[0].mean["state"].shape[0], prm[0].mean["privileged_state"].shape[0]
    nets = ppo_networks.make_ppo_networks({"state": (n_state,), "privileged_state": (n_priv,)}, z["act"].shape[1],
                                          preprocess_observations_fn=running_statistics.normalize, **meta["network"])
    infer = jax.jit(ppo_networks.make_inference_fn(nets)(prm[:2], deterministic=True))
    o = z["obs"]
    ref = np.asarray(infer({"state": o, "privileged_state": np.zeros((len(o), n_priv), np.float32)},
                           jax.random.PRNGKey(0))[0])
    err = np.abs(ref - z["act"])
    # the npz must be exactly the pickle's policy arrays
    w = np.load(ASSETS / "t1_policy.npz")
    pol = prm[1]["params"]
    arrays = {"mean": np.asarray(prm[0].mean["state"]), "std": np.asarray(prm[0].std["state"])}
    for name, layer in pol.items():
        arrays[f"{name}_kernel"], arrays[f"{name}_bias"] = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
    npz_equal = {k: bool(np.array_equal(w[k], v)) for k, v in arrays.items()}
    npz_equal["no_extra"] = set(w.files) == set(arrays)
    stored = (ASSETS / "t1_model.mjb.sha256").read_text().split()[0]
    res = {"n_obs": int(len(o)), "action_dim": int(z["act"].shape[1]),
           "max_abs_action_diff": float(err.max()), "mean_abs_action_diff": float(err.mean()),
           "equivalent_float32": bool(err.max() < 1e-4), "npz_matches_pickle": npz_equal,
           "sha256": {"t1_model.mjb": sha(ASSETS / "t1_model.mjb"), "t1_policy.npz": sha(ASSETS / "t1_policy.npz"),
                      "t1_params.pkl": sha(params)},
           "mjb_matches_stored_hash": sha(ASSETS / "t1_model.mjb") == stored,
           "training_versions": meta["versions"]}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "sha256"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    m = sp.add_parser("compare")
    m.add_argument("--obs", type=Path, required=True)
    m.add_argument("--params", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    collect(a.out) if a.cmd == "collect" else compare(a.obs, a.params, a.out)


if __name__ == "__main__":
    main()
