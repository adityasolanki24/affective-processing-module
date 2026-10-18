"""K6: train T1JoystickFlatTerrain with mujoco_playground's published PPO configuration (GPU). Saves params with numpy.

This is the humanoid's *motor controller*, trained once and frozen. It is not part of APM: the APM core only issues
abstract intents through an embodiment adapter, and this controller is the humanoid's equivalent of NERVA's frozen
neutral policy. Its locomotion envelope must be validated separately before any APM experiment uses it.
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mujoco
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="T1JoystickFlatTerrain")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--steps", type=int, default=0, help="override num_timesteps (0: the published value)")
a = ap.parse_args()
env = registry.load(a.env)
env_cfg = registry.get_default_config(a.env)
params = locomotion_params.brax_ppo_config(a.env).to_dict()
if a.steps:
    params["num_timesteps"] = a.steps
net = params.pop("network_factory", None)
factory = functools.partial(ppo_networks.make_ppo_networks, **net) if net else ppo_networks.make_ppo_networks
t0 = time.time()
log = []


def progress(step, metrics):
    r = float(metrics.get("eval/episode_reward", np.nan))
    log.append((int(step), r, round(time.time() - t0, 1)))
    print(f"step {step} reward {r:.2f} t {time.time() - t0:.0f}s", flush=True)


train_fn = functools.partial(ppo.train, **params, network_factory=factory,
                             randomization_fn=registry.get_domain_randomizer(a.env), progress_fn=progress)
make_inference_fn, prm, _ = train_fn(environment=env, eval_env=registry.load(a.env, config=env_cfg),
                                     wrap_env_fn=wrapper.wrap_for_brax_training)
a.out.mkdir(parents=True, exist_ok=True)
mujoco.mj_saveModel(env.mj_model, str(a.out / "t1_model.mjb"), None)  # the exact trained-on robot and scene
with open(a.out / "t1_params.pkl", "wb") as f:
    pickle.dump(jax.tree_util.tree_map(np.asarray, prm), f)
import importlib.metadata as md  # noqa: E402

versions = {p: md.version(p) for p in ("playground", "brax", "jax", "mujoco", "mujoco-mjx", "flax")}
(a.out / "t1_meta.json").write_text(json.dumps({"env": a.env, "versions": versions, "env_config": env_cfg.to_dict(),
                                                "ppo": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                                        for k, v in params.items()},
                                                "network": net, "log": log,
                                                "wall_s": round(time.time() - t0, 1)}, default=str), encoding="utf-8")
print("saved", a.out, "wall", round(time.time() - t0, 1))
