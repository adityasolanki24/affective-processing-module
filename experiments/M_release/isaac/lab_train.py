"""APM v1.0: train an Isaac Lab locomotion policy (rsl_rl PPO) for one of APM's bodies, headless.

    $ISAAC_PY experiments/M_release/isaac/lab_train.py --task APM-Velocity-Flat-T1-v0 --num-envs 4096 \
        --iterations 2500 --out out/t1_train

It writes:
- checkpoints in `out/<run>/model_*.pt`;
- `policy.pt` (TorchScript) and `policy.onnx`, the exported actor with its observation normaliser;
- `env_cfg.json` (the exact environment configuration, including joint order and default pose);
- `train_log.json` (mean reward and episode length per iteration).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--num-envs", type=int, default=4096)
ap.add_argument("--iterations", type=int, default=None)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import apm_isaac.lab  # noqa: E402,F401  (registers the APM tasks)

env_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = a.num_envs
env_cfg.seed = a.seed
agent_cfg.seed = a.seed
if a.iterations:
    agent_cfg.max_iterations = a.iterations
a.out.mkdir(parents=True, exist_ok=True)
env = gym.make(a.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(a.out), device="cuda:0")
log: list = []
orig_log = runner.log


def log_hook(locs, *args, **kw):
    try:
        log.append({"it": int(locs["it"]), "reward": float(torch.tensor(list(locs["rewbuffer"])).mean())
                    if len(locs["rewbuffer"]) else None,
                    "ep_len": float(torch.tensor(list(locs["lenbuffer"])).mean()) if len(locs["lenbuffer"]) else None})
        if locs["it"] % 50 == 0:
            print("iter", log[-1], flush=True)
    except Exception:  # noqa: BLE001
        pass
    return orig_log(locs, *args, **kw)


runner.log = log_hook
t0 = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
normalizer = getattr(runner, "obs_normalizer", None)
export_policy_as_jit(policy_nn, normalizer, path=str(a.out), filename="policy.pt")
export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(a.out), filename="policy.onnx")
robot = env.unwrapped.scene["robot"]
info = {"task": a.task, "num_envs": a.num_envs, "iterations": agent_cfg.max_iterations, "seed": a.seed,
        "wall_s": round(time.time() - t0, 1), "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "action_scale": env_cfg.actions.joint_pos.scale, "sim_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation,
        "obs_terms": list(env_cfg.observations.policy.to_dict().keys())}
(a.out / "env_cfg.json").write_text(json.dumps(info, indent=1, default=str))
(a.out / "train_log.json").write_text(json.dumps(log))
print("done", info["wall_s"], "s", flush=True)
env.close()
app.close()
