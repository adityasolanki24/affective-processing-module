"""Fine-tune Booster's official T1 policy in Isaac Lab (PhysX), starting from its exact weights.

    $ISAAC_PY experiments/M_release/isaac/lab_finetune_booster.py --num-envs 4096 --iterations 800 \
        --warmup 100 --out out/t1_booster_ft

1. The actor (47 -> 256 -> 128 -> 128 -> 12, ELU) is loaded from `policies/t1_booster_candidate/policy.pt`.
2. Critic warm-up: `--warmup` iterations with the actor and its noise frozen, so an untrained value function
   cannot pull the gait away from the official one.
3. Everything is fine-tuned for the remaining iterations at a low learning rate.
4. Export: `policy.pt` (TorchScript actor, the same interface as the official file) and `env_cfg.json` (the
   candidate's deployment profile, plus command smoothing as in Booster's own deployment). The directory drops
   into `apm_isaac.booster.BoosterPolicyBody` unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=4096)
ap.add_argument("--iterations", type=int, default=800)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", type=Path, required=True)
a, _ = ap.parse_known_args()
app = AppLauncher({"headless": True}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import apm_isaac.lab  # noqa: E402,F401  (registers the APM tasks)

TASK = "APM-Velocity-Flat-T1-BoosterFT-v0"
SRC = REPO / "experiments" / "M_release" / "policies" / "t1_booster_candidate"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = a.num_envs
env_cfg.seed = agent_cfg.seed = a.seed
a.out.mkdir(parents=True, exist_ok=True)
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(a.out), device="cuda:0")
policy = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

# 1. official weights into the actor (identical layer layout: Linear, ELU, Linear, ELU, Linear, ELU, Linear)
official = torch.jit.load(str(SRC / "policy.pt"), map_location="cuda:0")
with torch.no_grad():
    for i in (0, 2, 4, 6):
        src = getattr(official, str(i))
        policy.actor[i].weight.copy_(src.weight)
        policy.actor[i].bias.copy_(src.bias)
obs = env.get_observations()
obs_t = obs[0] if isinstance(obs, tuple) else obs
obs_t = obs_t["policy"] if hasattr(obs_t, "keys") else obs_t
with torch.no_grad():
    gap = float((policy.actor(obs_t[:64]) - official(obs_t[:64])).abs().max())
print("loaded official actor; max output difference", gap, flush=True)
if gap > 1e-4:
    raise RuntimeError("the actor does not reproduce the official policy")

log: list = []
orig_log = runner.log


def log_hook(locs, *args, **kw):
    try:
        log.append({"it": int(locs["it"]), "reward": float(torch.tensor(list(locs["rewbuffer"])).mean())
                    if len(locs["rewbuffer"]) else None,
                    "ep_len": float(torch.tensor(list(locs["lenbuffer"])).mean()) if len(locs["lenbuffer"]) else None})
        if locs["it"] % 25 == 0:
            print("iter", log[-1], flush=True)
    except Exception:  # noqa: BLE001
        pass
    return orig_log(locs, *args, **kw)


runner.log = log_hook
t0 = time.time()
# 2. critic warm-up: actor and its noise frozen
frozen = list(policy.actor.parameters()) + [p for n, p in policy.named_parameters() if "std" in n]
for p in frozen:
    p.requires_grad_(False)
runner.learn(num_learning_iterations=a.warmup, init_at_random_ep_len=True)
for p in frozen:
    p.requires_grad_(True)
# 3. fine-tune everything
runner.learn(num_learning_iterations=a.iterations - a.warmup, init_at_random_ep_len=False)
# 4. export, in the official interface
export_policy_as_jit(policy, None, path=str(a.out), filename="policy.pt")
cfg = json.loads((SRC / "env_cfg.json").read_text())
cfg["deployment_profile"] = {**cfg.get("deployment_profile", {}), "command_ramp_m_s2": 1.0}
cfg["fine_tune"] = {"task": TASK, "num_envs": a.num_envs, "iterations": a.iterations, "warmup": a.warmup,
                    "seed": a.seed, "wall_s": round(time.time() - t0, 1),
                    "initial_policy_sha256": hashlib.sha256((SRC / "policy.pt").read_bytes()).hexdigest(),
                    "policy_sha256": hashlib.sha256((a.out / "policy.pt").read_bytes()).hexdigest()}
(a.out / "env_cfg.json").write_text(json.dumps(cfg, indent=1))
for f in ("LICENSE", "T1.yaml", "source.json"):
    shutil.copy(SRC / f, a.out / f)
(a.out / "train_log.json").write_text(json.dumps(log))
print("done", cfg["fine_tune"]["wall_s"], "s", flush=True)
from apm_isaac import hard_exit  # noqa: E402

hard_exit(0)
