"""Fetch a pinned official T1 candidate into ignored local inputs; never fetch latest."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SHA = "da396a06d6eed99e2de72d7749c48ee8748950f9"
BASE = f"https://raw.githubusercontent.com/BoosterRobotics/booster_gym/{SHA}/"
EXPECTED = {
    "policy.pt": "697cac61dbed2e76f25519c97ceb686f244d065dabf7d5d7ca386573143d215b",
    "T1.yaml": "08382a8ac111c0a1f9ca26fa0801675ebb0c642ad03c027431f9c632558b6b41",
    "LICENSE": "e7c650725db8a696fc587cb7ec88b7685f81841f12002e2385b47d7e54e6c0d3",
    "upstream_policy.py": "67398327a05267bf9e26cd8a6a34723cb60172b1b3e22a7c5b587438d31accaf",
}


def main():
    import yaml

    from apm_isaac.booster import JOINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--simulation-profile", action="store_true", help="apply the corrective PhysX pilot profile")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = {"policy.pt": "deploy/models/T1.pt", "T1.yaml": "deploy/configs/T1.yaml",
             "LICENSE": "LICENSE", "upstream_policy.py": "deploy/utils/policy.py"}
    hashes = {}
    for dest, source in files.items():
        data = urllib.request.urlopen(BASE + source, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != EXPECTED[dest]:
            raise ValueError(f"pinned upstream hash mismatch: {source}")
        (args.out / dest).write_bytes(data)
        hashes[source] = hashlib.sha256(data).hexdigest()
    cfg = yaml.safe_load((args.out / "T1.yaml").read_text())
    common = cfg["common"]
    env = {"joint_names": JOINTS, "default_joint_pos": common["default_qpos"],
           "controller_kind": "booster_gym_t1", "forward_command_scale": 1.0,
           "stiffness": common["stiffness"], "damping": common["damping"],
           "torque_limit": common["torque_limit"], "action_scale": 1.0}
    if args.simulation_profile:
        env["deployment_profile"] = {"simulation_gains": True, "target_filter": 1.0, "gait_hz": 1.0}
    (args.out / "env_cfg.json").write_text(json.dumps(env, indent=2))
    (args.out / "source.json").write_text(json.dumps({"repository": "BoosterRobotics/booster_gym",
                                                     "commit": SHA, "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
