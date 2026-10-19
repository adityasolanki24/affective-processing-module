"""Experimental PhysX adapter for Booster Robotics' published T1 policy.

Observation and target conventions follow booster_gym deploy/utils/policy.py
at da396a06d6eed99e2de72d7749c48ee8748950f9 (Apache-2.0).
This is a sim-to-sim candidate, not a validated replacement controller.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from apm_isaac.robots import DT, LabPolicyBody, physx_like_isaac_lab

JOINTS = ["AAHead_yaw", "Head_pitch"] + [
    f"{side}_{joint}" for side in ("Left", "Right")
    for joint in ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")
] + ["Waist"] + [
    f"{side}_{joint}" for side in ("Left", "Right")
    for joint in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")
]


def observation(q, dq, default, gravity, gyro, command, phase, moving, last):
    """The official 47-value policy input; only twelve leg joints are controlled by RL."""
    return np.concatenate([gravity, gyro, command,
                           [math.cos(2 * math.pi * phase) * moving,
                            math.sin(2 * math.pi * phase) * moving],
                           (q - default)[11:], dq[11:] * 0.1, last]).astype(np.float32)


class BoosterPolicyBody(LabPolicyBody):
    def __init__(self, world, path, pos, yaw, seed, policy_dir: Path):
        # Reuse asset loading, joint-name mapping, IMU and safety machinery.
        from isaacsim.storage.native import get_assets_root_path
        usd = get_assets_root_path() + "/Isaac/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd"
        super().__init__(world, path, usd, pos, yaw, seed, policy_dir, 0.72, physics_dt=0.002)
        physx_like_isaac_lab(world.stage, path)
        self.vmap = None
        self.kp = np.asarray(self.cfg["stiffness"], float)
        self.kd = np.asarray(self.cfg["damping"], float)
        self.limits = np.asarray(self.cfg["torque_limit"], float)
        self.head_target = np.zeros(2)
        self.upper_offset = np.zeros(11)  # expressive posture: head, arms, waist (never the RL leg joints)
        self.target_filter = 0.2
        self.gait_frequency = 1.0
        profile = self.cfg.get("deployment_profile", {})
        if profile.get("simulation_gains", False):
            self.kp[11:] = [200., 200., 200., 200., 50., 50.] * 2
            self.kd[11:] = [5., 5., 5., 5., 1., 1.] * 2
        self.target_filter = float(profile.get("target_filter", self.target_filter))
        self.gait_frequency = float(profile.get("gait_hz", self.gait_frequency))
        # Booster's own deployment rate-limits commands (1 m/s^2 at 50 Hz); off unless the profile asks for it,
        # so the battery-tested candidate stays exactly as tested
        ramp = profile.get("command_ramp_m_s2")
        self.command_ramp = float(ramp) if ramp is not None else None
        self.cmd_smoothed = np.zeros(3)

    def post_reset(self, gains=None):
        super().post_reset()
        # Effort PD at each 2 ms physics step, torque limited as the official deployment.
        n = len(self.art.dof_names)
        self.art.get_articulation_controller().set_gains(kps=np.zeros(n), kds=np.zeros(n))
        self.last = np.zeros(12)
        self.filtered = self.default.copy()
        self.head_target = np.zeros(2)
        self.upper_offset = np.zeros(11)
        self.cmd_smoothed = np.zeros(3)

    def advance(self):
        q = np.asarray(self.art.get_joint_positions(), float)[self.idx]
        dq = np.asarray(self.art.get_joint_velocities(), float)[self.idx]
        r = self.rot()
        cmd = np.array([self.v_cmd, 0.0, self.w_cmd])
        if self.command_ramp is not None:
            step = self.command_ramp * DT
            self.cmd_smoothed = self.cmd_smoothed + np.clip(cmd - self.cmd_smoothed, -step, step)
            cmd = self.cmd_smoothed.copy()
        moving = float(np.linalg.norm(cmd) > 1e-5)
        obs = observation(q, dq, self.default, r.T @ np.array([0., 0., -1.]),
                          r.T @ np.asarray(self.art.get_angular_velocity()), cmd,
                          (self.t * self.gait_frequency) % 1.0, moving, self.last)
        with self.torch.no_grad():
            self.last = np.clip(self.policy(self.torch.from_numpy(obs)[None]).numpy()[0], -1., 1.)
        target = self.default.copy()
        target[11:] += self.last
        # Head motion is a separately bounded motor command; the leg policy is unchanged.
        if self.tilt() < self.TILT_TRIP and self.t >= self.brake_until:
            target[:2] += self.head_target
            target[:11] += self.upper_offset
        for _ in range(self.decim):
            self.filtered += self.target_filter * (target - self.filtered)
            q = np.asarray(self.art.get_joint_positions(), float)[self.idx]
            dq = np.asarray(self.art.get_joint_velocities(), float)[self.idx]
            effort = np.zeros(len(self.art.dof_names))
            effort[self.idx] = np.clip(self.kp * (self.filtered - q) - self.kd * dq,
                                       -self.limits, self.limits)
            self.art.set_joint_efforts(effort)
            self.world.step(render=False)
        self.t += DT
        return self._row()
