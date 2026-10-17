"""ValidatedMotionController for NERVA's Open Duck Mini v2 biped (MuJoCo), used read-only from NERVA's repository.

Policies (frozen ONNX files, hash-checked):
  "neutral"   turn-translation candidate: passes NERVA's long neutral motor gate (docs/neutral_motor_gate.md)
  "e1_prime"  E1′ styled candidate: passes the same gate at e = (0, 0); e_pitch moves torso pitch 1–3.4° while
              walking (docs/expressive_posture_e1prime.md); its crouch input is ignored and fixed at 0 here

Everything goes through NERVA's own deployment contract (`set_neutral_motor_contract`): only the seven trained
commands are accepted, the gait clock is NERVA's, and E1′ style values are domain-checked by NERVA.

Head channel: NERVA's contract forbids head offsets, because the policy was trained with zero head commands. With
`head_channel=True` this wrapper writes `sim.head_offset` directly. That adds the offsets to the head motor targets
after NERVA's speed limit, as the hardware runtime does, and leaves the observation's command slots at zero, as the
contract requires. This exists only so that H0 can *measure* an envelope (`prereg/H0_ADDENDUM.md`). The adapter
uses the channel only if H0 admits it. NERVA's code and contract are unchanged.

Requires the NERVA inference environment (Open Duck Playground at the pinned commit, mujoco 3.3.0,
onnxruntime 1.21.0) and the environment variable NERVA_ROOT (path to the NERVA checkout).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import math
import os
from pathlib import Path

import numpy as np

from apm_express.adapter import COMMAND_NAMES, COMMANDS, MotionRequest

POLICIES = {
    "neutral": ("experiments/cloud_runs/turn_translation-20261011-000711/candidate.onnx",
                "610b1c59446cb9f6067238a61a641a60aa706c48f4a56dc8b07d1956d4813446", False),
    "e1_prime": ("experiments/cloud_runs/e1_prime-20261011-132718/candidate.onnx",
                 "f67d5cac446f5ea81f3f5ffa26ea69155c50843443f57f0439d2e53d65ee2042", True),
}
DT = 0.02
HEAD_SLICE = slice(5, 9)  # actuator order: legs 0-4, neck_pitch, head_pitch, head_yaw, head_roll, legs 9-13


def nerva_root() -> Path:
    root = os.environ.get("NERVA_ROOT")
    if not root:
        raise RuntimeError("set NERVA_ROOT to the NERVA checkout")
    return Path(root)


def tilt_deg(quat_wxyz) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    cz = 1.0 - 2.0 * (x * x + y * y)  # body z axis . world z
    return math.degrees(math.acos(max(-1.0, min(1.0, cz))))


def yaw_pitch(quat_wxyz) -> tuple[float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return yaw, pitch


class NervaBiped:
    dt = DT

    def __init__(self, policy: str = "e1_prime", seed: int = 0, latency: bool = False, obs_noise: bool = True,
                 init_joint_noise: float = 0.02, head_channel: bool = False, scene_extender=None,
                 start_yaw: float = 0.0, max_head_rate: float | None = None, max_pitch_rate: float | None = None):
        from nerva.analysis.motor_eval import deployment_metadata
        from nerva.sim.open_duck import SCENE_BACKLASH, OpenDuckSim

        rel, digest, styled = POLICIES[policy]
        path = nerva_root() / rel
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"policy {policy} hash mismatch")
        with contextlib.redirect_stdout(io.StringIO()):  # upstream constructor prints local paths
            self.sim = OpenDuckSim(policy_path=path, scene=SCENE_BACKLASH, raw_accel=True, obs_noise=obs_noise,
                                   init_joint_noise=init_joint_noise, seed=seed, action_delay=latency,
                                   scene_extender=scene_extender)
        if start_yaw:
            self._set_start_yaw(start_yaw)
        self.sim.set_neutral_motor_contract(deployment_metadata(digest, styled))
        self.policy, self.styled, self.head_channel = policy, styled, head_channel
        # Controller-level change limits, independent of the adapter (None = unlimited; H0 measures raw steps).
        self.max_head_rate, self.max_pitch_rate = max_head_rate, max_pitch_rate
        self._head_applied = np.zeros(4)
        self._pitch_applied = 0.0
        if styled:
            self.sim.set_neutral_style(0.0, 0.0)
        model = self.sim.model
        joint_ids = model.actuator_trnid[:, 0]
        self.joint_lo = model.jnt_range[joint_ids, 0].copy()
        self.joint_hi = model.jnt_range[joint_ids, 1].copy()
        self.force_limit = np.abs(model.actuator_forcerange[:, 1]).copy()
        self._qpos_addr = model.jnt_qposadr[joint_ids]
        self.t = 0.0
        self._origin = self.sim.data.qpos[0:2].copy()
        # Head "face" direction: the world +x direction in the head body frame at the start pose, i.e. the
        # robot-eye camera axis of NERVA's world (nerva/sim/world.py, which assumes the home pose looks along +x).
        self._head_body = model.body("head_assembly").id
        r_head = self.sim.data.xmat[self._head_body].reshape(3, 3)
        base_yaw, _ = yaw_pitch(self.sim.data.qpos[3:7])
        forward_world = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
        self._face_local = r_head.T @ forward_world

    def head_direction(self) -> tuple[float, float]:
        """(yaw of the face relative to the base heading, elevation above horizontal) in rad."""
        f = self.sim.data.xmat[self._head_body].reshape(3, 3) @ self._face_local
        base_yaw, _ = yaw_pitch(self.sim.data.qpos[3:7])
        rel = (math.atan2(f[1], f[0]) - base_yaw + math.pi) % (2 * math.pi) - math.pi
        return rel, math.asin(max(-1.0, min(1.0, float(f[2]) / max(1e-9, float(np.linalg.norm(f))))))

    def _set_start_yaw(self, yaw: float) -> None:
        import mujoco

        q = self.sim.data.qpos
        q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.sim.model, self.sim.data)

    # ── controller interface (apm_express.episode.Controller) ────────────────
    def pose(self) -> tuple[float, float, float]:
        q = self.sim.data.qpos
        yaw, _ = yaw_pitch(q[3:7])
        return float(q[0] - self._origin[0]), float(q[1] - self._origin[1]), yaw

    def physical(self) -> dict:
        d = self.sim.data
        return {"tilt_deg": tilt_deg(d.qpos[3:7]), "angular_speed": float(np.linalg.norm(d.qvel[3:6])),
                "joint_pos": d.qpos[self._qpos_addr].copy(), "joint_lo": self.joint_lo, "joint_hi": self.joint_hi,
                "actuator_force": d.actuator_force.copy(), "force_limit": self.force_limit}

    def apply(self, request: MotionRequest) -> None:
        from nerva.interfaces import BehaviourCommand

        if request.command not in COMMAND_NAMES:
            raise ValueError(f"unknown command {request.command!r}")
        # NERVA validates (command, head offsets) together and requires zero head offsets; the command is
        # validated with zero offsets, and the head channel (if enabled) is set afterwards, motor targets only.
        self.sim.head_offset = np.zeros(4)
        self.sim.set_behaviour(BehaviourCommand(*COMMANDS[request.command]))
        if self.styled:
            e = float(request.e_pitch)
            if self.max_pitch_rate is not None:
                step = self.max_pitch_rate * DT
                e = float(np.clip(e, self._pitch_applied - step, self._pitch_applied + step))
            self._pitch_applied = e
            self.sim.set_neutral_style(e, 0.0)  # NERVA checks the admitted domain
        elif request.e_pitch != 0.0:
            raise ValueError("the neutral policy has no pitch input")
        head = np.asarray(request.head, dtype=float)
        if not self.head_channel:
            if np.any(head != 0.0):
                raise ValueError("head channel disabled")
        else:
            if head.shape != (4,) or not np.isfinite(head).all():
                raise ValueError("bad head request")
            if self.max_head_rate is not None:
                step = self.max_head_rate * DT
                head = np.clip(head, self._head_applied - step, self._head_applied + step)
            self._head_applied = head
            self.sim.head_offset = head  # motor targets only; observation command slots stay zero

    def push(self, dvx: float, dvy: float) -> None:
        self.sim.push(dvx, dvy)

    def advance(self) -> dict:
        row = self.sim.run(DT)[0]
        self.t = round(self.t + DT, 10)
        d = self.sim.data
        yaw, pitch = yaw_pitch(row.base_quat)
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        face_yaw, face_elev = self.head_direction() if finite else (float("nan"), float("nan"))
        return {"face_yaw": face_yaw, "face_elev": face_elev, "t": self.t,
                "x": float(row.base_pos[0] - self._origin[0]), "y": float(row.base_pos[1] - self._origin[1]),
                "z": float(row.base_pos[2]), "yaw": yaw, "pitch": pitch, "tilt": tilt_deg(row.base_quat),
                "vx_world": float(row.base_linvel[0]), "vy_world": float(row.base_linvel[1]),
                "angular_speed": float(np.linalg.norm(row.base_angvel)), "joint_pos": row.joint_pos,
                "head_joints": row.joint_pos[HEAD_SLICE].copy(), "power": float(row.power), "tau_sq": float(row.tau_sq),
                "contacts": row.contacts.astype(int), "finite": finite}
