"""Programme K6: a third embodiment. The Booster T1 humanoid (MuJoCo) on a learned walking controller.

**The motor controller** is a joystick walking policy trained with mujoco_playground's published PPO recipe
(`experiments/K_recovery/humanoid/train_t1.py`). It was trained with random pushes of 0.1–1.0 m/s, then frozen. It is
the humanoid's counterpart of NERVA's frozen neutral policy: it is not part of APM, and APM never sends it joint
commands.

**Here** it runs in plain MuJoCo:
- the exact compiled model it was trained on (`t1_model.mjb`);
- NumPy inference of the same network (observation normalisation, swish MLP, tanh mean action);
- the same observation: local linear velocity, gyro, gravity in the IMU frame, command (vx, vy, ω),
  joint-angle offsets, joint velocities, last action, gait phase;
- no training noise.

Joint targets are `default_pose + action`, sent to the model's position actuators. A zero command uses the
standing phase, as in training.

**Its other layers**, the same as the wheeled robot's:
- the (v, ω) expression adapter (`apm_express.wheeled.WheeledAdapter`; speeds within the policy's training range);
- an independent safety supervisor: tilt beyond 25° → zero command for 1 s and a counted trip;
- a fall is tilt > 50° or the pelvis below 0.45 m;
- the observation model (`wheeled_world.WheeledLogger`: quiet-window IMU, self-calibrated detector, its own acc8
  edges);
- the same `Camera` extension.

`closed_loop_encounter` is `wheeled_world.closed_loop_encounter` with this robot.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

DT = 0.02
TILT_TRIP, TILT_FALL, Z_FALL, BRAKE_S = math.radians(25), math.radians(50), 0.45, 1.0
GAIT_HZ = 1.5
ASSETS = Path(__file__).resolve().parents[2] / "experiments" / "K_recovery" / "humanoid" / "trained"


@lru_cache(maxsize=1)
def _policy(path: str) -> tuple:
    """Plain-NumPy copy of the trained Brax PPO policy (`t1_policy.npz`: observation mean/std for the 'state' key and
    the MLP layers hidden_0..hidden_3, converted from the training pickle)."""
    z = np.load(path)
    n = len([k for k in z.files if k.endswith("_kernel")])
    weights = [(z[f"hidden_{i}_kernel"], z[f"hidden_{i}_bias"]) for i in range(n)]
    return z["mean"], z["std"], weights


def _act(obs: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: list) -> np.ndarray:
    x = (obs - mean) / std
    for i, (w, b) in enumerate(weights):
        x = x @ w + b
        if i < len(weights) - 1:
            x = x / (1.0 + np.exp(-x))  # swish
    loc = x[: x.shape[0] // 2]
    return np.tanh(loc)


class HumanoidRobot:
    """Same interface as `wheeled_world.WheeledRobot`: apply(v, ω), advance(), pose(), push(), tilt(), fallen()."""

    # training-time observation noise scales (mujoco_playground T1 joystick); L audit A7 tests the policy with them
    NOISE = {"linvel": 0.1, "gyro": 0.2, "gravity": 0.05, "joint_pos": 0.03, "joint_vel": 1.5}

    def __init__(self, seed: int, scene_extender=None, assets: Path = ASSETS, obs_noise: float = 0.0):
        import mujoco

        if scene_extender is not None:
            raise NotImplementedError("rendering extras for the humanoid are added after the scene is loaded")
        self.mj = mujoco
        self.model = mujoco.MjModel.from_binary_path(str(assets / "t1_model.mjb"))
        self.data = mujoco.MjData(self.model)
        key = self.model.key("home")
        self.data.qpos[:] = key.qpos
        self.data.ctrl[:] = key.qpos[7:]
        mujoco.mj_forward(self.model, self.data)
        self.default = np.array(key.qpos[7:], float)
        self.n_sub = int(round(DT / self.model.opt.timestep))
        self.mean, self.std, self.weights = _policy(str(assets / "t1_policy.npz"))
        self.last_act = np.zeros(self.model.nu)
        self.phase = np.array([0.0, math.pi])
        self.phase_dt = 2 * math.pi * DT * GAIT_HZ
        self.rng = np.random.default_rng(seed)
        self.obs_noise = obs_noise
        self.noise_rng = np.random.default_rng([seed, 7])  # separate stream: obs_noise=0 leaves K's runs unchanged
        self.t, self.dt = 0.0, DT
        self.v_cmd = self.w_cmd = 0.0
        self.brake_until, self.trips = -1.0, 0
        self.imu_site = self.model.site("imu").id
        sens = {self.model.sensor(i).name: (self.model.sensor_adr[i], self.model.sensor_dim[i])
                for i in range(self.model.nsensor)}
        self.s_gyro, self.s_linvel, self.s_acc = sens["gyro"], sens["local_linvel"], sens["accelerometer"]
        self._origin = np.array(self.data.qpos[:2], float)

    def _sensor(self, s) -> np.ndarray:
        adr, dim = s
        return np.array(self.data.sensordata[adr:adr + dim], float)

    def pose(self) -> tuple[float, float, float]:
        q = self.data.qpos
        w, x, y, z = q[3:7]
        return float(q[0]), float(q[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def tilt(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))

    def push(self, dvx: float, dvy: float) -> None:
        self.data.qvel[0] += dvx
        self.data.qvel[1] += dvy

    def apply(self, v: float, w: float) -> bool:
        tripped = False
        if self.tilt() > TILT_TRIP:
            if self.t >= self.brake_until:
                self.trips += 1
            self.brake_until = self.t + BRAKE_S
            tripped = True
        if self.t < self.brake_until:
            v, w = 0.0, 0.0
        self.v_cmd += float(np.clip(v - self.v_cmd, -1.0 * DT, 1.0 * DT))
        self.w_cmd += float(np.clip(w - self.w_cmd, -3.0 * DT, 3.0 * DT))
        return tripped

    def _obs(self, cmd: np.ndarray) -> np.ndarray:
        gravity = self.data.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        phase = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
        parts = {"linvel": self._sensor(self.s_linvel), "gyro": self._sensor(self.s_gyro), "gravity": gravity,
                 "joint_pos": self.data.qpos[7:] - self.default, "joint_vel": np.array(self.data.qvel[6:], float)}
        if self.obs_noise > 0:  # uniform noise of the training recipe: x + (2U - 1) * level * scale
            parts = {k: v + (2 * self.noise_rng.random(v.shape) - 1) * self.obs_noise * self.NOISE[k]
                     for k, v in parts.items()}
        return np.concatenate([parts["linvel"], parts["gyro"], parts["gravity"], cmd, parts["joint_pos"],
                               parts["joint_vel"], self.last_act, phase])

    def advance(self) -> dict:
        cmd = np.array([self.v_cmd, 0.0, self.w_cmd])
        act = _act(self._obs(cmd), self.mean, self.std, self.weights)
        self.data.ctrl[:] = self.default + act
        for _ in range(self.n_sub):
            self.mj.mj_step(self.model, self.data)
        self.last_act = act
        self.t += DT
        nxt = np.fmod(self.phase + self.phase_dt + math.pi, 2 * math.pi) - math.pi
        self.phase = nxt if np.linalg.norm(cmd) > 0.01 else np.ones(2) * math.pi
        return {"acc": self._sensor(self.s_acc), "gyro": self._sensor(self.s_gyro), "tilt": math.degrees(self.tilt()),
                "finite": bool(np.all(np.isfinite(self.data.qpos))), "z": float(self.data.qpos[2])}

    def fallen(self, row: dict) -> bool:
        return math.radians(row["tilt"]) > TILT_FALL or row["z"] < Z_FALL or not row["finite"]


def closed_loop_encounter(seed: int, etype: str, intent, channels: dict | None = None, **kw):
    from apm_transfer.wheeled_world import closed_loop_encounter as enc
    return enc(seed, etype, intent, channels, robot_factory=HumanoidRobot, **kw)
