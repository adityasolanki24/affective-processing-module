"""Isaac Lab fine-tuning task for Booster's official T1 policy (booster_gym `da396a0`, Apache-2.0) in PhysX.

Why: the official policy's gait is far better than v1's in the gait battery (stance slip -69%, joint acceleration
-60%, tilt -44%). It is less robust to 0.5 m/s pushes while walking, and it settles more slowly after a stop, so
the frozen promotion rule rejected it. This task fine-tunes **those exact weights** in PhysX, keeping the
deployment conventions identical so the result drops into `apm_isaac.booster.BoosterPolicyBody` unchanged.

Conventions (booster_gym deploy/utils/policy.py):
- **Observation, 47 values, in this order:**
  - projected gravity (3);
  - base angular velocity (3);
  - commands (vx, vy, wz), zeroed while standing (3);
  - cos and sin of the 1 Hz gait phase, zeroed while standing (2);
  - leg joint offsets from the default pose (12);
  - leg joint velocities x 0.1 (12);
  - last action (12).
- **Action:** 12 leg offsets from the default pose, scale 1, clipped to ±1.
- **Control:** 50 Hz (physics 2 ms, decimation 10).
- **Motors:** explicit torque PD (`IdealPDActuatorCfg`) with Booster's simulation gains for the legs, Booster's
  stiffness/damping for the head, arms and waist, and Booster's torque limits.
"""
from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from apm_isaac.lab.t1 import T1FlatEnvCfg, T1FlatPPORunnerCfg, T1Rewards

JOINTS = ["AAHead_yaw", "Head_pitch"] + [
    f"{side}_{joint}" for side in ("Left", "Right")
    for joint in ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")] + ["Waist"] + [
    f"{side}_{joint}" for side in ("Left", "Right")
    for joint in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]
LEGS = JOINTS[11:]
DEFAULT = [0, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0]
KP = [20.0] * 10 + [200.0] + [200.0, 200.0, 200.0, 200.0, 50.0, 50.0] * 2  # legs: Booster's simulation gains
KD = [0.2, 0.2] + [0.5] * 8 + [5.0] + [5.0, 5.0, 5.0, 5.0, 1.0, 1.0] * 2
TAU = [7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 30, 60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15]
GAIT_HZ = 1.0


def _pd(names: list[str]) -> IdealPDActuatorCfg:
    idx = [JOINTS.index(n) for n in names]
    return IdealPDActuatorCfg(joint_names_expr=names, stiffness={n: KP[i] for n, i in zip(names, idx, strict=True)},
                              damping={n: KD[i] for n, i in zip(names, idx, strict=True)},
                              effort_limit={n: float(TAU[i]) for n, i in zip(names, idx, strict=True)},
                              velocity_limit=30.0)


T1_BOOSTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                                                                   solver_position_iteration_count=8,
                                                                   solver_velocity_iteration_count=4)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.72),
                                               joint_pos=dict(zip(JOINTS, map(float, DEFAULT), strict=True))),
    soft_joint_pos_limit_factor=0.95,
    actuators={"upper": _pd(JOINTS[:11]), "legs": _pd(LEGS)},
)


def _moving(env) -> torch.Tensor:
    cmd = env.command_manager.get_command("base_velocity")
    return (torch.linalg.norm(cmd, dim=1, keepdim=True) > 1e-5).float()


def gated_commands(env) -> torch.Tensor:
    return env.command_manager.get_command("base_velocity") * _moving(env)


def gait_phase(env) -> torch.Tensor:
    t = env.episode_length_buf.float().unsqueeze(1) * env.step_dt
    phase = torch.fmod(t * GAIT_HZ, 1.0)
    m = _moving(env)
    return torch.cat([torch.cos(2 * torch.pi * phase) * m, torch.sin(2 * torch.pi * phase) * m], dim=1)


def stand_still(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """While commanded to stand: base horizontal speed squared plus leg joint speeds squared (faster settling)."""
    robot = env.scene[asset_cfg.name]
    still = 1.0 - _moving(env).squeeze(1)
    v = torch.sum(robot.data.root_lin_vel_b[:, :2] ** 2, dim=1)
    dq = torch.sum(robot.data.joint_vel[:, asset_cfg.joint_ids] ** 2, dim=1)
    return still * (v + 0.01 * dq)


def nonfinite_state(env) -> torch.Tensor:
    """Reset any environment whose physics state is non-finite or exploding (a crashed simulation)."""
    data = env.scene["robot"].data
    bad = ~torch.isfinite(data.root_pos_w).all(dim=1) | ~torch.isfinite(data.root_lin_vel_w).all(dim=1)
    bad = bad | ~torch.isfinite(data.joint_vel).all(dim=1)
    # exploding but still finite (a crashing robot): such values poison a gradient update before they become NaN
    return bad | (torch.linalg.norm(torch.nan_to_num(data.root_lin_vel_w), dim=1) > 10.0) |         (torch.nan_to_num(data.joint_vel).abs().amax(dim=1) > 100.0)


LEG_CFG = SceneEntityCfg("robot", joint_names=LEGS, preserve_order=True)


@configclass
class BoosterObs(ObsGroup):
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    commands = ObsTerm(func=gated_commands)
    phase = ObsTerm(func=gait_phase)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEG_CFG})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": LEG_CFG}, scale=0.1)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BoosterObservations:
    policy: BoosterObs = BoosterObs()


@configclass
class BoosterFTRewards(T1Rewards):
    stand_still = RewTerm(func=stand_still, weight=-2.0, params={"asset_cfg": LEG_CFG})


@configclass
class T1BoosterFTEnvCfg(T1FlatEnvCfg):
    observations: BoosterObservations = BoosterObservations()
    rewards: BoosterFTRewards = BoosterFTRewards()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = T1_BOOSTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.actions.joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=LEGS, scale=1.0,
                                                            use_default_offset=True, preserve_order=True,
                                                            clip={".*": (-1.0, 1.0)})
        # robustness: pushes up to 0.6 m/s (above the battery's 0.5), every 4-8 s, while walking and standing
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}}
        self.terminations.nonfinite = DoneTerm(func=nonfinite_state)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.rel_standing_envs = 0.25
        # keep the official gait: stronger penalties on sliding, jerk and action changes
        self.rewards.feet_slide.weight = -0.25
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.joint_deviation_upper = None  # upper joints are not actuated by the policy


@configclass
class T1BoosterFTRunnerCfg(T1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "apm_t1_booster_ft"
        self.max_iterations = 800
        self.empirical_normalization = False  # the official policy has no observation normaliser
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        self.policy.init_noise_std = 0.05  # fine-tuning a converged gait: large noise knocks it over
        self.policy.noise_std_type = "log"  # always positive (a scalar std can be driven below zero)
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001


gym.register(id="APM-Velocity-Flat-T1-BoosterFT-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}:T1BoosterFTEnvCfg",
                     "rsl_rl_cfg_entry_point": f"{__name__}:T1BoosterFTRunnerCfg"})
