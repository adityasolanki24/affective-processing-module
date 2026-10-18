"""APM v1.0: Isaac Lab velocity-tracking task for the Booster T1 (stock Isaac Sim USD), flat ground.

Why it exists: the MuJoCo-trained T1 policy does not transfer to PhysX. See `experiments/M_release` M3 for the
measurements.

**Template.** It adapts Isaac Lab's G1 flat velocity task (`config/g1`), with:
- **robot:** T1 joint and link names, a home pose equal to the MuJoCo playground keyframe, and PD gains sized for
  PhysX implicit drives (legs 100–150 N·m/rad, ankles 40, arms and head 20–40);
- **commands for APM's range:** forward −0.2–0.6 m/s, lateral ±0.2, yaw ±1 rad/s, with 15% standing-still
  commands (stop and start);
- **robustness:** pushes stay on, base velocity ±0.5 m/s every 10–15 s, matching L A7's measured envelope.

Policy: 50 Hz (physics 200 Hz, decimation 4). The observations are Isaac Lab's standard ones: base linear and
angular velocity, projected gravity, commands, joint position and velocity offsets, last action.
"""
from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

HOME = {"AAHead_yaw": 0.0, "Head_pitch": 0.0, "Left_Shoulder_Pitch": 0.0, "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": -0.4, "Right_Shoulder_Pitch": 0.0, "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Pitch": 0.0, "Right_Elbow_Yaw": 0.4, "Waist": 0.0, "Left_Hip_Pitch": -0.2,
        "Left_Hip_Roll": 0.0, "Left_Hip_Yaw": 0.0, "Left_Knee_Pitch": 0.4, "Left_Ankle_Pitch": -0.2,
        "Left_Ankle_Roll": 0.0, "Right_Hip_Pitch": -0.2, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
        "Right_Knee_Pitch": 0.4, "Right_Ankle_Pitch": -0.2, "Right_Ankle_Roll": 0.0}  # MuJoCo playground home

T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                                                                   solver_position_iteration_count=8,
                                                                   solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.70), joint_pos=HOME),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*_Hip_.*", ".*_Knee_Pitch", "Waist"],
                                    stiffness={".*_Hip_.*": 100.0, ".*_Knee_Pitch": 150.0, "Waist": 100.0},
                                    damping={".*_Hip_.*": 3.0, ".*_Knee_Pitch": 4.0, "Waist": 3.0},
                                    armature=0.01),
        "feet": ImplicitActuatorCfg(joint_names_expr=[".*_Ankle_.*"], stiffness=40.0, damping=2.0, armature=0.01),
        "arms": ImplicitActuatorCfg(joint_names_expr=[".*_Shoulder_.*", ".*_Elbow_.*"], stiffness=40.0,
                                    damping=2.0, armature=0.01),
        "head": ImplicitActuatorCfg(joint_names_expr=["AAHead_yaw", "Head_pitch"], stiffness=20.0, damping=1.0,
                                    armature=0.005),
    },
)


@configclass
class T1Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
                                   params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0,
                                  params={"command_name": "base_velocity", "std": 0.5})
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.75,
                            params={"command_name": "base_velocity", "threshold": 0.4,
                                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.1,
                         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                                 "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_.*"])})
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                                  params={"asset_cfg": SceneEntityCfg("robot",
                                                                      joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])})
    joint_deviation_upper = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                                        ".*_Shoulder_.*", ".*_Elbow_.*", "Waist", "AAHead_yaw", "Head_pitch"])})


@configclass
class T1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: T1Rewards = T1Rewards()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 3.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}}
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_Hip_.*",
                                                                                          ".*_Knee_Pitch"])
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_Hip_.*",
                                                                                              ".*_Knee_Pitch"])
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"


@configclass
class T1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "apm_t1_flat"
        self.max_iterations = 2500


gym.register(id="APM-Velocity-Flat-T1-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}:T1FlatEnvCfg",
                     "rsl_rl_cfg_entry_point": f"{__name__}:T1FlatPPORunnerCfg"})


@configclass
class T1FlatEnvCfgV2(T1FlatEnvCfg):
    """v2, after the v1 gate battery (v1 undershoots slow commands: 0.1 m/s → ~0, 0.2 → 0.13 m/s).
    - Yaw-rate commands sampled directly (APM sends ω, not a heading).
    - Tighter velocity tracking (σ 0.25 instead of 0.5).
    - Forward −0.1–0.5 m/s and lateral ±0.1 m/s (APM's operating range).
    - Training pushes up to ±0.8 m/s (v1: ±0.5)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.events.push_robot.params = {"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}}


@configclass
class T1FlatPPORunnerCfgV2(T1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "apm_t1_flat_v2"
        self.max_iterations = 3000


gym.register(id="APM-Velocity-Flat-T1-v2", entry_point="isaaclab.envs:ManagerBasedRLEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}:T1FlatEnvCfgV2",
                     "rsl_rl_cfg_entry_point": f"{__name__}:T1FlatPPORunnerCfgV2"})
