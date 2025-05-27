# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg, SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import TiledCameraCfg, CameraCfg, ContactSensorCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg

from . import mdp

#Pre defined configs

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                      init_state = ArticulationCfg.InitialStateCfg(
                                                          joint_pos={
                                                            "panda_joint1": 0.0,
                                                            "panda_joint2": -0.569,
                                                            "panda_joint3": 0.0,
                                                            "panda_joint4": -1.810,
                                                            "panda_joint5": 0.0,
                                                            "panda_joint6": 1.4, #3.037 for simpel agent
                                                            "panda_joint7": 0.741,
                                                            "panda_finger_joint.*": 0.04,
                                                        },
                                                      )
                                                      )

    
    #Depth camera to the side of base
    
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path = "{ENV_REGEX_NS}/Robot/panda_link0/camera",
        update_period = 0.1,
        width = 320, #1280 #214
        height = 240, #720 #120
        data_types=["rgb", "depth"],
        depth_clipping_behavior = "zero",
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix = [192.384, 0, 158.967, 0, 192.384, 122.075, 0, 0, 1], #641.281, 0, 636.557, 0, 641.281, 366.917, 0, 0, 1
                                                                 width = 320,
                                                                 height = 240,
                                                                 clipping_range = (0.01, 2),
        ),
        offset = TiledCameraCfg.OffsetCfg(pos=(1.2, 0.4, 0.55), rot=(-0.269, 0.457, 0.730, -0.429), convention = "ros"),
        debug_vis = True,
    )

    """
    #Second camera

    tiled_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path = "{ENV_REGEX_NS}/Robot/panda_link0/camera2",
        update_period = 0.1,
        width = 320, #1280 #214
        height = 240, #720 #120
        data_types=["rgb", "depth"],
        depth_clipping_behavior = "zero",
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix = [192.384, 0, 158.967, 0, 192.384, 122.075, 0, 0, 1], #641.281, 0, 636.557, 0, 641.281, 366.917, 0, 0, 1
                                                                 width = 320,
                                                                 height = 240,
                                                                 clipping_range = (0.01, 2),
        ),
        offset = TiledCameraCfg.OffsetCfg(pos=(0.0, 0.6, 0.4), rot=(0.192, -0.319, 0.794, -0.479), convention = "ros"),
        debug_vis = True,
    )"""

    """
    #End effector cam
    
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/camera",
        update_period = 0.1,
        width = 320, #1280 #214
        height = 240, #720 #120
        data_types=["rgb", "depth"],
        depth_clipping_behavior = "zero",
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix = [48.7, 0, 158.967, 0, 36.5, 122.075, 0, 0, 1], #641.281, 0, 636.557, 0, 641.281, 366.917, 0, 0, 1
                                                                 width = 320,
                                                                 height = 240,
                                                                 clipping_range = (0.01, 10),
        ),
        offset = TiledCameraCfg.OffsetCfg(pos=(0.12, 0.0, 0.02), rot=(0.7064338, 0.0308436, -0.0308436, -0.7064338), convention = "ros"),
        debug_vis = True,
    )"""

    """contact_force_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger", 
        update_period=0.0, 
        history_length=6, 
        debug_vis=True,
        filter_prim_paths_expr = ["/World/envs/env_.*/Object"],
    )"""

    """contact_force_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", 
        update_period=0.0,
        history_length=1, 
        debug_vis=True,
        filter_prim_paths_expr = ["{ENV_REGEX_NS}/Object"],
    )"""

    #marker_cfg = FRAME_MARKER_CFG.copy()
    #marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    #marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
    
    object: RigidObjectCfg | DeformableObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            #debug_vis = True,
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.01], rot=[1, 0, 0, 0]), #0.7071, 0, 0, 0.7071
            spawn=UsdFileCfg(
                #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                #usd_path=f"/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/rock_grasp/scene_objects/rock_1/rock.usd",
                usd_path=f"/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/rock_grasp/scene_objects/rock_obj/rock_test.usd",
                scale=(0.65, 0.65, 0.65),
                #rigid_props=RigidBodyPropertiesCfg(
                #    solver_position_iteration_count=16,
                #    solver_velocity_iteration_count=1,
                #    max_angular_velocity=1000.0,
                #    max_linear_velocity=1000.0,
                #    max_depenetration_velocity=5.0,
                #    disable_gravity=False,
                #    enable_gyroscopic_forces = True,
                #),
                #mass_props = MassPropertiesCfg(
                #    mass = 0,
                #    density = 2582,
                #),
            ),
        )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        depth_img = ObsTerm(func=mdp.image, params={"data_type": "depth"})
        #depth_img2 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg('tiled_camera2'), "data_type": "depth"})
        #rgb_img = ObsTerm(func=mdp.image, params={"data_type": "rgb", "normalize": False})
        #rgb_img2 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg('tiled_camera2'), "data_type": "rgb"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        #object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        #actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object_fine = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=10.0)

    reaching_object_rough = RewTerm(func=mdp.object_ee_distance, params={"std": 0.3}, weight=20.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=100.0)

    ee_ori = RewTerm(func=mdp.ee_orientation, weight = 1)  #Reward ee poitning downwards

    ee_open_dist = RewTerm(func=mdp.gripper_open_when_far, weight = 1) #Reward for open until within 2 cm of rock

    in_between = RewTerm(func=mdp.in_between, weight = 1000) #Reward for the endeffector is within 2 cm of rock

    ee_close_dist = RewTerm(func=mdp.gripper_closure_near_object, weight = 20) #reward closing gripper when withing 2 cm of rock

    object_height_tracking = RewTerm(func=mdp.object_height_tracking, params={"target_height": 0.15, "minimal_height": 0.04, "std": 0.3}, weight = 16.0)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #effort = RewTerm(func=mdp.applied_torque_limits, weight = -1)

    terminated = RewTerm(func=mdp.is_terminated, weight = -500)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    #ill_effor = DoneTerm(func=mdp.joint_effort_out_of_limit)

    #contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_force_left")})

    #object_dropping = DoneTerm(
    #    func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    #)

    termination_cfg = DoneTerm(func=mdp.object_out_of_bounds, params={"center": (0.5, 0.0), "radius": 0.45}, time_out=False)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    """action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )"""



##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5) #env_spacing=2.5
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
