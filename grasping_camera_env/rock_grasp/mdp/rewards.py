# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def ee_orientation(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    scale: float = 1.0,
) -> torch.Tensor:
    
    ee_frame = env.scene[ee_frame_cfg.name]

    # Retrieve the quaternion of the end-effector in the world frame
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # Shape: (num_envs, 4)

    # Extract the quaternion components
    w, x, y, z = ee_quat_w.unbind(-1)  # Unbind the quaternion (w, x, y, z)

    # Compute the z-axis from the quaternion using the standard formula
    # The z-axis is the third column of the rotation matrix for this quaternion
    z_axis = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)
    ], dim=-1)  # Shape: (num_envs, 3)

    # Define the world down vector
    down_vector = torch.tensor([0.0, 0.0, -1.0], device=z_axis.device)

    # Compute the cosine similarity between the z-axis and the down vector
    cos_theta = torch.sum(z_axis * down_vector, dim=1)  # Shape: (num_envs,)

    # Compute the reward
    reward = scale * cos_theta

    return reward

def gripper_open_when_far(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.02,
    gripper_open_threshold: float = 0.07,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward the agent for keeping the gripper open when the end-effector is more than a specified distance away from the object.
    """
    # Retrieve object and end-effector positions
    object_pos = env.scene[object_cfg.name].data.root_pos_w  # Shape: (num_envs, 3)
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)

    # Compute distance between end-effector and object
    distance = torch.norm(ee_pos - object_pos, dim=1)  # Shape: (num_envs,)

    # Retrieve the robot articulation
    robot = env.scene[robot_cfg.name]

    # Get joint positions
    joint_positions = robot.data.joint_pos  # Shape: (num_envs, num_joints)

    # Define indices for the gripper finger joints
    # Replace these indices with the actual indices for your robot's gripper joints
    left_finger_index = 7
    right_finger_index = 8

    # Compute gripper opening width
    gripper_width = joint_positions[:, left_finger_index] + joint_positions[:, right_finger_index]  # Shape: (num_envs,)

    # Determine if the gripper is considered open
    gripper_open = gripper_width > gripper_open_threshold  # Shape: (num_envs,)

    # Determine if the end-effector is far from the object
    ee_far = distance > distance_threshold  # Shape: (num_envs,)

    # Reward is 1.0 if both conditions are met, else 0.0
    reward = (gripper_open & ee_far).float()  # Shape: (num_envs,)

    return reward

def gripper_closure_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.02,
    max_gripper_width: float = 0.07,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward the agent based on how closed the gripper is when the end-effector is within a specified proximity to the object.
    """
    # Retrieve object and end-effector positions
    object_pos = env.scene[object_cfg.name].data.root_pos_w  # Shape: (num_envs, 3)
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)

    # Compute distance between end-effector and object
    distance = torch.norm(ee_pos - object_pos, dim=1)  # Shape: (num_envs,)

    # Retrieve the robot articulation
    robot = env.scene[robot_cfg.name]

    # Get joint positions
    joint_positions = robot.data.joint_pos  # Shape: (num_envs, num_joints)

    # Define indices for the gripper finger joints
    # Replace these indices with the actual indices for your robot's gripper joints
    left_finger_index = 7
    right_finger_index = 8

    # Compute gripper opening width
    gripper_width = joint_positions[:, left_finger_index] + joint_positions[:, right_finger_index]  # Shape: (num_envs,)

    # Normalize gripper closure: 1.0 when fully closed, 0.0 when fully open
    gripper_closure = 1.0 - (gripper_width / max_gripper_width)
    gripper_closure = torch.clamp(gripper_closure, 0.0, 1.0)  # Ensure values are within [0, 1]

    # Determine if the end-effector is within the proximity threshold
    within_proximity = distance < distance_threshold  # Shape: (num_envs,)

    # Compute the reward: gripper closure scaled by proximity condition
    reward = torch.where(within_proximity, gripper_closure, torch.zeros_like(gripper_closure))  # Shape: (num_envs,)

    return reward

def in_between(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for keeping the gripper open when the end-effector is more than a specified distance away from the object.
    """
    # Retrieve object and end-effector positions
    object_pos = env.scene[object_cfg.name].data.root_pos_w  # Shape: (num_envs, 3)
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)

    # Compute distance between end-effector and object
    distance = torch.norm(ee_pos - object_pos, dim=1)  # Shape: (num_envs,)

    # Determine if the end-effector is far from the object
    ee_far = distance < distance_threshold  # Shape: (num_envs,)

    # Reward is 1.0 if both conditions are met, else 0.0
    reward = (ee_far)  # Shape: (num_envs,)

    return reward

def object_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float,
    minimal_height: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward based on the object's proximity to a target z-height using a tanh kernel.
    """
    # Extract the object from the environment
    object: RigidObject = env.scene[object_cfg.name]

    # Get the object's current z-position
    current_z = object.data.root_pos_w[:, 2]

    # Calculate the absolute difference from the target height
    height_error = torch.abs(current_z - target_height)

    # Compute the reward using a tanh kernel
    reward = (current_z > minimal_height) * (1 - torch.tanh(height_error / std))

    return reward


