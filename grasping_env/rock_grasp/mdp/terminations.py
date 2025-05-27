# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
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
    return distance < threshold

def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    center: tuple = (0.0, 0.0),
    radius: float = 1.0,
) -> torch.Tensor:
    """
    Terminate the episode if the object moves outside a circular boundary in the robot's frame.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot.
        object_cfg: Configuration for the object to monitor.
        center: The (x, y) coordinates of the circle's center in the robot's frame.
        radius: The radius of the circular boundary.

    Returns:
        A boolean tensor indicating which environments should terminate.
    """
    # Access the robot and object from the scene
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Get the object's position in the world frame
    object_pos_w = obj.data.root_pos_w[:, :3]

    # Transform the object's position to the robot's root frame
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    # Calculate the distance from the specified center in the robot's frame
    center_tensor = torch.tensor(center, device=object_pos_b.device)
    distance = torch.norm(object_pos_b[:, :2] - center_tensor, dim=1)

    # Determine which environments exceed the boundary
    return distance > radius


def object_height_reached(
    env: ManagerBasedRLEnv,
    target_height: float = 0.15,
    threshold: float = 0.02,
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

    success = height_error < threshold

    return success
