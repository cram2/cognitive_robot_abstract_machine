from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from typing_extensions import Optional, Any

from coraplex.config.action_conf import ActionConfig
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from coraplex.robot_plans.parameter_mixins import JointStatesKept, TargetLookedAt
from semantic_digital_twin.spatial_types import (
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class FaceAtAction(ActionDescription, TargetLookedAt, JointStatesKept):
    """
    Turn the robot chassis such that is faces the ``look_at_target`` and after that perform a
    look
    at action.
    """

    keep_joint_states: bool = field(
        default=ActionConfig.face_at_keep_joint_states, kw_only=True
    )
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        # get the robot position
        robot_position = self.robot.root.global_transform

        # calculate orientation for robot to face the object
        angle = (
            np.arctan2(
                robot_position.y - self.look_at_target.y,
                robot_position.x - self.look_at_target.x,
            )
            + np.pi
        )

        # create new robot pose
        new_robot_pose = Pose(
            robot_position.to_position(),
            Quaternion.from_rpy(0, 0, angle),
            reference_frame=self.world.root,
        )

        return sequential(
            [
                NavigateAction(target_location=new_robot_pose,
                        keep_joint_states=self.keep_joint_states,
                    ),  # turn robot
                    LookAtAction(look_at_target=self.look_at_target),  # look at the target
            ]
        )
