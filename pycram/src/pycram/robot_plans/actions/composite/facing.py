from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Type, Any, Iterable

from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from ..core.navigation import LookAtActionDescription, NavigateActionDescription
from ....config.action_conf import ActionConfig
from ....datastructures.partial_designator import PartialDesignator
from ....has_parameters import has_parameters
from ....language import SequentialPlan
from ....robot_plans.actions.base import ActionDescription
from ....tf_transformations import quaternion_from_euler


@has_parameters
@dataclass
class FaceAtAction(ActionDescription):
    """
    Turn the robot chassis such that is faces the ``pose`` and after that perform a look at action.
    """

    pose: Pose
    """
    The pose to face 
    """
    keep_joint_states: bool = ActionConfig.face_at_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self) -> None:
        # get the robot position
        robot_position = self.robot_view.root.global_pose

        # calculate orientation for robot to face the object
        angle = (
            np.arctan2(
                robot_position.to_position().y.to_np()[0]
                - self.pose.to_position().y.to_np()[0],
                robot_position.to_position().x.to_np()[0]
                - self.pose.to_position().x.to_np()[0],
            )
            + np.pi
        )

        # create new robot pose
        new_robot_pose = Pose(
            robot_position.to_position(),
            Quaternion.from_rpy(0, 0, angle),
            reference_frame=self.world.root,
        )

        # turn robot
        SequentialPlan(
            self.context,
            NavigateActionDescription(new_robot_pose, self.keep_joint_states),
            # look at target
            LookAtActionDescription(self.pose),
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        # The validation will be done in the LookAtActionPerformable.perform() method so no need to validate here.
        pass

    @classmethod
    def description(
        cls,
        pose: Union[Iterable[Pose], Pose],
        keep_joint_states: Union[
            Iterable[bool], bool
        ] = ActionConfig.face_at_keep_joint_states,
    ) -> PartialDesignator[Type[FaceAtAction]]:
        return PartialDesignator(
            FaceAtAction, pose=pose, keep_joint_states=keep_joint_states
        )


FaceAtActionDescription = FaceAtAction.description
