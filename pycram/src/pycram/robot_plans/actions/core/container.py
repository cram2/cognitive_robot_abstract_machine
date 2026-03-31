from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Optional, Any

from pycram.config.action_conf import ActionConfig
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.pick_up import GraspingAction
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.container import OpeningMotion, ClosingMotion
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenAction(ActionDescription):
    """
    Opens a container like object
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be opened
    """
    arm: Arms
    """
    Arm that should be used for opening the container
    """
    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters the gripper should be at in the x-axis away from the handle.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
        )

        self.add_subplan(
            sequential(
                [
                    GraspingAction(self.object_designator, self.arm, grasp_description),
                    OpeningMotion(self.object_designator, self.arm),
                    MoveGripperMotion(
                        GripperState.OPEN, self.arm, allow_gripper_collision=True
                    ),
                ]
            )
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the container is opened, this assumes that the container state can be read accurately from the
        real world.
        """
        validate_close_open(self.object_designator, self.arm, OpenAction)


@dataclass
class CloseAction(ActionDescription):
    """
    Closes a container like object.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be closed
    """
    arm: Arms
    """
    Arm that should be used for closing
    """
    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters between the gripper and the handle before approaching to grasp.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
        )

        self.add_subplan(
            sequential(
                [
                    GraspingAction(self.object_designator, self.arm, grasp_description),
                    ClosingMotion(self.object_designator, self.arm),
                    MoveGripperMotion(
                        GripperState.OPEN, self.arm, allow_gripper_collision=True
                    ),
                ]
            )
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the container is closed, this assumes that the container state can be read accurately from the
        real world.
        """
        validate_close_open(self.object_designator, self.arm, CloseAction)
