from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world_description.world_entity import Body, Connection
from typing_extensions import Union, Optional, Type, Any, Iterable

from .pick_up import GraspingActionDescription
from ...motions.container import OpeningMotion, ClosingMotion
from ...motions.gripper import MoveGripperMotion
from ....config.action_conf import ActionConfig
from ....datastructures.enums import (
    Arms,
    ContainerManipulationType,
    ApproachDirection,
    VerticalAlignment,
)
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....failures import ContainerManipulationError
from ....has_parameters import has_parameters
from ....language import SequentialPlan
from ....robot_description import ViewManager
from ....robot_plans.actions.base import ActionDescription


@has_parameters
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
        arm = ViewManager.get_arm_view(self.arm, self.robot_view)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
            rotate_gripper=True,
        )

        SequentialPlan(
            self.context,
            GraspingActionDescription(
                self.object_designator, self.arm, grasp_description
            ),
            OpeningMotion(self.object_designator, self.arm),
            MoveGripperMotion(
                GripperState.OPEN, self.arm, allow_gripper_collision=True
            ),
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the container is opened, this assumes that the container state can be read accurately from the
        real world.
        """
        validate_close_open(self.object_designator, self.arm, OpenAction)

    @classmethod
    def description(
        cls,
        object_designator_description: Union[Iterable[Body], Body],
        arm: Union[Iterable[Arms], Arms] = None,
        grasping_prepose_distance: Union[
            Iterable[float], float
        ] = ActionConfig.grasping_prepose_distance,
    ) -> PartialDesignator[OpenAction]:
        return PartialDesignator[OpenAction](
            OpenAction,
            object_designator=object_designator_description,
            arm=arm,
            grasping_prepose_distance=grasping_prepose_distance,
        )


@has_parameters
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
        arm = ViewManager.get_arm_view(self.arm, self.robot_view)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
            rotate_gripper=True,
        )

        SequentialPlan(
            self.context,
            GraspingActionDescription(
                self.object_designator, self.arm, grasp_description
            ),
            ClosingMotion(self.object_designator, self.arm),
            MoveGripperMotion(
                GripperState.OPEN, self.arm, allow_gripper_collision=True
            ),
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the container is closed, this assumes that the container state can be read accurately from the
        real world.
        """
        validate_close_open(self.object_designator, self.arm, CloseAction)

    @classmethod
    def description(
        cls,
        object_designator_description: Union[Iterable[Body], Body],
        arm: Union[Iterable[Arms], Arms] = None,
        grasping_prepose_distance: Union[
            Iterable[float], float
        ] = ActionConfig.grasping_prepose_distance,
    ) -> PartialDesignator[CloseAction]:
        return PartialDesignator[CloseAction](
            CloseAction,
            object_designator=object_designator_description,
            arm=arm,
            grasping_prepose_distance=grasping_prepose_distance,
        )


OpenActionDescription = OpenAction.description
CloseActionDescription = CloseAction.description
