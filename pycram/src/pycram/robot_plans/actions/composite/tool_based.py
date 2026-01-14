from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Union, Optional, Type, Any, Iterable

from semantic_digital_twin.robots.abstract_robot import ParallelGripper
from semantic_digital_twin.world_description.world_entity import Body
from ...motions.gripper import MoveTCPMotion
from .... import utils

from ....datastructures.enums import (
    Arms,
    VerticalAlignment,
    ApproachDirection,
    MovementType,
)
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped
from ....has_parameters import has_parameters
from ....language import SequentialPlan
from ....robot_plans.actions.base import ActionDescription


logger = logging.getLogger(__name__)


@has_parameters
@dataclass
class SimplePouringAction(ActionDescription):
    """
    Park the arms of the robot.
    """

    object_designator: Body
    """
    The object to pick up
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        gripper = self.world.get_semantic_annotations_by_type(ParallelGripper)[0]
        # tool_frame = gripper.tool_frame
        # grasp=gripper.front_facing_orientation

        # for arm_chain in self.robot_view.manipulator_chains:
        grasp = GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
        ).calculate_grasp_orientation(gripper.front_facing_orientation.to_np())

        object_pose = self.object_designator.global_pose
        object_pose.x += 0.009
        object_pose.y -= 0.125
        object_pose.z += 0.17

        def approach_or_rotate(rotate: bool) -> PoseStamped:
            ros_pose = PoseStamped.from_spatial_type(object_pose)

            if rotate:
                q = utils.axis_angle_to_quaternion(
                    [1, 0, 0], -110
                )  # ros_pose.rotate_by_quaternion(utils.quat_np_list(q))
                ros_pose.rotate_by_quaternion(utils.quat_np_list(q))
            tool_frame = gripper.tool_frame

            # man = next(iter(self.robot_view.manipulators))
            # for man in self.robot_view.manipulators:
            #     print(man)
            # tool_frame = man.tool_frame
            poseTg = PoseStamped.from_spatial_type(
                self.world.transform(ros_pose.to_spatial_type(), tool_frame)
            )
            poseTg.rotate_by_quaternion(grasp)

            return PoseStamped.from_spatial_type(
                self.world.transform(poseTg.to_spatial_type(), self.world.root)
            )

        pose = approach_or_rotate(False)
        pose_rot = approach_or_rotate(True)

        SequentialPlan(
            self.context,
            MoveTCPMotion(
                pose,
                self.arm,
                allow_gripper_collision=True,
                movement_type=MovementType.CARTESIAN,
            ),
            MoveTCPMotion(
                pose_rot,
                self.arm,
                allow_gripper_collision=True,
                movement_type=MovementType.CARTESIAN,
            ),
        ).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass

    @classmethod
    def description(
        cls,
        object_designator: Union[Iterable[PoseStamped], PoseStamped],
        arm: Union[Iterable[Arms], Arms],
    ) -> PartialDesignator[Type[SimplePouringAction]]:
        return PartialDesignator(cls, object_designator=object_designator, arm=arm)


SimplePouringActionDescription = SimplePouringAction.description
