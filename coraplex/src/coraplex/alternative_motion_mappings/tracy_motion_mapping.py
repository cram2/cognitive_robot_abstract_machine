import logging
from dataclasses import dataclass
from typing import Optional
from semantic_digital_twin.datastructures.definitions import GripperState


from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    ActionServerTask,
    RobotiqGripperActionServerTask,
)
from control_msgs.action import ParallelGripperCommand

from semantic_digital_twin.robots.tracy import Tracy
from coraplex.datastructures.enums import ExecutionType, Arms
from coraplex.view_manager import ViewManager
from coraplex.robot_plans import (
    MoveJointsMotion,
    MoveToolCenterPointMotion,
    LookingMotion,
    MoveGripperMotion,
)

from coraplex.robot_plans.motions.base import AlternativeMotion

logger = logging.getLogger(__name__)

class TracyJointMotionMapping(MoveJointsMotion,AlternativeMotion[Tracy]):

    execution_type =  ExecutionType.REAL

    def perform(self):
        logger.debug(f"performing {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self):
        joint_goal = super()._motion_chart

        return joint_goal

class TracyGripMotion(MoveGripperMotion, AlternativeMotion[Tracy]):
    execution_type = ExecutionType.REAL

    def __init__(
            self,
            gripper: Arms,
            motion: Optional[GripperState] = None,
            position: Optional[float] = None,
            allow_gripper_collision: Optional[bool] = None
    ):
        if motion is None and position is None:
            raise ValueError("You must specify either 'motion' or 'position'.")
        if motion is not None and position is not None:
            raise ValueError("Cannot specify both 'motion' and 'position' at the same time.")

        base_motion_placeholder = motion if motion is not None else GripperState.OPEN

        super().__init__(
            motion=base_motion_placeholder,
            gripper=gripper,
            allow_gripper_collision=allow_gripper_collision
        )

        self.position = position

    def perform(self):
        logger.debug(f"performing {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self) -> RobotiqGripperActionServerTask:
        if self.gripper is None:
            raise ValueError("No gripper specified")

        if self.gripper == Arms.LEFT:
            action_topic = "/left_gripper/robotiq_gripper_controller/gripper_cmd"
        elif self.gripper == Arms.RIGHT:
            action_topic = "/right_gripper/robotiq_gripper_controller/gripper_cmd"
        else:
            raise ValueError(f"Unsupported gripper {self.gripper}")

        if self.position is not None:
            target_position = self.position
        else:
            if self.motion == GripperState.OPEN:
                target_position = 0.0
            elif self.motion == GripperState.CLOSE:
                target_position = 0.7
            else:
                raise ValueError(f"Unsupported motion state: {self.motion}")

        return RobotiqGripperActionServerTask(
            action_topic=action_topic,
            message_type=ParallelGripperCommand,
            target_position=target_position,
        )