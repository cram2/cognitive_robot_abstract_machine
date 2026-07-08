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


@dataclass
class TracyJointMotionMapping(MoveJointsMotion, AlternativeMotion[Tracy]):

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        logger.debug(f"performing {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self):
        joint_goal = super()._motion_chart
        return joint_goal


@dataclass
class TracyGripperMotion(MoveGripperMotion, AlternativeMotion[Tracy]):
    """
    Uses RobotiqGripperActionServerTask to move Tracy's gripper.
    """

    execution_type = ExecutionType.REAL

    motion: Optional[GripperState] = None
    position: Optional[float] = None

    def __post_init__(self):
        if self.motion is None and self.position is None:
            raise ValueError("You must specify either 'motion' or 'position'.")
        if self.motion is not None and self.position is not None:
            raise ValueError(
                "Cannot specify both 'motion' and 'position' at the same time."
            )

    def perform(self):
        logger.debug(f"Performing {self.__class__.__name__}")
        return

    @property
    def _motion_chart(self) -> RobotiqGripperActionServerTask:
        if self.gripper == Arms.LEFT:
            action_topic = "/left_gripper/robotiq_gripper_controller/gripper_cmd"
        elif self.gripper == Arms.RIGHT:
            action_topic = "/right_gripper/robotiq_gripper_controller/gripper_cmd"
        else:
            raise ValueError(f"Unsupported gripper: {self.gripper}")

        # Determine target position
        if self.position is not None:
            target_position = self.position
        elif self.motion == GripperState.OPEN:
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
