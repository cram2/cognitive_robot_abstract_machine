import logging
from dataclasses import dataclass
from coraplex.robot_plans.motions.base import AlternativeMotion
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.datastructures.definitions import GripperState
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    RobotiqGripperActionServerTask,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from control_msgs.action import ParallelGripperCommand
from coraplex.datastructures.enums import ExecutionType, Arms
from coraplex.view_manager import ViewManager
from coraplex.robot_plans import (
    MoveMotion,
    MoveToolCenterPointMotion,
    LookingMotion,
    MoveGripperMotion,
)

logger = logging.getLogger(__name__)


@dataclass
class TiagoGripperMotion(MoveGripperMotion, AlternativeMotion[Tiago]):
    """
    Uses RobotiqGripperActionServerTask to move Tiago's gripper.
    """

    execution_type = ExecutionType.REAL

    @property
    def _motion_chart(self) -> RobotiqGripperActionServerTask | Parallel:

        # Map GripperState to Robotiq target position.
        position_map = {
            GripperState.OPEN: 0.0,
            GripperState.MEDIUM: 0.35,
            GripperState.CLOSE: 0.7,
        }

        if self.motion not in position_map:
            raise ValueError(f"Unsupported motion state: {self.motion}")

        target_position = position_map[self.motion]

        # Map each arm to its Robotiq action server.
        arm_topics = {
            Arms.LEFT: "/left_gripper/robotiq_gripper_controller/gripper_cmd",
            Arms.RIGHT: "/right_gripper/robotiq_gripper_controller/gripper_cmd",
        }

        if self.gripper == Arms.BOTH:
            return Parallel(
                nodes=[
                    RobotiqGripperActionServerTask(
                        action_topic=arm_topics[arm],
                        message_type=ParallelGripperCommand,
                        target_position=target_position,
                    )
                    for arm in (Arms.LEFT, Arms.RIGHT)
                ]
            )

        if self.gripper not in arm_topics:
            raise ValueError(f"Unsupported gripper: {self.gripper}")

        return RobotiqGripperActionServerTask(
            action_topic=arm_topics[self.gripper],
            message_type=ParallelGripperCommand,
            target_position=target_position,
        )
