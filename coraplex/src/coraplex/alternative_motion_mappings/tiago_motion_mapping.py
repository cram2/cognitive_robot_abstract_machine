import logging
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.motions.base import AlternativeMotion
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.datastructures.definitions import GripperState
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
    ActionServerTask,
    RobotiqGripperActionServerTask,
)
from control_msgs.action import ParallelGripperCommand
from coraplex.datastructures.enums import ExecutionType, Arms
from coraplex.view_manager import ViewManager
from coraplex.robot_plans import (
    MoveMotion,
    MoveToolCenterPointMotion,
    LookingMotion,
    MoveGripperMotion,
)

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None

logger = logging.getLogger(__name__)


class TiagoMoveSim(MoveMotion, AlternativeMotion[Tiago]):
    """
    Uses a diff drive goal for the tiago base.
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self):

        world_T_target = self.world.transform(self.target, self.world.root)
        world_T_target.z = 0
        return DifferentialDriveBaseGoal(goal_pose=world_T_target, threshold=0.01)


class TiagoGripMotion(MoveGripperMotion, AlternativeMotion[Tiago]):
    """
    Uses RobotiqGripperActionServerTask to move Tiago's gripper.
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        logger.info(f"Performing action {self.__class__.__name__}")

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

        return RobotiqGripperActionServerTask(
            action_topic=action_topic,
            message_type=ParallelGripperCommand,
            target_position=self.target_position,
        )


# class TiagoMoveSim(MoveMotion, AlternativeMotion[Tiago]):
#     """
#     Uses a Nav2 action server to move the simulated Tiago base.
#     """
#
#     execution_type = ExecutionType.SIMULATED
#
#     def perform(self):
#         return
#
#     @property
#     def _motion_chart(self) -> NavigateActionServerTask:
#
#         return NavigateActionServerTask(
#             target_pose=self.target,
#             base_link=self.robot.root,
#             action_topic="/navigate_to_pose",
#             message_type=NavigateToPose,
#         )
