from giskardpy.motion_statechart.tasks.ros_tasks import ActionServerTask
from semantic_digital_twin.robots.hsrb import HSRB
from ..datastructures.enums import ExecutionType
from ..robot_plans import MoveMotion, MoveGripperMotion

from ..robot_plans.motions.base import AlternativeMotionMapping

from nav2_msgs.action import NavigateToPose


class HSRBMoveMotionMapping(MoveMotion, AlternativeMotionMapping[HSRB]):
    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> ActionServerTask:
        nav_goal = NavigateToPose.Goal(pose=self.target.ros_message())
        return ActionServerTask(
            action_topic="/hsrb/move_base/move/goal",  # Adapt to real topic
            goal_msg=nav_goal,
            node_handle=self.plan.context.ros_node,
        )


class HSRBMoveGripper(MoveGripperMotion, AlternativeMotionMapping[HSRB]):
    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> ActionServerTask:
        return ActionServerTask(
            action_topic="/hsrb/gripper",  # Adapt to real topic
            goal_msg=None,
            node_handle=self.plan.context.ros_node,
        )
