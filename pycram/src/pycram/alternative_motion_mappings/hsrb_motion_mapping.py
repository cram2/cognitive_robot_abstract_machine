from giskardpy.motion_statechart.tasks.pointing import Pointing

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    NavigateActionServerTask,
)
from semantic_digital_twin.robots.hsrb import HSRB
from ..datastructures.enums import ExecutionType
from ..view_manager import ViewManager
from ..robot_plans import MoveMotion, MoveTCPMotion, LookingMotion

from ..robot_plans.motions.base import AlternativeMotion


class HSRBMoveMotion(MoveMotion, AlternativeMotion[HSRB]):
    """
    Uses a Nav2 action server to move the base of the real HSRB
    """

    execution_type = ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self) -> NavigateActionServerTask:
        return NavigateActionServerTask(
            target_pose=self.target.to_spatial_type(),
            base_link=self.robot_view.root,
            action_topic="/hsrb/move_base",
            message_type=NavigateToPose,
        )
