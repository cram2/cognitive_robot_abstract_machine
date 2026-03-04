from dataclasses import dataclass

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.pointing import Pointing

from pycram.robot_plans.motions.base import BaseMotion
from pycram.datastructures.pose import PoseStamped


@dataclass
class MoveMotion(BaseMotion):
    """
    Moves the robot to a designated location
    """

    target: PoseStamped
    """
    Location to which the robot should be moved
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return SetOdometry(base_pose=self.target.to_spatial_type())

    @property
    def collision_rules(self) -> list[MotionStatechartNode]:
        return []
