from dataclasses import dataclass

from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from pycram.datastructures.pose import PoseStamped
from pycram.robot_plans.motions.base import BaseMotion


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
