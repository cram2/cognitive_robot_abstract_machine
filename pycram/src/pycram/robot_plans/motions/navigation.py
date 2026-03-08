from dataclasses import dataclass

from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
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

    teleport: bool = False
    """
    If the robot should teleport to the target location instead of moving to it,
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        if self.teleport:
            return SetOdometry(base_pose=self.target.to_spatial_type())
        else:
            return CartesianPose(
                root_link=self.world.root,
                tip_link=self.robot_view.root,
                goal_pose=self.target.to_spatial_type(),
            )
