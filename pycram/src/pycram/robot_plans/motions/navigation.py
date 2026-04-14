from dataclasses import dataclass

from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from pycram.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class MoveMotion(BaseMotion):
    """
    Moves the robot to a designated location
    """

    target: Pose
    """
    Location to which the robot should be moved
    """

    keep_joint_states: bool = False
    """
    Keep the joint states of the robot during/at the end of the motion
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return CartesianPose(
            root_link=self.world.root,
            tip_link=self.robot.root,
            goal_pose=self.target,
        )
