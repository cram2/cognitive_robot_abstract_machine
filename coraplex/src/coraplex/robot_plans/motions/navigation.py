from dataclasses import dataclass

from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.robot_plans.parameter_mixins import NavigationParameters
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class MoveMotion(BaseMotion, NavigationParameters):
    """
    Moves the robot to a designated location.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return (
            SetOdometry(
                base_pose=self.target_location.to_homogeneous_matrix(),
                odom_connection=self.robot.root.parent_connection,
            )
            if GiskardExecutable.execution_type == ExecutionType.SIMULATED
            else CartesianPose(
                root_link=self.world.root,
                tip_link=self.robot.root,
                goal_pose=self.target_location,
            )
        )
