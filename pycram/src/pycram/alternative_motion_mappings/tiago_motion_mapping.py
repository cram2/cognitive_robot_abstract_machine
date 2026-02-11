from giskardpy.motion_statechart.goals.cartesian_goals import DiffDriveBaseGoal
from pycram.datastructures.enums import ExecutionType
from pycram.robot_plans import MoveMotion
from pycram.robot_plans.motions.base import AlternativeMotion
from semantic_digital_twin.robots.tiago import Tiago


class StretchMoveSim(MoveMotion, AlternativeMotion[Tiago]):
    """
    Uses a diff drive goal for the tiago base.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):

        return DiffDriveBaseGoal(
            goal_pose=self.target.to_spatial_type(),
        )
