from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans import MoveMotion
from coraplex.robot_plans.motions.base import AlternativeMotion
from semantic_digital_twin.robots.tiago import Tiago


class TiagoMoveSim(MoveMotion, AlternativeMotion[Tiago]):
    """
    Uses a diff drive goal for the tiago base.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):

        return DifferentialDriveBaseGoal(
            goal_pose=self.target,
        )
