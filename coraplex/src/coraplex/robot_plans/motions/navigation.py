from dataclasses import dataclass

from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.executables import GiskardExecutable
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    Vector3,
)


@dataclass
class MoveMotion(BaseMotion):
    """
    Moves the robot to a designated location.
    """

    target: Pose
    """
    Location to which the robot should be moved
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return (
            SetOdometry(
                base_pose=self.target.to_homogeneous_matrix(),
                odom_connection=self.robot.root.parent_connection,
            )
            if GiskardExecutable.execution_type == ExecutionType.SIMULATED
            else CartesianPose(
                root_link=self.world.root,
                tip_link=self.robot.root,
                goal_pose=self.target,
            )
        )


@dataclass
class TurnMotion(BaseMotion):
    """
    Turns the robot's base on the spot until its front faces a target.

    The base keeps the position it has when the motion starts, so the turn is towards
    the target from wherever an earlier motion left it.
    """

    target: Pose
    """
    What the base's front turns towards; only its horizontal position matters.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self) -> Parallel:
        return Parallel(
            nodes=[
                Pointing(
                    root_link=self.world.root,
                    tip_link=self.robot.root,
                    goal_point=self._target_at_base_height(),
                    pointing_axis=Vector3(
                        *self.robot.mobile_base.forward_axis.to_np()[:3],
                        reference_frame=self.robot.root,
                    ),
                ),
                CartesianPosition(
                    root_link=self.world.root,
                    tip_link=self.robot.root,
                    goal_point=Point3(reference_frame=self.robot.root),
                ),
            ]
        )

    def _target_at_base_height(self) -> Point3:
        """
        :return: :attr:`target` moved vertically to the height of the base, which can
            only turn about the vertical and so can only point level.
        """
        root_P_target = self.world.transform(self.target, self.world.root).to_position()
        root_P_target.z = self.robot.root.global_pose.z
        return root_P_target
