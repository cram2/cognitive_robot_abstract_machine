from copy import deepcopy

from giskardpy.motion_statechart.goals.cartesian_goals import DiffDriveBaseGoal
from giskardpy.motion_statechart.goals.open_close import Close
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.pointing import Pointing
from pycram.datastructures.enums import ExecutionType
from pycram.robot_plans import MoveTCPMotion, MoveMotion, ClosingMotion
from pycram.robot_plans.motions.base import AlternativeMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix


class StretchMoveTCP(MoveTCPMotion, AlternativeMotion[Stretch]):
    """
    Better motions for stretch to move the tool center point to the given goal, first rotates the base such that the
    gripper is pointing at the goal pose and then uses full body control to move the TCP to the goal.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self) -> Sequence:
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        goal_copy = deepcopy(self.target.to_spatial_type())
        goal_copy = self.world.transform(goal_copy, self.robot_view.root)
        goal_point = goal_copy.to_position()
        goal_point.z = 0
        return Sequence(
            [
                Pointing(
                    root_link=self.world.root,
                    tip_link=self.robot_view.root,
                    goal_point=goal_point,
                    pointing_axis=Vector3(
                        0, -1, 0, reference_frame=self.robot_view.root
                    ),
                ),
                CartesianPose(
                    root_link=self.world.root,
                    tip_link=tip,
                    goal_pose=self.target.to_spatial_type(),
                ),
            ]
        )


class StretchMoveSim(MoveMotion, AlternativeMotion[Stretch]):
    """
    Different giskard goal for moving stretch to a goal pose, this uses a goal optimal for a diff drive
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):

        return DiffDriveBaseGoal(
            goal_pose=self.target.to_spatial_type(),
        )


class StretchClose(ClosingMotion, AlternativeMotion[Stretch]):
    """
    Optimized close motion for the stretch robot. This puts the stretch directly in front of the container while holding
    the handle and then pushes it arm forward to close the container.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        cart = CartesianPose(
            name="Keep holding handle",
            root_link=self.object_part,
            tip_link=tip,
            goal_pose=HomogeneousTransformationMatrix(
                reference_frame=tip, child_frame=tip
            ),
        )
        align = AlignPlanes(
            root_link=self.world.root,
            tip_link=self.robot_view.root,
            goal_normal=Vector3(1, 0, 0, reference_frame=self.object_part),
            tip_normal=Vector3(0, -1, 0, self.robot_view.root),
        )
        close = Close(tip_link=tip, environment_link=self.object_part)
        return Parallel([cart, align, close])
