from dataclasses import dataclass

from typing_extensions import Optional

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.motions.base import StandaloneMotion
from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager


@dataclass
class ContainerMotion(StandaloneMotion):
    """
    Base for the motions that drive a container's own degree of freedom while the hand
    holds its handle.
    """

    object_part: Body
    """
    Object designator for the drawer handle.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    goal_joint_state: Optional[float] = None
    """
    How far the container is driven, in the unit of its own degree of freedom.

    ``None`` drives it as far as that degree of freedom goes.
    """

    stall_time: float = 1.0
    """
    How long, in seconds, the container has to stop moving before it counts as stuck.

    Long enough that the slow approach onto a limit is not mistaken for having stopped.
    """

    def perform(self):
        return

    def _chart_around(self, goal: Open) -> MotionStatechartNode:
        """
        Wrap a container goal in what every container motion needs around it.

        :param goal: The goal driving the container's degree of freedom.
        :return: That goal, done as soon as the container arrives *or* stops moving,
            with the hand free to touch the handle it is holding.
        """
        return Parallel(
            [
                Parallel(
                    [goal, self._container_stalled()],
                    minimum_success=1,
                    name=goal.name,
                ),
                *self._only_allow_gripper_collision_rules(self.arm),
            ],
            name=goal.name,
        )

    def _container_stalled(self) -> LocalMinimumReached:
        """
        :return: A monitor that turns true once the container has stopped moving, which
            is as far as it goes when its own limit is only approached asymptotically.
        """
        connection = self.object_part.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        return LocalMinimumReached(
            degrees_of_freedom=[connection.raw_dof],
            minimum_time=self.stall_time,
            measure_from_own_start=True,
        )


@dataclass
class OpeningMotion(ContainerMotion):
    """
    Designator for opening container.
    """

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return self._chart_around(
            Open(
                tip_link=tip,
                environment_link=self.object_part,
                goal_joint_state=self.goal_joint_state,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                name="Open",
            )
        )


@dataclass
class ClosingMotion(ContainerMotion):
    """
    Designator for closing a container.
    """

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return self._chart_around(
            Close(
                tip_link=tip,
                environment_link=self.object_part,
                goal_joint_state=self.goal_joint_state,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                name="Close",
            )
        )
