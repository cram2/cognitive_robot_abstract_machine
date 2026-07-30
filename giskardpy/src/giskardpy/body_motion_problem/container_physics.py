from __future__ import annotations

from dataclasses import dataclass

from giskardpy.body_motion_problem.giskard_physics_model import GiskardPhysicsModel
from giskardpy.data_types.exceptions import HandleActuatorMismatchError
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.motion import MotionTrajectory
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ContainerManipulationPhysicsModel(GiskardPhysicsModel):
    """
    Physics model for opening or closing articulated containers (drawers, doors).

    Builds and runs a Giskard :class:`~giskardpy.motion_statechart.goals.open_close.Open`
    goal motion statechart internally, driving the container joint to
    :attr:`goal_joint_state`.
    """

    handle: Body
    """The handle body used as both the gripper tip and the environment link in the Open goal."""

    actuator: ActiveConnection1DOF
    """
    The revolute or prismatic joint being driven.
    """

    goal_joint_state: float
    """Target joint position to drive the container to."""

    def build_motion_statechart(self, effect: Effect, world: World) -> MotionStatechart:
        """
        Build a motion statechart with a single
        :class:`~giskardpy.motion_statechart.goals.open_close.Open` goal targeting
        :attr:`goal_joint_state`.

        :raises HandleActuatorMismatchError: If the handle's driving 1-DOF connection is
            not :attr:`actuator`, since the goal would then move a joint whose positions
            are never recorded.
        """
        handle_connection = self.handle.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        if handle_connection is not self.actuator:
            raise HandleActuatorMismatchError(
                handle_name=str(self.handle.name),
                actuator_name=str(self.actuator.name),
                handle_connection_name=str(handle_connection.name),
            )
        motion_statechart = MotionStatechart()
        goal = Open(
            tip_link=self.handle,
            environment_link=self.handle,
            goal_joint_state=self.goal_joint_state,
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        return motion_statechart

    def _build_motion_trajectory(self, effect: Effect) -> MotionTrajectory:
        """
        :return: Trajectory for the container joint driven by this model.
        """
        return MotionTrajectory(
            {
                self.actuator: self._extract_dof_positions(
                    self._recorded_trajectory, self.actuator
                )
            }
        )

    def interaction_body(self) -> Body:
        """
        :return: The handle body the robot physically interacts with.
        """
        return self.handle
