from __future__ import annotations

from dataclasses import dataclass

from giskardpy.body_motion_problem.giskard_physics_model import GiskardPhysicsModel
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ContainerManipulationPhysicsModel(GiskardPhysicsModel):
    """
    Physics model for opening or closing articulated containers (drawers, doors).

    Builds and runs a Giskard :class:`~giskardpy.motion_statechart.goals.open_close.Open`
    goal MSC internally, driving the container joint to :attr:`goal_joint_state`.
    """

    handle: Body
    """The handle body used as both the gripper tip and the environment link in the Open goal."""

    actuator: ActiveConnection1DOF
    """The revolute or prismatic joint being driven."""

    goal_joint_state: float
    """Target joint position to drive the container to."""

    @property
    def primary_connection(self) -> ActiveConnection1DOF:
        return self.actuator

    def build_motion_statechart(self, effect: Effect, world: World) -> MotionStatechart:
        """
        Build an MSC with a single :class:`~giskardpy.motion_statechart.goals.open_close.Open`
        goal targeting :attr:`goal_joint_state`.
        """
        msc = MotionStatechart()
        goal = Open(
            tip_link=self.handle,
            environment_link=self.handle,
            goal_joint_state=self.goal_joint_state,
        )
        msc.add_node(goal)
        msc.add_node(EndMotion.when_true(goal))
        return msc

    def interaction_body(self):
        """
        :return: The handle body, so :class:`MotionStatechartCanPerform` tracks
                 the correct interaction point.
        """
        return self.handle
