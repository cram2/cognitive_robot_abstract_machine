from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.weights import DefaultWeights, VelocityScale
from giskardpy.motion_statechart.exceptions import EmptyGoalStateError
from giskardpy.motion_statechart.graph_node import MotionNodeArtifacts, Task
from giskardpy.motion_statechart.graph_node import ConvergingTask
from giskardpy.motion_statechart.mixins import (
    HasCruiseVelocityParameter,
    HasGoalStateParameter,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)


@dataclass(eq=False, repr=False, kw_only=True)
class JointPositionList(
    ConvergingTask, HasGoalStateParameter, HasCruiseVelocityParameter
):
    """
    Moves the robot to a given joint position.

    Its error is the largest joint position error, in rad or m.
    """

    cruise_velocity: float = 2.0
    """
    How fast the joints approach the goal, in rad/s or m/s; not strictly enforced.

    A joint never moves faster than its velocity limit.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build one equality constraint per joint of the goal state.

        :return: The artifacts of this task, whose error is the largest absolute joint
            position error, so the task succeeds only once every joint is within the
            threshold.
        """
        if len(self.goal_state) == 0:
            raise EmptyGoalStateError(node=self)

        artifacts = MotionNodeArtifacts()
        errors = []
        for connection, target in self.goal_state.items():
            current = connection.dof.variables.position
            target = self.apply_limits_to_target(target, connection)
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = sm.shortest_angular_distance(current, target)
            else:
                error = target - current
            artifacts.constraints.add_equality_constraint(
                name=str(connection.name),
                cruise_velocity=self.cruise_velocity,
                velocity_scale=min(
                    connection.dof.limits.upper.velocity, VelocityScale.JOINT
                ),
                equality_bound=error,
                quadratic_weight=self.weight,
                task_expression=current,
            )
            errors.append(sm.abs(error))
        artifacts.error = sm.max(sm.Vector(errors))
        return artifacts

    def apply_limits_to_target(
        self, target: float, connection: ActiveConnection1DOF
    ) -> sm.Scalar:
        ul_pos = connection.dof.limits.upper.position
        ll_pos = connection.dof.limits.lower.position
        if ll_pos is not None:
            target = sm.limit(target, ll_pos, ul_pos)
        return target


@dataclass(eq=False, repr=False, kw_only=True)
class JointVelocityLimit(Task):
    """
    Limits the velocity of a set of joints to a maximum value.

    The joint-space equivalent of
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`:
    enforcement is performed by adding a real inequality constraint to the optimizer
    for each given joint, not by tuning a cruise velocity or a velocity scale.
    """

    connections: list[ActiveConnection1DOF]
    """
    The joints whose velocity is constrained.
    """

    max_joint_velocity: float = 0.1
    """
    Maximum allowed velocity (in rad/s or m/s, per joint).
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Optimization weight determining how strongly the velocity limit is enforced.

    Defaults to the same weight as its Cartesian equivalents
    (:class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`,
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianRotationVelocityLimit`):
    a velocity *limit* is a physical constraint like a joint limit, not a goal, so it
    is weighted above collision avoidance rather than below it like goal tasks (e.g.
    :class:`JointPositionList`).
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        artifacts = MotionNodeArtifacts()
        velocities = []
        for connection in self.connections:
            position = connection.dof.variables.position
            artifacts.constraints.add_velocity_constraint(
                upper_velocity_limit=self.max_joint_velocity,
                lower_velocity_limit=-self.max_joint_velocity,
                quadratic_weight=self.weight,
                task_expression=position,
                velocity_scale=connection.dof.limits.upper.velocity,
                name=str(connection.name),
            )
            velocities.append(connection.dof.variables.velocity)
        artifacts.observation = sm.logic_all(
            sm.abs(sm.Vector(velocities)) <= self.max_joint_velocity
        )
        return artifacts
