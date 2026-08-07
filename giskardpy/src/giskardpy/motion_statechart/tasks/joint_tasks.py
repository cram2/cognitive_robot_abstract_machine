from __future__ import annotations

from dataclasses import field, dataclass
from typing_extensions import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import EmptyGoalStateError
from giskardpy.motion_statechart.error_signals import SymbolicErrorSignal
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.graph_node import ConvergingTask
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
)


@dataclass(eq=False, repr=False)
class JointPositionList(ConvergingTask):
    """
    Moves the robot to a given joint position.
    """

    goal_state: JointState = field(kw_only=True)
    """
    The goal joint state.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    If all joint position errors are smaller than this threshold, the task's observation
    state is true.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    The weight of this task.
    """

    max_velocity: float = field(default=1.0, kw_only=True)
    """
    The maximum velocity of the joints.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build one equality constraint per joint of the goal state.

        :param context: Provides access to world model and kinematic expressions.
        :return: The artifacts of this task, whose error is the largest absolute joint
            position error, so the task succeeds only once every joint is within the
            threshold.
        """
        if len(self.goal_state) == 0:
            raise EmptyGoalStateError(node=self)

        artifacts = NodeArtifacts()
        errors = []
        for connection, target in self.goal_state.items():
            current = connection.dof.variables.position
            target = self.apply_limits_to_target(target, connection)
            velocity = self.apply_limits_to_velocity(self.max_velocity, connection)
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = sm.shortest_angular_distance(current, target)
            else:
                error = target - current
            artifacts.constraints.add_equality_constraint(
                name=str(connection.name),
                reference_velocity=velocity,
                equality_bound=error,
                quadratic_weight=self.weight,
                task_expression=current,
            )
            errors.append(sm.abs(error))
        artifacts.error = SymbolicErrorSignal(sm.max(sm.Vector(errors)))
        return artifacts

    def apply_limits_to_target(
        self, target: float, connection: ActiveConnection1DOF
    ) -> sm.Scalar:
        ul_pos = connection.dof.limits.upper.position
        ll_pos = connection.dof.limits.lower.position
        if ll_pos is not None:
            target = sm.limit(target, ll_pos, ul_pos)
        return target

    def apply_limits_to_velocity(
        self, velocity: float, connection: ActiveConnection1DOF
    ) -> sm.Scalar:
        ul_vel = connection.dof.limits.upper.velocity
        ll_vel = connection.dof.limits.lower.velocity
        return sm.limit(velocity, ll_vel, ul_vel)


@dataclass(eq=False, repr=False)
class JointVelocityLimit(Task):
    """
    Limits the velocity of a set of joints to a maximum value.

    The joint-space equivalent of
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`:
    enforcement is performed by adding a real inequality constraint to the optimizer
    for each given joint, not by tuning a reference/normalization velocity.
    """

    connections: List[ActiveConnection1DOF] = field(kw_only=True)
    """
    The joints whose velocity is constrained.
    """

    max_velocity: float = field(default=0.1, kw_only=True)
    """
    Maximum allowed velocity (in rad/s or m/s, per joint).
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Optimization weight determining how strongly the velocity limit is enforced.

    Defaults to the same weight as its Cartesian equivalents
    (:class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`,
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianRotationVelocityLimit`):
    a velocity *limit* is a physical constraint like a joint limit, not a goal, so it
    is weighted above collision avoidance rather than below it like goal tasks (e.g.
    :class:`JointPositionList`).
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        velocities = []
        for connection in self.connections:
            position = connection.dof.variables.position
            artifacts.constraints.add_velocity_constraint(
                upper_velocity_limit=self.max_velocity,
                lower_velocity_limit=-self.max_velocity,
                quadratic_weight=self.weight,
                task_expression=position,
                velocity_limit=self.max_velocity,
                name=str(connection.name),
            )
            velocities.append(connection.dof.variables.velocity)
        artifacts.observation = sm.logic_all(
            sm.abs(sm.Vector(velocities)) <= self.max_velocity
        )
        return artifacts
