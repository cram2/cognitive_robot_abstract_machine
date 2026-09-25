from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Tuple

from krrood.symbolic_math.symbolic_math import Scalar, SymbolicMathType
from semantic_digital_twin.world_description.degree_of_freedom import (
    PositionVariable,
    VelocityVariable,
)

# %% differentiating with respect to time


def joint_position_and_velocity_variables(
    expression: SymbolicMathType,
) -> Tuple[List[PositionVariable], List[VelocityVariable]]:
    """
    Collect the joint positions `expression` depends on, paired with the velocity of the
    same degree of freedom.

    Free variables that are not joint positions are skipped, so they are treated as
    constants when differentiating. That is correct for values which only change between
    runs of a node, such as a goal captured by
    :class:`~giskardpy.motion_statechart.binding_policy.ForwardKinematicsBinding`, and
    wrong for values rewritten every control cycle.

    :param expression: The expression to inspect.
    :return: The joint position variables and their matching velocity variables.
    """
    position_variables: List[PositionVariable] = [
        variable
        for variable in expression.free_variables()
        if isinstance(variable, PositionVariable)
    ]
    velocity_variables = [
        variable.dof.variables.velocity for variable in position_variables
    ]
    return position_variables, velocity_variables


def time_derivative_from_joint_motion(expression: Scalar) -> Scalar:
    """
    Differentiate `expression` with respect to time, assuming it changes only because
    the robot's joints move.

    :param expression: The scalar expression to differentiate.
    :return: The rate of change of `expression`, in its own units per second.
    """
    position_variables, velocity_variables = joint_position_and_velocity_variables(
        expression
    )
    if not position_variables:
        return Scalar(0)
    return expression.total_derivative(position_variables, velocity_variables)[0]


# %% error signals


@dataclass
class ErrorSignal:
    """
    How far a task is from its goal.
    """

    expression: Scalar
    """
    The current error in the task's own units, where zero means the goal is reached.
    """
