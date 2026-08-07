from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import List, Optional, Tuple

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
class ErrorSignal(ABC):
    """
    How far a task is from its goal, and how that distance can be watched over time.
    """

    expression: Scalar
    """
    The current error in the task's own units, where zero means the goal is reached.
    """

    @abstractmethod
    def create_rate_expression(self) -> Optional[Scalar]:
        """
        :return: The rate of change of :attr:`expression` in units per second, or None
            when the rate can only be measured by differencing across control cycles.
        """


@dataclass
class SymbolicErrorSignal(ErrorSignal):
    """
    An error that depends on the world only through the robot's joint positions, so its
    rate of change follows from the chain rule and the current joint velocities.

    The rate is exact, available on the first control cycle, and costs nothing extra to
    evaluate because it compiles into the observation state updater.

    .. warning:: Change caused by anything other than joint motion is invisible. An error
        whose goal is rewritten during execution, for example from perception, needs
        :class:`SampledErrorSignal`.
    """

    def create_rate_expression(self) -> Scalar:
        return time_derivative_from_joint_motion(self.expression)


@dataclass
class SampledErrorSignal(ErrorSignal):
    """
    An error whose rate of change is measured by differencing its value across control
    cycles.

    Applies to any error, including one that moves for reasons unrelated to the robot's
    motion, at the cost of a one cycle delay and sensitivity to noise.
    """

    def create_rate_expression(self) -> None:
        return None
