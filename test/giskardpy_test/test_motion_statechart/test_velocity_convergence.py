"""
Tests for the velocity-convergence expression shared by ``EndMotion`` and the local-
minimum monitor.
"""

import pytest
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import velocity_convergence_expression
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from typing_extensions import List, Optional

#: thresholds the expression is built with; the values themselves do not matter here
CONVERGENCE_THRESHOLDS = {
    "joint_convergence_threshold": 0.01,
    "minimum_threshold": 0.001,
    "maximum_threshold": 0.1,
}


def add_degree_of_freedom(
    world: World, name: str, velocity_limit: Optional[float]
) -> DegreeOfFreedom:
    """
    Register a degree of freedom in a world, with or without an upper velocity limit.

    Registering it is what gives the degree of freedom its symbolic variables, so a
    standalone instance cannot be used here.

    :param world: The world to register the degree of freedom in.
    :param name: Name of the degree of freedom.
    :param velocity_limit: Upper velocity limit, or None for a degree of freedom parsed
        without one.
    """
    upper = DerivativeMap()
    upper.velocity = velocity_limit
    degree_of_freedom = DegreeOfFreedom(
        name=PrefixedName(name),
        limits=DegreeOfFreedomLimits(lower=DerivativeMap(), upper=upper),
    )
    with world.modify_world():
        world.add_degree_of_freedom(degree_of_freedom)
    return degree_of_freedom


@pytest.fixture()
def world() -> World:
    return World()


@pytest.fixture()
def context(world) -> MotionStatechartContext:
    """
    A context carrying the cycle counter the :class:`Executor` would otherwise install.
    """
    built = MotionStatechartContext(world=world)
    built.control_cycle_variable = FloatVariable("control_cycles")
    return built


def convergence_of(
    context: MotionStatechartContext, degrees_of_freedom: List[DegreeOfFreedom]
) -> str:
    """
    The convergence expression over the given degrees of freedom, as comparable text.

    :param context: The context the expression is built against.
    :param degrees_of_freedom: The degrees of freedom to check for convergence.
    """
    return str(
        velocity_convergence_expression(
            context, degrees_of_freedom=degrees_of_freedom, **CONVERGENCE_THRESHOLDS
        )
    )


class TestVelocityConvergenceExpression:
    def test_a_degree_of_freedom_without_a_velocity_limit_is_skipped(
        self, world, context
    ):
        """
        A joint without a velocity limit is excluded rather than failing the expression.

        There is no maximum velocity to derive a convergence threshold from, and
        environment URDFs routinely omit the limit.
        """
        limited = add_degree_of_freedom(world, "limited", velocity_limit=1.0)
        unlimited = add_degree_of_freedom(world, "unlimited", velocity_limit=None)

        assert convergence_of(context, [limited, unlimited]) == convergence_of(
            context, [limited]
        )

    def test_a_limited_degree_of_freedom_is_checked(self, world, context):
        """
        Skipping the unlimited ones must not skip the limited ones too.
        """
        limited = add_degree_of_freedom(world, "limited", velocity_limit=1.0)

        assert str(limited.variables.velocity) in convergence_of(context, [limited])
