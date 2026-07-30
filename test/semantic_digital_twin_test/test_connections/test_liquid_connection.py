from copy import deepcopy

import pytest

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MissingFillLevelLimitsError
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    GatedArticulatedPouringEquation,
    InflowEquation,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import LiquidConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% builders


def build_world_with_liquid_connection(
    dof_limits: DegreeOfFreedomLimits | None = None,
) -> tuple[World, LiquidConnection]:
    """
    Build a world containing a container body with a fill-level connection.
    """
    world = World()
    container = Body(name=PrefixedName("container"))
    phantom = Body(name=PrefixedName("container_fill_level_phantom"))
    with world.modify_world():
        world.add_body(container)
    with world.modify_world():
        connection = LiquidConnection.create_with_dofs(
            world=world,
            parent=container,
            child=phantom,
            axis=Vector3.Z(),
            dof_limits=dof_limits,
        )
        world.add_connection(connection)
    return world, connection


def build_fill_level_limits() -> DegreeOfFreedomLimits:
    """
    The [0, 1] fill-level limits a fill DOF is normally created with.
    """
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(position=0.0, velocity=-1.0),
        upper=DerivativeMap(position=1.0, velocity=1.0),
    )


# %% passive fill DOF


def test_fill_dof_is_passive():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    assert connection.active_dofs == []
    assert connection.passive_dofs == [connection.raw_dof]
    assert connection.raw_dof in world.passive_degrees_of_freedom
    assert connection.raw_dof not in world.active_degrees_of_freedom
    assert connection.is_controlled is False


# %% clamping contract


def test_update_state_without_position_limits_raises():
    _, connection = build_world_with_liquid_connection(dof_limits=None)
    with pytest.raises(MissingFillLevelLimitsError) as error_info:
        connection.update_state(0.05)
    assert error_info.value.connection_name == connection.name


# %% serialization round trip


def test_json_round_trip_restores_ungated_outflow():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.outflow_equation = GatedArticulatedPouringEquation(
        container_height=0.2,
        container_width=0.1,
        outflow_rate_constant=2.0,
        gate=sm.Scalar(0.5),
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored = LiquidConnection.from_json(
        connection.to_json(), **tracker.create_kwargs()
    )

    assert type(restored) is LiquidConnection
    assert type(restored.outflow_equation) is ArticulatedPouringEquation
    assert restored.outflow_equation.container_height == pytest.approx(0.2)
    assert restored.outflow_equation.container_width == pytest.approx(0.1)
    assert restored.outflow_equation.outflow_rate_constant == pytest.approx(2.0)
    assert restored.inflow_equation is None
    assert restored.raw_dof.id == connection.raw_dof.id


def test_json_round_trip_without_outflow_equation():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored = LiquidConnection.from_json(
        connection.to_json(), **tracker.create_kwargs()
    )
    assert restored.outflow_equation is None
    assert restored.inflow_equation is None


# %% copy_for_world


def test_copy_for_world_preserves_equations():
    world, connection = build_world_with_liquid_connection(build_fill_level_limits())
    connection.outflow_equation = ArticulatedPouringEquation(
        container_height=0.2, container_width=0.1
    )
    connection.inflow_equation = InflowEquation(
        container_height=0.2, container_width=0.1, inflow=sm.Scalar(0.005)
    )
    world_copy = deepcopy(world)

    copied_connection = connection.copy_for_world(world_copy)

    assert type(copied_connection) is LiquidConnection
    assert copied_connection.outflow_equation is connection.outflow_equation
    assert copied_connection.inflow_equation is connection.inflow_equation
    assert copied_connection.parent is world_copy.get_kinematic_structure_entity_by_id(
        connection.parent.id
    )
    assert copied_connection.child is world_copy.get_kinematic_structure_entity_by_id(
        connection.child.id
    )
    assert copied_connection.raw_dof is world_copy.get_degree_of_freedom_by_id(
        connection.raw_dof.id
    )
