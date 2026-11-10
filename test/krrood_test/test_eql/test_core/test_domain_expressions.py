"""
Tests for matches and variables whose domain is an expression: they range, for each
binding of the expression, over the elements of the value it evaluates to.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import (
    a,
    flat_variable,
    set_of,
    variable,
    variable_from,
)

from ...dataset.semantic_world_like_classes import (
    Body,
    Container,
    Connection,
    FixedConnection,
    Handle,
    PrismaticConnection,
    World,
)

# %% fixtures


@dataclass
class TwoWorlds:
    """
    Two worlds, each with one fixed and one prismatic connection of its own.
    """

    first: World
    """
    The first world.
    """

    second: World
    """
    The second world.
    """

    @property
    def worlds(self) -> list[World]:
        """
        :return: Both worlds, the first one first.
        """
        return [self.first, self.second]


def world_with_connections(identifier: int) -> World:
    """
    :param identifier: The identifier of the world.
    :return: A world with a fixed and a prismatic connection between a container and a
        handle.
    """
    container = Container(f"container {identifier}")
    handle = Handle(f"handle {identifier}")
    return World(
        id=identifier,
        bodies=[container, handle],
        connections=[
            FixedConnection(parent=container, child=handle),
            PrismaticConnection(parent=container, child=handle),
        ],
    )


@pytest.fixture
def two_worlds() -> TwoWorlds:
    """
    :return: Two fresh worlds for each test.
    """
    return TwoWorlds(first=world_with_connections(1), second=world_with_connections(2))


# %% typed domains


def test_match_ranges_over_the_elements_of_its_domain_expression(two_worlds):
    world = variable(World, two_worlds.worlds)

    result = a(Connection).from_(world.connections).tolist()

    assert result == [*two_worlds.first.connections, *two_worlds.second.connections]


def test_match_keeps_only_the_elements_of_its_type(two_worlds):
    world = variable(World, two_worlds.worlds)

    result = a(FixedConnection).from_(world.connections).tolist()

    assert result == [
        two_worlds.first.connections[0],
        two_worlds.second.connections[0],
    ]


def test_each_element_is_bound_with_the_binding_it_came_from(two_worlds):
    world = variable(World, two_worlds.worlds)
    connection = a(FixedConnection).from_(world.connections)

    result = set_of(world, connection).tolist()

    assert [(row[world], row[connection]) for row in result] == [
        (two_worlds.first, two_worlds.first.connections[0]),
        (two_worlds.second, two_worlds.second.connections[0]),
    ]


def test_match_selected_before_its_domain_binds_the_domain_expression(two_worlds):
    world = variable(World, two_worlds.worlds)
    connection = a(FixedConnection).from_(world.connections)

    result = set_of(connection, world).tolist()

    assert [(row[world], row[connection]) for row in result] == [
        (two_worlds.first, two_worlds.first.connections[0]),
        (two_worlds.second, two_worlds.second.connections[0]),
    ]


def test_domain_expression_can_be_filtered_by_its_elements(two_worlds):
    world = variable(World, two_worlds.worlds)
    body = a(Body).from_(world.bodies)

    result = body.where(body.name == "handle 2").tolist()

    assert result == [two_worlds.second.bodies[1]]


def test_match_with_a_pattern_ranges_over_its_domain_expression(two_worlds):
    world = variable(World, two_worlds.worlds)

    result = a(Body)(name="handle 2").from_(world.bodies).tolist()

    assert result == [two_worlds.second.bodies[1]]


def test_where_condition_before_a_domain_expression_constrains_the_match(
    two_worlds,
):
    world = variable(World, two_worlds.worlds)
    body = a(Body)
    body.where(body.name == "handle 2")

    result = body.from_(world.bodies).tolist()

    assert result == [two_worlds.second.bodies[1]]


# %% untyped domains


def test_variable_from_an_expression_ranges_over_its_elements(two_worlds):
    world = variable(World, two_worlds.worlds)

    result = variable_from(world.bodies).tolist()

    assert result == [*two_worlds.first.bodies, *two_worlds.second.bodies]


def test_flattening_keeps_values_that_are_not_collections(two_worlds):
    body = variable(Body, two_worlds.first.bodies)

    result = flat_variable(body.name).tolist()

    assert result == [live_body.name for live_body in two_worlds.first.bodies]
