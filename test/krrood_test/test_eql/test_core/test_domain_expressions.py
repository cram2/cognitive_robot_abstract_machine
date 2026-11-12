"""
Tests for matches and variables whose domain is an expression: they range, for each
binding of the expression, over the elements of the value it evaluates to.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    a,
    exists,
    flat_variable,
    for_all,
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
from ...dataset.journeys import Journey, Route, Stop

# %% fixtures


@dataclass(kw_only=True)
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
    container = Container(name=f"container {identifier}")
    handle = Handle(name=f"handle {identifier}")
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
    """
    A match whose domain is an expression takes each element of the expression's value,
    for each binding of the expression, in that order.
    """
    world = variable(World, two_worlds.worlds)

    result = a(Connection).from_(domain=world.connections).tolist()

    assert result == [*two_worlds.first.connections, *two_worlds.second.connections]


def test_match_keeps_only_the_elements_of_its_type(two_worlds):
    """
    Elements of the domain expression that are not instances of the match's type are
    left out.
    """
    world = variable(World, two_worlds.worlds)

    result = a(FixedConnection).from_(domain=world.connections).tolist()

    assert result == [
        two_worlds.first.connections[0],
        two_worlds.second.connections[0],
    ]


def test_each_element_is_bound_with_the_binding_it_came_from(two_worlds):
    """
    Selected together with the variable its domain is read from, a match pairs each
    value of that variable with its own elements only.
    """
    world = variable(World, two_worlds.worlds)
    connection = a(FixedConnection).from_(domain=world.connections)

    result = set_of(world, connection).tolist()

    assert [(row[world], row[connection]) for row in result] == [
        (two_worlds.first, two_worlds.first.connections[0]),
        (two_worlds.second, two_worlds.second.connections[0]),
    ]


def test_match_selected_before_its_domain_binds_the_domain_expression(two_worlds):
    """
    The pairing holds when the match is selected before the variable its domain is read
    from.
    """
    world = variable(World, two_worlds.worlds)
    connection = a(FixedConnection).from_(domain=world.connections)

    result = set_of(connection, world).tolist()

    assert [(row[world], row[connection]) for row in result] == [
        (two_worlds.first, two_worlds.first.connections[0]),
        (two_worlds.second, two_worlds.second.connections[0]),
    ]


def test_domain_expression_can_be_filtered_by_its_elements(two_worlds):
    """
    A condition on the match's values filters the elements of its domain expression.
    """
    world = variable(World, two_worlds.worlds)
    body = a(Body).from_(domain=world.bodies)

    result = body.where(body.name == "handle 2").tolist()

    assert result == [two_worlds.second.bodies[1]]


def test_match_with_a_pattern_ranges_over_its_domain_expression(two_worlds):
    """
    A domain expression given after a pattern call ranges the match over its elements
    and keeps the pattern.
    """
    world = variable(World, two_worlds.worlds)

    result = a(Body)(name="handle 2").from_(domain=world.bodies).tolist()

    assert result == [two_worlds.second.bodies[1]]


def test_where_condition_before_a_domain_expression_constrains_the_match(
    two_worlds,
):
    """
    A condition recorded before the domain expression is given still constrains the
    match.
    """
    world = variable(World, two_worlds.worlds)
    body = a(Body)
    body.where(body.name == "handle 2")

    result = body.from_(domain=world.bodies).tolist()

    assert result == [two_worlds.second.bodies[1]]


# %% untyped domains


def test_variable_from_an_expression_ranges_over_its_elements(two_worlds):
    """
    An untyped variable over an expression takes each element of the expression's value.
    """
    world = variable(World, two_worlds.worlds)

    result = variable_from(world.bodies).tolist()

    assert result == [*two_worlds.first.bodies, *two_worlds.second.bodies]


def test_flattening_keeps_values_that_are_not_collections(two_worlds):
    """
    A value declared as one object, such as a string, is its only element rather than
    being iterated.
    """
    body = variable(Body, two_worlds.first.bodies)

    result = flat_variable(body.name).tolist()

    assert result == [live_body.name for live_body in two_worlds.first.bodies]


def test_variable_from_an_expression_is_a_variable_over_it(two_worlds):
    """
    An untyped variable over an expression is a variable, like a typed one.
    """
    world = variable(World, two_worlds.worlds)

    assert isinstance(variable_from(world.bodies), Variable)


def test_flat_variable_is_deprecated_for_variable_from(two_worlds):
    """
    Flattening warns that it is deprecated and gives what :func:`variable_from` gives.
    """
    world = variable(World, two_worlds.worlds)

    with pytest.deprecated_call():
        bodies = flat_variable(world.bodies)

    assert bodies.tolist() == variable_from(world.bodies).tolist()


def test_attribute_through_an_untyped_domain_reads_the_element_type(two_worlds):
    """
    An attribute of an untyped variable over a collection is read on its elements.
    """
    world = variable(World, two_worlds.worlds)

    result = variable_from(world.bodies).name.tolist()

    assert result == [
        body.name for world_ in two_worlds.worlds for body in world_.bodies
    ]


# %% values that are absent or not collections


@dataclass(kw_only=True)
class Journeys:
    """
    Journeys that cover a missing collection, an iterable single object and a value of
    undeclared type.
    """

    harbour: Stop
    """
    A stop of the route.
    """

    tower: Stop
    """
    A stop added to the route.
    """

    planned: Journey
    """
    A journey with an extra stop.
    """

    unplanned: Journey
    """
    A journey whose extra stops are ``None``.
    """

    @property
    def journeys(self) -> list[Journey]:
        """
        :return: Both journeys, the planned one first.
        """
        return [self.planned, self.unplanned]


@pytest.fixture
def journeys() -> Journeys:
    """
    :return: Two journeys along one route of one stop.
    """
    harbour, tower = Stop(name="harbour"), Stop(name="tower")
    route = Route(stops=[harbour])
    return Journeys(
        harbour=harbour,
        tower=tower,
        planned=Journey(name="planned", route=route, extra_stops=[tower]),
        unplanned=Journey(name="unplanned", route=route),
    )


def test_absent_collection_has_no_elements(journeys):
    """
    A collection that is ``None`` gives no element and drops the binding it came from.
    """
    journey = variable(Journey, journeys.journeys)
    extra_stop = variable_from(journey.extra_stops)

    result = set_of(journey, extra_stop).tolist()

    assert [(row[journey], row[extra_stop]) for row in result] == [
        (journeys.planned, journeys.tower)
    ]


def test_iterable_object_declared_as_one_value_is_its_only_element(journeys):
    """
    An object declared as one value is not iterated, even when its class can be.
    """
    journey = variable(Journey, journeys.journeys)

    result = variable_from(journey.route).tolist()

    assert result == [journeys.planned.route, journeys.unplanned.route]


def test_value_of_undeclared_type_is_read_as_its_runtime_kind(journeys):
    """
    A value whose kind is not declared gives its elements when it is a collection.
    """
    journey = variable(Journey, [journeys.planned])

    result = variable_from(journey.every_stop()).tolist()

    assert result == journeys.planned.every_stop()


# %% quantifiers over a match


def test_existential_quantifier_over_a_match_ranges_over_its_domain(journeys):
    """
    Some extra stop of a journey is the tower, for the journey whose stops are its own.
    """
    journey = a(Journey).from_(domain=journeys.journeys)
    stop = a(Stop).from_(domain=journey.extra_stops)

    result = journey.where(exists(stop, stop.name == journeys.tower.name)).tolist()

    assert result == [journeys.planned]


def test_universal_quantifier_over_a_match_ranges_over_its_domain(journeys):
    """
    Every extra stop of a journey is the harbour, vacuously so for a journey without
    extra stops.
    """
    journey = a(Journey).from_(domain=journeys.journeys)
    stop = a(Stop).from_(domain=journey.extra_stops)

    result = journey.where(for_all(stop, stop.name == journeys.harbour.name)).tolist()

    assert result == [journeys.unplanned]


def test_quantifier_over_a_match_states_its_pattern(journeys):
    """
    A quantifier over a match with a pattern states the pattern of each value.
    """
    journey = a(Journey).from_(domain=journeys.journeys)
    tower = a(Stop)(name=journeys.tower.name).from_(domain=journey.extra_stops)

    result = journey.where(exists(tower, True)).tolist()

    assert result == [journeys.planned]
