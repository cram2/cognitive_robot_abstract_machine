"""
Tests for the objects a :class:`~krrood.planning.state.State` adds and removes: a match
without a domain ranges over the live symbol graph's instances less the objects the
state removes plus those it adds, and a world's collection reads through the state
without the objects it removes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import a, literal, variable, variable_from
from krrood.planning.state import State

from ..dataset.objects_created_for_a_state import PlannedFixedConnection
from ..dataset.semantic_world_like_classes import (
    Connection,
    FixedConnection,
    World,
)

# %% fixtures


@dataclass(kw_only=True)
class PlannedConnectionWorld:
    """
    A world of handles and containers, and a fixed connection planned between two of its
    bodies that the live world does not have.
    """

    world: World
    """
    The live world.
    """

    planned_connection: PlannedFixedConnection
    """
    The connection a state adds.
    """

    @property
    def fixed_connections(self) -> list[FixedConnection]:
        """
        :return: The live world's fixed connections, in the order the world holds them.
        """
        return [
            connection
            for connection in self.world.connections
            if isinstance(connection, FixedConnection)
        ]


@pytest.fixture
def planned_connection_world(handles_and_containers_world) -> PlannedConnectionWorld:
    """
    :return: The handles and containers world, with a connection planned between its
        first two bodies.
    """
    first_body, second_body = handles_and_containers_world.bodies[:2]
    return PlannedConnectionWorld(
        world=handles_and_containers_world,
        planned_connection=PlannedFixedConnection(parent=first_body, child=second_body),
    )


def fixed_connections_in(state: State | None = None) -> list[FixedConnection]:
    """
    :param state: The state to evaluate in, or ``None`` for the live objects.
    :return: Every fixed connection, as a match without a domain ranges over them.
    """
    return a(FixedConnection).tolist(state=state)


# %% default domains


def test_match_without_domain_ranges_over_objects_the_state_adds(
    planned_connection_world,
):
    """
    An object the live symbol graph never holds is an instance of its type in a state
    that adds it, after the live instances.
    """
    state = State().with_objects_added(planned_connection_world.planned_connection)

    assert fixed_connections_in(state) == [
        *planned_connection_world.fixed_connections,
        planned_connection_world.planned_connection,
    ]


def test_match_without_domain_skips_objects_the_state_removes(
    planned_connection_world,
):
    """
    A live instance the state removes is not a value of a match without a domain.
    """
    removed, *kept = planned_connection_world.fixed_connections
    state = State().with_objects_removed(removed)

    assert fixed_connections_in(state) == kept


def test_match_without_domain_ranges_over_objects_added_and_removed_together(
    planned_connection_world,
):
    """
    A state that removes one object and adds another gives the live instances less the
    removed one, followed by the added one.
    """
    removed, *kept = planned_connection_world.fixed_connections
    state = (
        State()
        .with_objects_removed(removed)
        .with_objects_added(planned_connection_world.planned_connection)
    )

    result = a(FixedConnection).tolist(state=state)

    assert result == [*kept, planned_connection_world.planned_connection]


def test_one_query_reads_the_objects_of_each_state_it_is_evaluated_with(
    planned_connection_world,
):
    """
    The domain of a query is read again for each evaluation, so evaluating it without a
    state after a state that removed an object gives that object back.
    """
    removed, *kept = planned_connection_world.fixed_connections
    query = a(FixedConnection)

    assert query.tolist(state=State().with_objects_removed(removed)) == kept
    assert query.tolist() == planned_connection_world.fixed_connections


def test_objects_of_another_type_are_not_in_the_domain(planned_connection_world):
    """
    An added object is a value only of a match whose type it is an instance of.
    """
    state = State().with_objects_added(planned_connection_world.planned_connection)

    result = a(World).tolist(state=state)

    assert result == [planned_connection_world.world]


# %% the live objects stay unchanged


def test_state_objects_never_change_the_live_objects(planned_connection_world):
    """
    Building a state that adds and removes objects leaves the live instances as they
    were.
    """
    removed = planned_connection_world.fixed_connections[0]
    State().with_objects_removed(removed).with_objects_added(
        planned_connection_world.planned_connection
    )

    assert fixed_connections_in() == planned_connection_world.fixed_connections


# %% explicit domains


def test_removed_object_leaves_the_collections_that_hold_it(
    planned_connection_world,
):
    """
    A live collection read through a state leaves out the objects the state removes.
    """
    removed, *kept = planned_connection_world.fixed_connections
    state = State().with_objects_removed(removed)

    result = (
        a(FixedConnection)
        .from_(domain=literal(planned_connection_world.world).connections)
        .tolist(state=state)
    )

    assert result == kept


def test_removed_object_leaves_the_collections_the_state_assigns(
    planned_connection_world,
):
    """
    A collection a fact assigns is read without the objects the state removes, even when
    the assigned value holds them.
    """
    world = planned_connection_world.world
    removed, *kept = planned_connection_world.fixed_connections
    planned = planned_connection_world.planned_connection
    state = (
        State()
        .with_objects_added(planned)
        .with_facts(literal(world).connections == [*world.connections, planned])
        .with_objects_removed(removed)
    )

    result = (
        a(FixedConnection).from_(domain=literal(world).connections).tolist(state=state)
    )

    assert result == [*kept, planned]


def test_removed_object_leaves_a_captured_list(planned_connection_world):
    """
    A list given as a domain when the query is built is read without the objects the
    state removes.
    """
    removed, *kept = planned_connection_world.fixed_connections
    state = State().with_objects_removed(removed)

    result = (
        a(FixedConnection)
        .from_(domain=planned_connection_world.fixed_connections)
        .tolist(state=state)
    )

    assert result == kept


def test_removed_object_leaves_an_untyped_domain(planned_connection_world):
    """
    A variable given no type reads its given domain without the objects the state
    removes.
    """
    removed, *kept = planned_connection_world.fixed_connections
    state = State().with_objects_removed(removed)

    result = variable_from(planned_connection_world.fixed_connections).tolist(
        state=state
    )

    assert result == kept


def test_literal_still_names_a_removed_object(planned_connection_world):
    """
    A literal names its object rather than ranging over a domain, so a state that
    removes the object leaves the literal's value unchanged.
    """
    removed = planned_connection_world.fixed_connections[0]
    state = State().with_objects_removed(removed)

    result = literal(removed).tolist(state=state)

    assert result == [removed]


def test_world_collection_reads_the_collection_the_state_assigns(
    planned_connection_world,
):
    """
    A collection reached through a variable reads the value a fact assigns, which is how
    an added object enters it.
    """
    world = planned_connection_world.world
    planned = planned_connection_world.planned_connection
    state = (
        State()
        .with_objects_added(planned)
        .with_facts(literal(world).connections == [*world.connections, planned])
    )

    result = (
        a(FixedConnection)
        .from_(domain=variable(World, [world]).connections)
        .tolist(state=state)
    )

    assert result == [*planned_connection_world.fixed_connections, planned]


# %% building states


def test_adding_a_removed_object_restores_it(planned_connection_world):
    """
    Adding an object the state removed makes it a live instance of the state again, in
    its live position.
    """
    removed = planned_connection_world.fixed_connections[0]
    state = State().with_objects_removed(removed).with_objects_added(removed)

    assert fixed_connections_in(state) == planned_connection_world.fixed_connections


def test_removing_an_added_object_drops_it(planned_connection_world):
    """
    Removing an object the state added leaves only the live instances.
    """
    planned = planned_connection_world.planned_connection
    state = State().with_objects_added(planned).with_objects_removed(planned)

    assert fixed_connections_in(state) == planned_connection_world.fixed_connections


def test_adding_an_object_twice_adds_it_once(planned_connection_world):
    """
    Objects are added by identity, so adding the same object again leaves the state's
    objects as they were.
    """
    planned = planned_connection_world.planned_connection
    state = State().with_objects_added(planned).with_objects_added(planned)

    assert fixed_connections_in(state) == [
        *planned_connection_world.fixed_connections,
        planned,
    ]


def test_adding_a_live_object_leaves_it_where_it_is(planned_connection_world):
    """
    Adding an object the live symbol graph already holds neither repeats nor reorders
    it.
    """
    live = planned_connection_world.fixed_connections[0]
    state = State().with_objects_added(live)

    assert fixed_connections_in(state) == planned_connection_world.fixed_connections


# %% comparing states


def test_states_with_the_same_objects_are_equal_and_hash_alike(
    planned_connection_world,
):
    """
    Two states built from the same additions are equal and have the same hash.
    """
    planned = planned_connection_world.planned_connection
    first = State().with_objects_added(planned)
    second = State().with_objects_added(planned)

    assert first == second
    assert hash(first) == hash(second)


def test_states_with_different_objects_differ(planned_connection_world):
    """
    A state that removes an object differs from the empty state.
    """
    removed = planned_connection_world.fixed_connections[0]

    assert State().with_objects_removed(removed) != State()


def test_states_with_equal_but_distinct_objects_differ(planned_connection_world):
    """
    Objects are compared by identity, so adding an equal copy gives a different state.
    """
    planned = planned_connection_world.planned_connection
    equal_copy = PlannedFixedConnection(parent=planned.parent, child=planned.child)

    assert State().with_objects_added(planned) != State().with_objects_added(equal_copy)


def test_connections_of_every_kind_follow_the_state(planned_connection_world):
    """
    A match on a base class leaves out a removed instance of any of its subclasses.
    """
    live = a(Connection).tolist()
    removed = live[0]
    state = State().with_objects_removed(removed)

    assert a(Connection).tolist(state=state) == live[1:]
