from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import (
    entity,
    set_of,
    variable,
    the,
    an,
    a,
)
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.query.match import Match, is_underspecified
from krrood.entity_query_language.core.base_expressions import UnificationDict
from krrood.parametrization.random_events_translator import is_literal_comparator
from ..dataset.example_classes import KRROODPositions, KRROODPosition
from ..dataset.semantic_world_like_classes import (
    FixedConnection,
    Container,
    Handle,
)


def test_doc_match():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str
        battery: int

    robots = [Robot("R2D2", 100), Robot("C3PO", 0)]
    query = an(Robot)(name="R2D2", battery=100).from_(robots)
    assert query.tolist()[0].name == "R2D2"


def test_match(handles_and_containers_world):
    world = handles_and_containers_world

    fixed_connection = an(FixedConnection)(
        parent=an(Container)(name="Container1"),
        child=an(Handle)(name="Handle1"),
    ).from_(world.connections)
    fixed_connection_query = the(fixed_connection.expression)

    fc = variable(FixedConnection, domain=None)
    fixed_connection_query_manual = the(
        entity(fc).where(
            HasType(fc.parent, Container),
            HasType(fc.child, Handle),
            fc.parent.name == "Container1",
            fc.child.name == "Handle1",
        )
    )

    fixed_connection_match_result = fixed_connection_query.tolist()[0]
    fixed_connection_manual_result = fixed_connection_query_manual.tolist()[0]
    assert fixed_connection_match_result == fixed_connection_manual_result
    assert fixed_connection.first() == fixed_connection_manual_result
    assert isinstance(fixed_connection_match_result, FixedConnection)
    assert fixed_connection_match_result.parent.name == "Container1"
    assert isinstance(fixed_connection_match_result.child, Handle)
    assert fixed_connection_match_result.child.name == "Handle1"


def test_select(handles_and_containers_world):
    world = handles_and_containers_world

    # Method 1
    fixed_connection = an(FixedConnection)(
        parent=an(Container)(name="Container1"), child=an(Handle)(name="Handle1")
    ).from_(world.connections)
    container_and_handle = the(
        set_of(
            container := fixed_connection.expression.parent,
            handle := fixed_connection.expression.child,
        )
    )

    # Method 2
    fixed_connection_2 = variable(FixedConnection, domain=world.connections)
    container_and_handle_2 = the(
        set_of(
            container_2 := fixed_connection_2.parent,
            handle_2 := fixed_connection_2.child,
        ).where(
            HasType(container_2, Container),
            HasType(handle_2, Handle),
            container_2.name == "Container1",
            handle_2.name == "Handle1",
        )
    )

    assert set(container_and_handle_2.tolist()[0].values()) == set(
        container_and_handle.tolist()[0].values()
    )

    answers = container_and_handle.tolist()[0]
    assert isinstance(answers, UnificationDict)
    assert answers[container].name == "Container1"
    assert answers[handle].name == "Handle1"


def test_select_where(handles_and_containers_world):
    world = handles_and_containers_world

    # Method 1
    fixed_connection = an(FixedConnection)(
        parent=an(Container),
        child=an(Handle),
    ).from_(world.connections)
    container_and_handle = a(
        set_of(
            container := fixed_connection.expression.parent,
            handle := fixed_connection.expression.child,
        ).where(container.size > 1)
    )
    # Method 2
    fixed_connection_2 = variable(FixedConnection, domain=world.connections)
    container_and_handle_2 = the(
        set_of(
            container_2 := fixed_connection_2.parent,
            handle_2 := fixed_connection_2.child,
        ).where(
            HasType(container_2, Container),
            HasType(handle_2, Handle),
            container_2.size > 1,
        )
    )

    assert set(
        map(lambda x: tuple(x.values()), container_and_handle_2.tolist())
    ) == set(map(lambda x: tuple(x.values()), container_and_handle.tolist()))

    answers = container_and_handle.tolist()
    assert len(answers) == 1
    assert isinstance(answers[0], UnificationDict)
    assert answers[0][container].name == "Container3"
    assert answers[0][handle].name == "Handle3"


def test_domain_carrying_an_is_a_select():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str
        battery: int

    robots = [Robot("R2D2", 100), Robot("C3PO", 0)]
    # an(...)(kwargs).from_(domain) stays one Match carrying the domain; the backend decides
    # whether to select over it or generate from it, so from_ does not collapse it to an Entity.
    query = an(Robot)(name="R2D2", battery=100).from_(robots)
    assert isinstance(query, Match)
    assert query.domain is robots
    assert query.tolist()[0].name == "R2D2"


def test_expression_gives_symbolic_access(handles_and_containers_world):
    world = handles_and_containers_world
    fixed_connection = an(FixedConnection)(
        parent=an(Container)(name="Container1"), child=an(Handle)(name="Handle1")
    ).from_(world.connections)
    # .expression lowers the domain match to its selection query, whose .parent / .child read
    # the matched object's attributes symbolically and carry the match's conditions.
    container_and_handle = the(
        set_of(
            container := fixed_connection.expression.parent,
            handle := fixed_connection.expression.child,
        )
    )
    answers = container_and_handle.tolist()[0]
    assert answers[container].name == "Container1"
    assert answers[handle].name == "Handle1"


def test_from_restricts_the_search():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str

    subset = [Robot("R2D2"), Robot("C3PO")]
    # from_ selects only over the given domain.
    selected = an(Robot)().from_(subset).tolist()
    assert {robot.name for robot in selected} == {"R2D2", "C3PO"}


def test_is_underspecified_detects_deferred_match():
    @dataclass(unsafe_hash=True)
    class Robot:
        name: str
        battery: int

    robots = [Robot("R2D2", 100)]
    # is_underspecified is a structural check: any Match is a deferred query a backend must
    # resolve (whether it selects or generates over the domain is the backend's concern); a
    # concrete instance is not.
    assert is_underspecified(an(Robot)(name="R2D2")) is True
    assert is_underspecified(an(Robot)(name="R2D2").from_(robots)) is True
    assert is_underspecified(Robot("R2D2", 100)) is False


def test_from_without_kwargs_selects_all(handles_and_containers_world):
    world = handles_and_containers_world
    # an(Type)().from_(X) with no kwargs is a valid "any Type in X" select.
    selected = an(FixedConnection)().from_(world.connections).tolist()
    assert selected
    assert all(isinstance(connection, FixedConnection) for connection in selected)


def test_match_without_domain_selects_from_symbol_graph():
    """
    A domain-less match evaluated standalone (default selective backend) *selects* from the
    SymbolGraph for ``Symbol`` types: it returns the existing registered instance rather than
    constructing a new one. Generation requires an explicit generative backend.
    """
    existing = KRROODPosition(1.0, 2.0, 3.0)
    result = an(KRROODPosition)(x=1.0, y=2.0, z=3.0).tolist()
    # the existing object itself is returned (selection), not a freshly-built equal one
    assert any(r is existing for r in result)
    assert all(isinstance(r, KRROODPosition) and r == existing for r in result)


def test_match_with_list():
    domain = [
        KRROODPositions([KRROODPosition(1, 2, 3), KRROODPosition(1, 2, 3)], ["a", "b"]),
        KRROODPositions([KRROODPosition(1, 2, 3)], ["a"]),
    ]

    q = an(KRROODPositions)(
        positions=[
            an(KRROODPosition)(
                x=1,
                y=2,
            ),
            KRROODPosition(1, 2, 3),
        ],
        some_strings=["a", "b"],
    ).from_(domain)

    r = q.tolist()
    assert r == [domain[0]]
