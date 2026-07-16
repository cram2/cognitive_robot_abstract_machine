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
from krrood.entity_query_language.exceptions import MatchTypeCannotBeDetermined
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.query.match import Match
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
    query = a(Robot)(name="R2D2", battery=100).from_(robots)
    assert query.tolist()[0].name == "R2D2"


def test_match(handles_and_containers_world):
    world = handles_and_containers_world

    fixed_connection = a(FixedConnection)(
        parent=a(Container)(name="Container1"),
        child=a(Handle)(name="Handle1"),
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
    fixed_connection = a(FixedConnection)(
        parent=a(Container)(name="Container1"), child=a(Handle)(name="Handle1")
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
    fixed_connection = a(FixedConnection)(
        parent=a(Container),
        child=a(Handle),
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
    query = a(Robot)(name="R2D2", battery=100).from_(robots)
    assert isinstance(query, Match)
    assert query.domain is robots
    assert query.tolist()[0].name == "R2D2"


def test_expression_gives_symbolic_access(handles_and_containers_world):
    world = handles_and_containers_world
    fixed_connection = a(FixedConnection)(
        parent=a(Container)(name="Container1"), child=a(Handle)(name="Handle1")
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
    selected = a(Robot)().from_(subset).tolist()
    assert {robot.name for robot in selected} == {"R2D2", "C3PO"}


def test_from_without_kwargs_selects_all(handles_and_containers_world):
    world = handles_and_containers_world
    # an(Type)().from_(X) with no kwargs is a valid "any Type in X" select.
    selected = a(FixedConnection)().from_(world.connections).tolist()
    assert selected
    assert all(isinstance(connection, FixedConnection) for connection in selected)


def test_match_with_list():
    domain = [
        KRROODPositions([KRROODPosition(1, 2, 3), KRROODPosition(1, 2, 3)], ["a", "b"]),
        KRROODPositions([KRROODPosition(1, 2, 3)], ["a"]),
    ]

    q = a(KRROODPositions)(
        positions=[
            a(KRROODPosition)(
                x=1,
                y=2,
            ),
            KRROODPosition(1, 2, 3),
        ],
        some_strings=["a", "b"],
    ).from_(domain)

    r = q.tolist()
    assert r == [domain[0]]


# ── an()/the() with a callable factory: target_type inference and default ────


def test_an_infers_target_type_from_annotated_callable():
    def make_position(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> KRROODPosition:
        return KRROODPosition(x, y, z)

    match = an(make_position)
    assert match.type is KRROODPosition


def test_the_infers_target_type_from_annotated_callable():
    def make_position(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> KRROODPosition:
        return KRROODPosition(x, y, z)

    match = the(make_position)
    assert match.type is KRROODPosition


def test_an_uses_explicit_target_type_for_unannotated_callable():
    def make_position(x, y, z):
        return KRROODPosition(x, y, z)

    match = an(make_position, target_type=KRROODPosition)
    assert match.type is KRROODPosition


def test_an_raises_when_callable_type_cannot_be_determined():
    def make_position(x, y, z):
        return KRROODPosition(x, y, z)

    with pytest.raises(MatchTypeCannotBeDetermined):
        an(make_position)


def test_a_infers_target_type_from_annotated_callable():
    def make_position(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> KRROODPosition:
        return KRROODPosition(x, y, z)

    match = a(make_position)
    assert match.type is KRROODPosition


def test_a_uses_explicit_target_type_for_unannotated_callable():
    def make_position(x, y, z):
        return KRROODPosition(x, y, z)

    match = a(make_position, target_type=KRROODPosition)
    assert match.type is KRROODPosition


# ── Match.has_ellipsis_attributes ─────────────────────────────────────────────


def test_has_ellipsis_attributes_true_for_direct_ellipsis():
    match = a(KRROODPosition)(x=..., y=2, z=3)
    assert match.has_ellipsis_attributes is True


def test_has_ellipsis_attributes_false_without_ellipsis():
    match = a(KRROODPosition)(x=1, y=2, z=3)
    assert match.has_ellipsis_attributes is False


def test_has_ellipsis_attributes_true_for_nested_ellipsis():
    match = a(KRROODPositions)(
        positions=[a(KRROODPosition)(x=..., y=2, z=3)],
        some_strings=["a"],
    )
    assert match.has_ellipsis_attributes is True


def test_has_ellipsis_attributes_true_for_ellipsis_element_in_plain_list():
    """
    An ``...`` element sitting inside an otherwise-concrete list (no nested ``Match``
    elements) is still resolved as one literal-valued attribute match, but is just as
    underspecified as a direct ``x=...`` assignment.
    """
    match = a(KRROODPositions)(
        positions=[KRROODPosition(1, 2, 3)],
        some_strings=["a", ..., "c"],
    )
    assert match.has_ellipsis_attributes is True


def test_has_ellipsis_attributes_true_for_ellipsis_mixed_with_nested_match_in_list():
    match = a(KRROODPositions)(
        positions=[a(KRROODPosition)(x=1, y=2, z=3), ...],
        some_strings=["a", "b"],
    )
    assert match.has_ellipsis_attributes is True


def test_has_ellipsis_attributes_true_for_ellipsis_element_in_plain_set():
    match = a(KRROODPositions)(
        positions=[KRROODPosition(1, 2, 3)],
        some_strings={"a", ..., "c"},
    )
    assert match.has_ellipsis_attributes is True
