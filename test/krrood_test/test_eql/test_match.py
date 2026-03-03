import pytest

from krrood.entity_query_language.factories import entity, set_of, variable, the, match, match_variable
from krrood.entity_query_language.failures import NoKwargsInMatchVar
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.core.base_expressions import UnificationDict
from ..dataset.semantic_world_like_classes import (
    FixedConnection,
    Container,
    Handle,
)


def test_match(handles_and_containers_world):
    world = handles_and_containers_world

    fixed_connection = match_variable(FixedConnection, domain=world.connections)(
        parent=match(Container)(name="Container1"),
        child=match(Handle)(name="Handle1"),
    )
    fixed_connection_query = the(fixed_connection)

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
    fixed_connection = match_variable(FixedConnection, domain=world.connections)(
        parent=match(Container)(name="Container1"), child=match(Handle)(name="Handle1")
    )
    container_and_handle = the(
        set_of(container := fixed_connection.parent, handle := fixed_connection.child)
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
    fixed_connection = match_variable(FixedConnection, domain=world.connections)(
        parent=match(Container),
        child=match(Handle),
    )
    container_and_handle = the(
        set_of(
            container := fixed_connection.parent, handle := fixed_connection.child
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

    assert set(container_and_handle_2.tolist()[0].values()) == set(
        container_and_handle.tolist()[0].values()
    )

    answers = container_and_handle.tolist()[0]
    assert isinstance(answers, UnificationDict)
    assert answers[container].name == "Container3"
    assert answers[handle].name == "Handle3"


def test_empty_conditions_match_var(handles_and_containers_world):
    world = handles_and_containers_world
    with pytest.raises(NoKwargsInMatchVar):
        match_variable(FixedConnection, domain=world.connections)()
