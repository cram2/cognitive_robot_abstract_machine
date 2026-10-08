"""
Native world access, collection presets and query result identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import an, entity, flat_variable, variable
from cramera.knowledge.presets import Preset
from cramera.knowledge.query_runner import EqlQueryRunner, RowRenderer
from cramera.knowledge.queryable_knowledge import QueryScope
from cramera.live.bridge import Bridge
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Table,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


# %% native world fixtures
@pytest.fixture()
def annotated_robot_world(pr2_world_copy: World) -> World:
    """
    Add a table and handle to an isolated robot world.

    :param pr2_world_copy: The robot scene receiving semantic annotations.
    :return: The scene containing robot parts, a supporting surface and a handle.
    """
    table = Body(name=PrefixedName("query_table"))
    handle = Body(name=PrefixedName("query_handle"))
    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            FixedConnection(parent=pr2_world_copy.root, child=table)
        )
        pr2_world_copy.add_connection(FixedConnection(parent=table, child=handle))
        pr2_world_copy.add_semantic_annotation(Table(root=table))
        pr2_world_copy.add_semantic_annotation(Handle(root=handle))
    return pr2_world_copy


# %% complete native world access
def test_world_queries_expose_the_native_world_without_copied_domains(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    An attached world is exposed directly without a catalog of duplicate domains.

    :param world_with_two_bodies: The empty native world being queried.
    """
    world, _, _ = world_with_two_bodies
    bridge = Bridge()
    bridge.attach(world)

    vocabulary = bridge.query_vocabulary()

    assert vocabulary.domains == []
    assert vocabulary.extra_names == {World.__name__.lower(): world}
    assert vocabulary.extra_names[World.__name__.lower()] is world
    assert bridge.query_scopes() == [QueryScope.CURRENT_STATE]


def test_native_connections_and_degrees_of_freedom_are_queryable(
    annotated_robot_world: World,
) -> None:
    """
    Queries reach native connections, their relations and degrees of freedom.

    :param annotated_robot_world: The robot whose complete model is queried.
    """
    bridge = Bridge()
    bridge.attach(annotated_robot_world)
    world = bridge.query_vocabulary().extra_names[World.__name__.lower()]
    world_variable = variable(World, domain=[world])
    connection = flat_variable(world_variable.connections)
    degree_of_freedom = flat_variable(world_variable.degrees_of_freedom)

    connections = bridge.run_query(an(entity(connection)))
    parents = bridge.run_query(an(entity(connection.parent)))
    degrees = bridge.run_query(an(entity(degree_of_freedom)))

    assert [row["__entity__"] for row in connections.rows] == [
        str(value.name) for value in world.connections
    ]
    assert [row["__entity__"] for row in parents.rows] == [
        str(value.parent.name) for value in world.connections
    ]
    assert [row["__entity__"] for row in degrees.rows] == [
        str(value.name) for value in world.degrees_of_freedom
    ]


@dataclass(eq=False)
class TaggedHandle(Handle):
    """
    A custom semantic annotation carrying domain-specific inspection information.
    """

    inspection_tag: str = ""
    """
    The inspection information attached to this handle.
    """


def test_custom_annotation_properties_remain_queryable_without_registration(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    Custom semantic fields remain accessible without a predefined entity domain.

    :param world_with_two_bodies: The world and bodies receiving a custom annotation.
    """
    world, parent, child = world_with_two_bodies
    handle = TaggedHandle(root=child, inspection_tag="inspection pending")
    with world.modify_world():
        world.add_connection(FixedConnection(parent=parent, child=child))
        world.add_semantic_annotation(handle)
    bridge = Bridge()
    bridge.attach(world)
    annotation = flat_variable(variable(World, domain=[world]).semantic_annotations)

    answer = bridge.run_query(an(entity(annotation.inspection_tag)))

    assert answer.rows == [{"value": handle.inspection_tag}]
    assert bridge.query_vocabulary().extra_names[
        World.__name__.lower()
    ].semantic_annotations == [handle]


# %% current native state
def test_queries_follow_body_and_annotation_additions_and_removals(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    Native world queries observe later model changes without rebuilding domains.

    :param world_with_two_bodies: The world and bodies added before removing a handle.
    """
    world, parent, child = world_with_two_bodies
    bridge = Bridge()
    bridge.attach(world)
    native_world = variable(World, domain=[world])
    bodies = an(entity(flat_variable(native_world.bodies)))
    annotations = an(entity(flat_variable(native_world.semantic_annotations)))

    assert bridge.run_query(bodies).rows == []
    assert bridge.run_query(annotations).rows == []
    with world.modify_world():
        world.add_connection(FixedConnection(parent=parent, child=child))
        handle = Handle(root=child)
        world.add_semantic_annotation(handle)

    assert [row["__entity__"] for row in bridge.run_query(bodies).rows] == [
        str(value.name) for value in world.bodies
    ]
    assert [row["__entity__"] for row in bridge.run_query(annotations).rows] == [
        str(handle.name)
    ]
    with world.modify_world():
        world.remove_semantic_annotation(handle)
        world.remove_kinematic_structure_entity(child)

    assert [row["__entity__"] for row in bridge.run_query(bodies).rows] == [
        str(parent.name)
    ]
    assert bridge.run_query(annotations).rows == []


# %% preset discovery
def test_preset_discovery_does_not_evaluate_world_properties(
    world_with_two_bodies: tuple[World, Body, Body],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preset discovery uses collection annotations without invoking property getters.

    :param world_with_two_bodies: The world whose body property becomes unreadable.
    :param monkeypatch: Replaces a collection getter with an explicit failure.
    """
    world, _, _ = world_with_two_bodies

    def fail_read(world: World) -> list[Body]:
        """
        Reject any attempt to evaluate the body collection.

        :param world: The world whose collection must remain unread.
        :raises AssertionError: Always, when preset discovery reads the property.
        """
        raise AssertionError("preset discovery read a world property")

    monkeypatch.setattr(World, "bodies", property(fail_read))

    assert Preset.of_world(world, World.__name__.lower())


def test_native_world_entities_keep_names_without_declared_domains(
    annotated_robot_world: World,
) -> None:
    """
    Native query results retain entity names without predefined query domains.

    :param annotated_robot_world: The scene supplying bodies, connections and
        annotations.
    """
    entities = [
        annotated_robot_world.bodies[0],
        annotated_robot_world.connections[0],
        annotated_robot_world.semantic_annotations[0],
    ]

    result = RowRenderer().rows_of(entities)

    assert [row.values["__entity__"] for row in result.rows] == [
        str(entity.name) for entity in entities
    ]
    assert result.highlight == [str(entity.name) for entity in entities]


def test_world_presets_use_native_verbalization_for_distinct_collection_labels(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    Native collection queries receive distinct labels from their EQL verbalization.

    :param world_with_two_bodies: The empty scene whose collection presets are offered.
    """
    world, _, _ = world_with_two_bodies
    name = World.__name__.lower()
    runner = EqlQueryRunner(domains=[], extra_names={name: world})

    presets = [preset.worded(runner) for preset in Preset.of_world(world, name)]

    assert presets
    assert len({preset.text for preset in presets}) == len(presets)
    for preset in presets:
        assert preset.verbalization is not None
        assert preset.text == preset.verbalization.text
        assert runner.run_source(preset.code).rows == []
