from __future__ import annotations

import pytest

from cramera.knowledge.query_runner import EqlQueryRunner
from cramera.knowledge.queryable_knowledge import QueryScope
from cramera.live.world_query import WorldQueryLabel, WorldQueryName, WorldQuerySource
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Table,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


# %% native world fixtures
@pytest.fixture()
def annotated_robot_world(pr2_world_copy: World) -> World:
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


# %% empty and robotless worlds
def test_empty_world_exposes_only_native_current_state_domains(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    world, _, _ = world_with_two_bodies
    source = WorldQuerySource(world)

    [knowledge] = source.knowledge()

    assert knowledge.scope is QueryScope.CURRENT_STATE
    assert {domain.name: domain.entity_type for domain in knowledge.domains} == {
        WorldQueryName.BODY: Body,
        WorldQueryName.ANNOTATION: SemanticAnnotation,
        WorldQueryName.HANDLE: Handle,
        WorldQueryName.SURFACE: HasSupportingSurface,
        WorldQueryName.ROBOT: AbstractRobot,
        WorldQueryName.ARM: Arm,
    }
    assert all(domain.objects == [] for domain in knowledge.domains)
    assert knowledge.extra_names == {}


@pytest.mark.parametrize("domain_name", list(WorldQueryName))
def test_each_empty_world_preset_returns_an_empty_answer(
    world_with_two_bodies: tuple[World, Body, Body], domain_name: WorldQueryName
) -> None:
    world, _, _ = world_with_two_bodies
    source = WorldQuerySource(world)
    [knowledge] = source.knowledge()
    index = [domain.name for domain in knowledge.domains].index(domain_name)
    preset = source.presets()[index]

    result = EqlQueryRunner(knowledge.domains).run(preset.code)

    assert result.ok
    assert result.rows == []
    assert result.count == 0


def test_robotless_world_keeps_bodies_and_semantic_annotations_queryable(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    world, parent, child = world_with_two_bodies
    handle = Handle(root=child)
    table = Table(root=parent)
    with world.modify_world():
        world.add_connection(FixedConnection(parent=parent, child=child))
        world.add_semantic_annotations([handle, table])

    [knowledge] = WorldQuerySource(world).knowledge()
    domains = {domain.name: domain.objects for domain in knowledge.domains}

    assert domains[WorldQueryName.BODY] == world.bodies
    assert domains[WorldQueryName.HANDLE] == [handle]
    assert domains[WorldQueryName.SURFACE] == [table]
    assert domains[WorldQueryName.ROBOT] == []
    assert domains[WorldQueryName.ARM] == []


# %% native identities and executable presets
def test_domains_preserve_native_robot_and_annotation_instances(
    annotated_robot_world: World,
) -> None:
    [knowledge] = WorldQuerySource(annotated_robot_world).knowledge()

    for domain in knowledge.domains:
        expected = (
            annotated_robot_world.bodies
            if domain.entity_type is Body
            else annotated_robot_world.get_semantic_annotations_by_type(
                domain.entity_type
            )
        )
        assert len(domain.objects) == len(expected)
        assert all(
            actual is original for actual, original in zip(domain.objects, expected)
        )


@pytest.mark.parametrize("domain_name", list(WorldQueryName))
def test_each_annotated_world_preset_returns_its_native_entities(
    annotated_robot_world: World, domain_name: WorldQueryName
) -> None:
    source = WorldQuerySource(annotated_robot_world)
    [knowledge] = source.knowledge()
    index = [domain.name for domain in knowledge.domains].index(domain_name)
    domain = knowledge.domains[index]
    preset = source.presets()[index]

    result = EqlQueryRunner(knowledge.domains).run(
        preset.code, limit=len(domain.objects)
    )

    assert result.ok
    assert domain.objects
    assert result.count == len(domain.objects)
    assert {row["__entity__"] for row in result.rows} == {
        str(entity.name) for entity in domain.objects
    }


def test_presets_offer_each_domain_with_its_own_label(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    world, _, _ = world_with_two_bodies
    source = WorldQuerySource(world)

    assert source.title() == WorldQueryLabel.TITLE
    assert [preset.text for preset in source.presets()] == [
        WorldQueryLabel.BODIES,
        WorldQueryLabel.ANNOTATIONS,
        WorldQueryLabel.HANDLES,
        WorldQueryLabel.SURFACES,
        WorldQueryLabel.ROBOTS,
        WorldQueryLabel.ARMS,
    ]
    assert all(preset.scope is QueryScope.CURRENT_STATE for preset in source.presets())
    assert source.unlisted_presets() == []


# %% model changes and shared locking
def test_domains_follow_body_and_annotation_additions_and_removals(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    world, parent, child = world_with_two_bodies
    source = WorldQuerySource(world)
    [initial] = source.knowledge()
    with world.modify_world():
        world.add_connection(FixedConnection(parent=parent, child=child))
        handle = Handle(root=child)
        world.add_semantic_annotation(handle)

    [added] = source.knowledge()
    added_domains = {domain.name: domain.objects for domain in added.domains}

    assert added_domains[WorldQueryName.BODY] == world.bodies
    assert added_domains[WorldQueryName.HANDLE] == [handle]
    assert all(domain.objects == [] for domain in initial.domains)

    with world.modify_world():
        world.remove_semantic_annotation(handle)
        world.remove_kinematic_structure_entity(child)

    [removed] = source.knowledge()
    removed_domains = {domain.name: domain.objects for domain in removed.domains}

    assert removed_domains[WorldQueryName.BODY] == [parent]
    assert removed_domains[WorldQueryName.HANDLE] == []
    assert removed_domains[WorldQueryName.ANNOTATION] == []


def test_read_scope_uses_the_native_world_lock(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    world, _, _ = world_with_two_bodies

    assert WorldQuerySource(world).read_scope() is world.state.world_lock
