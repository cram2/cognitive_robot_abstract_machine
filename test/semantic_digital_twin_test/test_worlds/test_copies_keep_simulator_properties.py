"""
Tests that copying and merging a world keep the simulator properties of the world and
its entities, and the degrees of freedom only an actuator drives.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import fields, is_dataclass

import numpy as np

import pytest

from typing_extensions import Any, Callable

from krrood.adapters.json_serializer import from_json, to_json
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.adapters.multi_sim import MujocoJoint, MujocoTendon
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.soft_trunk import SoftTrunk, SoftTrunkSection
from semantic_digital_twin.exceptions import DuplicateSimulatorPropertyError
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.soft_connections import (
    CosseratRodConnection,
    PiecewiseConstantCurvatureConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Actuator,
    Body,
    Connection,
    GravityCompensation,
    WorldEntityWithID,
)
from semantic_digital_twin.world_description.world_modification import (
    AddWorldSimulatorPropertyModification,
)


# %% fixtures


@pytest.fixture
def simulated_world() -> World:
    """
    A world whose body, hinge and the world itself carry simulator properties, with an
    actuator driving a degree of freedom that no connection uses, like a tendon.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    arm = Body(name=PrefixedName("arm"))
    arm.add_simulator_property(GravityCompensation(fraction=0.5))
    with world.modify_world():
        world.add_simulator_property(MujocoTendon(name="coupling"))
        hinge = RevoluteConnection.create_with_dofs(
            world=world, parent=root, child=arm, axis=Vector3.Z()
        )
        hinge.add_simulator_property(MujocoJoint(stiffness=[1.0, 2.0, 3.0]))
        world.add_connection(hinge)
        tendon = DegreeOfFreedom(name=PrefixedName("tendon"))
        world.add_degree_of_freedom(tendon)
        actuator = Actuator(name=PrefixedName("grip"))
        actuator.add_dof(tendon)
        world.add_actuator(actuator)
    return world


# %% degrees of freedom only an actuator drives


def test_degree_of_freedom_only_an_actuator_drives_is_kept(simulated_world: World):
    [actuator] = simulated_world.actuators

    assert actuator.dofs[0] in simulated_world.degrees_of_freedom


def test_copy_keeps_degree_of_freedom_only_an_actuator_drives(
    simulated_world: World,
):
    copied = deepcopy(simulated_world)

    [actuator] = copied.actuators
    assert actuator.dofs == [copied.get_degree_of_freedom_by_name("tendon")]


def test_merge_keeps_degree_of_freedom_only_an_actuator_drives(
    simulated_world: World,
):
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    world.merge_world(simulated_world)

    [actuator] = world.actuators
    assert actuator.dofs == [world.get_degree_of_freedom_by_name("tendon")]


def test_removing_a_branch_removes_the_actuators_of_its_joints(
    simulated_world: World,
):
    hinge = simulated_world.connections[0]
    joint_actuator = Actuator(name=PrefixedName("arm_motor"))
    joint_actuator.add_dof(hinge.dof)
    with simulated_world.modify_world():
        simulated_world.add_actuator(joint_actuator)

    with simulated_world.modify_world():
        simulated_world.remove_branch_from_world(hinge.child)

    assert (
        [actuator.name.name for actuator in simulated_world.actuators],
        [dof.name.name for dof in simulated_world.degrees_of_freedom],
    ) == (["grip"], ["tendon"])


# %% copying a world


def test_copy_keeps_properties_of_world_body_and_connection(simulated_world: World):
    copied = deepcopy(simulated_world)

    assert (
        copied.simulator_additional_properties,
        copied.get_body_by_name("arm").simulator_additional_properties,
        copied.connections[0].simulator_additional_properties,
    ) == (
        simulated_world.simulator_additional_properties,
        simulated_world.get_body_by_name("arm").simulator_additional_properties,
        simulated_world.connections[0].simulator_additional_properties,
    )


# %% merging a world


def test_merge_keeps_properties_of_world_body_and_connection(simulated_world: World):
    hinge_name = simulated_world.connections[0].name
    expected = (
        list(simulated_world.simulator_additional_properties),
        list(simulated_world.get_body_by_name("arm").simulator_additional_properties),
        list(simulated_world.connections[0].simulator_additional_properties),
    )
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    world.merge_world(simulated_world)

    assert (
        world.simulator_additional_properties,
        world.get_body_by_name("arm").simulator_additional_properties,
        world.get_connection_by_name(hinge_name).simulator_additional_properties,
    ) == expected


def test_merge_with_unique_property_in_both_worlds_raises(simulated_world: World):
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    with world.modify_world():
        world.add_simulator_property(GravityCompensation(fraction=1.0))
    with simulated_world.modify_world():
        simulated_world.add_simulator_property(GravityCompensation(fraction=0.5))

    with pytest.raises(DuplicateSimulatorPropertyError):
        world.merge_world(simulated_world)


def test_failed_merge_leaves_receiving_world_properties_unchanged(
    simulated_world: World,
):
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    with world.modify_world():
        world.add_simulator_property(GravityCompensation(fraction=1.0))
    with simulated_world.modify_world():
        simulated_world.add_simulator_property(GravityCompensation(fraction=0.5))
    properties = list(world.simulator_additional_properties)

    with pytest.raises(DuplicateSimulatorPropertyError):
        world.merge_world(simulated_world)

    assert world.simulator_additional_properties == properties


def test_failed_merge_leaves_merged_world_unchanged(simulated_world: World):
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    with world.modify_world():
        world.add_simulator_property(GravityCompensation(fraction=1.0))
    with simulated_world.modify_world():
        simulated_world.add_simulator_property(GravityCompensation(fraction=0.5))
    bodies = list(simulated_world.bodies)

    with pytest.raises(DuplicateSimulatorPropertyError):
        world.merge_world(simulated_world)

    assert simulated_world.bodies == bodies


# %% properties of the world itself


def test_merge_records_moving_the_world_properties(simulated_world: World):
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    world.merge_world(simulated_world)

    last_block = world.get_world_model_manager().model_modification_blocks[-1]
    assert [
        modification.simulator_property
        for modification in last_block
        if isinstance(modification, AddWorldSimulatorPropertyModification)
    ] == [MujocoTendon(name="coupling")]


def test_world_property_addition_survives_json(simulated_world: World):
    """
    The addition of a simulator property to a world, sent through JSON, adds the
    property to the receiving world.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    sent = AddWorldSimulatorPropertyModification(
        simulator_property=MujocoTendon(name="coupling")
    )

    received = from_json(json.loads(json.dumps(to_json(sent))))
    with world.modify_world():
        received.apply(world=world)

    assert world.simulator_additional_properties == [MujocoTendon(name="coupling")]


def test_rollback_of_world_property_addition_removes_it(simulated_world: World):
    version = simulated_world.get_world_model_manager().version
    with simulated_world.modify_world():
        simulated_world.add_simulator_property(MujocoTendon(name="second"))

    simulated_world.rollback_to_version(version)

    assert simulated_world.simulator_additional_properties == [
        MujocoTendon(name="coupling")
    ]


# %% looking entities up by id


@pytest.mark.parametrize(
    "entity_of",
    [
        lambda world: world.get_body_by_name("arm"),
        lambda world: world.connections[0],
        lambda world: world.get_degree_of_freedom_by_name("tendon"),
        lambda world: world.actuators[0],
    ],
    ids=["body", "connection", "degree_of_freedom", "actuator"],
)
def test_entity_is_found_by_its_id(
    simulated_world: World, entity_of: Callable[[World], WorldEntityWithID]
):
    entity = entity_of(simulated_world)

    assert simulated_world.find_world_entity_with_id(entity.id) is entity


# %% every field of every copied entity


def comparable_value(value: Any) -> Any:
    """
    :param value: A field value of a world entity.
    :return: A value equal for an original and its copy in another world: entities
        stand for their id, symbolic expressions for their text, containers and
        dataclasses for their contents.
    """
    if isinstance(value, WorldEntityWithID):
        return type(value).__name__, value.id
    if isinstance(value, SymbolicMathType):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [comparable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(repr(comparable_value(item)) for item in value)
    if isinstance(value, dict):
        return {repr(key): comparable_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return type(value).__name__, comparable_fields(value)
    return value


def comparable_fields(instance: Any) -> dict[str, Any]:
    """
    :param instance: A dataclass instance.
    :return: Every field but the world it belongs to, private and derived ones
        included, as :func:`comparable_value` makes it.
    """
    return {
        field_.name: comparable_value(getattr(instance, field_.name))
        for field_ in fields(instance)
        if field_.name != "_world"
    }


def world_entities(world: World) -> list[WorldEntityWithID]:
    """
    :param world: A world.
    :return: Every entity the world holds, of every kind.
    """
    return [
        *world.kinematic_structure_entities,
        *world.connections,
        *world.degrees_of_freedom,
        *world.actuators,
        *world.semantic_annotations,
    ]


@pytest.mark.parametrize("world_fixture", ["simulated_world", "pr2_world_copy"])
def test_copy_keeps_every_field_of_every_entity(
    world_fixture: str, request: pytest.FixtureRequest
):
    world: World = request.getfixturevalue(world_fixture)

    copied = deepcopy(world)

    assert [
        comparable_fields(copied.get_world_entity_with_id_by_id(entity.id))
        for entity in world_entities(world)
    ] == [comparable_fields(entity) for entity in world_entities(world)]


# %% copying every kind of connection


def connect_with_dofs(
    connection_type: type[Connection], **arguments
) -> Callable[[World, Body, Body], Connection]:
    """
    :param connection_type: A connection type with :meth:`create_with_dofs`.
    :param arguments: Further arguments of :meth:`create_with_dofs`.
    :return: A function connecting a parent to a child with a new connection of the type.
    """
    return lambda world, parent, child: connection_type.create_with_dofs(
        world=world, parent=parent, child=child, **arguments
    )


@pytest.mark.parametrize(
    "connect",
    [
        lambda world, parent, child: FixedConnection(parent=parent, child=child),
        connect_with_dofs(RevoluteConnection, axis=Vector3.Z()),
        connect_with_dofs(PrismaticConnection, axis=Vector3.X()),
        connect_with_dofs(ScrewConnection, axis=Vector3.Z(), screw_pitch=0.01),
        connect_with_dofs(Connection6DoF),
        connect_with_dofs(OmniDrive),
        connect_with_dofs(DifferentialDrive),
    ],
    ids=["fixed", "revolute", "prismatic", "screw", "6 dof", "omni", "differential"],
)
def test_copy_keeps_connection_properties(
    connect: Callable[[World, Body, Body], Connection],
):
    world = World()
    with world.modify_world():
        connection = connect(
            world, Body(name=PrefixedName("parent")), Body(name=PrefixedName("child"))
        )
        connection.add_simulator_property(MujocoJoint(stiffness=[1.0, 2.0, 3.0]))
        world.add_connection(connection)

    copied = deepcopy(world)

    assert (
        copied.connections[0].simulator_additional_properties
        == connection.simulator_additional_properties
    )


@pytest.mark.parametrize(
    "build_trunk, connection_type",
    [
        (
            SoftTrunk.build_piecewise_constant_curvature,
            PiecewiseConstantCurvatureConnection,
        ),
        (SoftTrunk.build_cosserat, CosseratRodConnection),
    ],
    ids=["piecewise constant curvature", "cosserat rod"],
)
def test_copy_keeps_soft_connection_properties(
    build_trunk: Callable[[World, list[SoftTrunkSection]], SoftTrunk],
    connection_type: type[Connection],
):
    world = World()
    build_trunk(world, [SoftTrunkSection(length=0.3, radius=0.02, resolution=2)])
    segment = world.get_connections_by_type(connection_type)[0]
    segment.add_simulator_property(MujocoJoint(stiffness=[1.0, 2.0, 3.0]))

    copied = deepcopy(world)

    assert (
        copied.get_connection_by_name(segment.name).simulator_additional_properties
        == segment.simulator_additional_properties
    )


# %% re-attached connections


def test_connection_with_new_parent_has_its_own_properties(simulated_world: World):
    hinge = simulated_world.connections[0]

    moved = hinge.copy_with_new_parent(
        Body(name=PrefixedName("new_parent")), HomogeneousTransformationMatrix()
    )

    assert (
        moved.simulator_additional_properties == hinge.simulator_additional_properties
        and moved.simulator_additional_properties
        is not hinge.simulator_additional_properties
    )


def test_connection_with_new_child_has_its_own_properties(simulated_world: World):
    hinge = simulated_world.connections[0]

    moved = hinge.copy_with_new_child(Body(name=PrefixedName("new_child")))

    assert (
        moved.simulator_additional_properties == hinge.simulator_additional_properties
        and moved.simulator_additional_properties
        is not hinge.simulator_additional_properties
    )
