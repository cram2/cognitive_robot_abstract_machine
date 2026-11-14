"""
Tests that copying and merging a world keep the simulator properties of the world and
its entities, and the degrees of freedom only an actuator drives.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from typing_extensions import Callable

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
    world.add_simulator_property(MujocoTendon(name="coupling"))
    with world.modify_world():
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
    world.add_simulator_property(GravityCompensation(fraction=1.0))
    simulated_world.add_simulator_property(GravityCompensation(fraction=0.5))

    with pytest.raises(DuplicateSimulatorPropertyError):
        world.merge_world(simulated_world)


def test_failed_merge_leaves_merged_world_unchanged(simulated_world: World):
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))
    world.add_simulator_property(GravityCompensation(fraction=1.0))
    simulated_world.add_simulator_property(GravityCompensation(fraction=0.5))
    bodies = list(simulated_world.bodies)

    with pytest.raises(DuplicateSimulatorPropertyError):
        world.merge_world(simulated_world)

    assert simulated_world.bodies == bodies


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
