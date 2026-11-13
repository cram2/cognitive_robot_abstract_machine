"""
Tests that copying and merging a world keep the simulator properties of the world and
its entities, and the degrees of freedom only an actuator drives.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.adapters.multi_sim import MujocoJoint, MujocoTendon
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    Actuator,
    Body,
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


def test_merge_keeps_properties_of_world_and_connection(simulated_world: World):
    hinge_name = simulated_world.connections[0].name
    expected = (
        list(simulated_world.simulator_additional_properties),
        list(simulated_world.connections[0].simulator_additional_properties),
    )
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    world.merge_world(simulated_world)

    assert (
        world.simulator_additional_properties,
        world.get_connection_by_name(hinge_name).simulator_additional_properties,
    ) == expected
