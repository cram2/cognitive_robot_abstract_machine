"""
Tests that :meth:`World.move_branch_to_new_world` carries the state of the moved degrees
of freedom into the new world.
"""

from __future__ import annotations

import pytest

from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% fixtures


@pytest.fixture
def door_world() -> World:
    """
    A world whose root carries a branch made of a fixed frame and a door hinged to it.
    """
    world = World()
    branch_root = Body(name=PrefixedName("branch_root"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(parent=Body(name=PrefixedName("root")), child=branch_root)
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=branch_root,
                child=Body(name=PrefixedName("door")),
                axis=Vector3.Z(),
            )
        )
    return world


@pytest.fixture
def door_hinge(door_world: World) -> RevoluteConnection:
    """
    The hinge of the door in :func:`door_world`.
    """
    return door_world.get_connections_by_type(RevoluteConnection)[0]


# %% a branch moved to a new world keeps its joint state


def test_branch_moved_to_new_world_keeps_its_joint_positions(
    door_world: World, door_hinge: RevoluteConnection
):
    """
    An opened door stays open after its branch moves to a new world.
    """
    door_hinge.position = 0.5

    new_world = door_world.move_branch_to_new_world(door_hinge.parent)

    assert new_world.get_connection(door_hinge.parent, door_hinge.child).position == 0.5


def test_branch_moved_to_new_world_keeps_its_joint_velocities(
    door_world: World, door_hinge: RevoluteConnection
):
    """
    A door swinging when its branch moves keeps swinging in the new world.
    """
    door_hinge.velocity = 0.3

    new_world = door_world.move_branch_to_new_world(door_hinge.parent)

    assert new_world.get_connection(door_hinge.parent, door_hinge.child).velocity == 0.3


def test_branch_moved_to_new_world_keeps_a_floating_body_orientation():
    """
    A body hanging from a :class:`Connection6DoF` keeps its quaternion, instead of
    arriving with the all-zero quaternion, which is not a rotation.
    """
    world = World()
    branch_root = Body(name=PrefixedName("branch_root"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(parent=Body(name=PrefixedName("root")), child=branch_root)
        )
        floating = Connection6DoF.create_with_dofs(
            world=world, parent=branch_root, child=Body(name=PrefixedName("box"))
        )
        world.add_connection(floating)
    positions_before_move = [world.state[dof.id].position for dof in floating.dofs]

    new_world = world.move_branch_to_new_world(branch_root)

    assert [
        new_world.state[dof.id].position for dof in floating.dofs
    ] == positions_before_move
