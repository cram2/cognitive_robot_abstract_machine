"""
Tests that rolling back modification blocks, and the reset state context around them,
give back the world as it was: its connections, the state of its degrees of freedom, and
a history whose versions still map to its blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    RollbackBeforeClearError,
    WorldMergedWithItselfError,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    ScrewAxisConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% fixtures


@pytest.fixture
def free_item_world() -> tuple[World, Body, Body, ScrewAxisConnection]:
    """
    A world whose root carries a free item on a 6 degree of freedom connection, a table
    it can be attached to, and an arm joint added after the item.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    item = Body(name=PrefixedName("item"))
    table = Body(name=PrefixedName("table"))
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=item)
        )
        world.add_connection(FixedConnection(parent=root, child=table))
    with world.modify_world():
        arm_joint = ScrewAxisConnection.create_with_dofs(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
            world=world,
            parent=root,
            child=Body(name=PrefixedName("arm")),
        )
        world.add_connection(arm_joint)
    with world.modify_world():
        item.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=3.0, yaw=0.5, reference_frame=root
        )
        arm_joint.position = 0.7
    return world, item, table, arm_joint


# %% removing a body with its connection


def test_rollback_of_body_removal_reconnects_it():
    world = World()
    root = Body(name=PrefixedName("root"))
    leaf = Body(name=PrefixedName("leaf"))
    with world.modify_world():
        connection = ScrewAxisConnection.create_with_dofs(
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
            world=world,
            parent=root,
            child=leaf,
        )
        world.add_connection(connection)
    version = world.get_world_model_manager().version
    with world.modify_world():
        world.remove_kinematic_structure_entity(leaf)

    world.rollback_to_version(version)

    assert leaf.parent_kinematic_structure_entity == root


# %% state of degrees of freedom brought back by a rollback


def test_rollback_of_attachment_restores_free_pose(free_item_world):
    world, item, table, _ = free_item_world
    pose_before = item.global_transform.to_np()
    version = world.get_world_model_manager().version
    with world.modify_world():
        world.move_branch_with_fixed_connection(item, table)

    world.rollback_to_version(version)

    np.testing.assert_allclose(item.global_transform.to_np(), pose_before)


def test_reset_state_context_restores_by_degree_of_freedom(free_item_world):
    world, item, table, arm_joint = free_item_world
    pose_before = item.global_transform.to_np()
    version = world.get_world_model_manager().version

    with world.reset_state_context():
        with world.modify_world():
            world.move_branch_with_fixed_connection(item, table)
        world.rollback_to_version(version)

    assert (arm_joint.position, item.global_transform.to_np().tolist()) == (
        0.7,
        pose_before.tolist(),
    )


def test_reset_state_context_restores_state_without_model_change(free_item_world):
    world, item, _, arm_joint = free_item_world
    pose_before = item.global_transform.to_np()

    with world.reset_state_context():
        arm_joint.position = -0.3
        item.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=5.0, reference_frame=item.parent_connection.parent
        )

    assert (arm_joint.position, item.global_transform.to_np().tolist()) == (
        0.7,
        pose_before.tolist(),
    )


def test_degree_of_freedom_added_in_reset_state_context_keeps_its_state(
    free_item_world,
):
    world, _, table, _ = free_item_world

    with world.reset_state_context():
        with world.modify_world():
            drawer_joint = ScrewAxisConnection.create_with_dofs(
                screw_axis=ScrewAxisConnection.create_revolute_screw_axis(
                    axis=Vector3.Z()
                ),
                world=world,
                parent=table,
                child=Body(name=PrefixedName("drawer")),
            )
            world.add_connection(drawer_joint)
        drawer_joint.position = 0.4

    assert drawer_joint.position == 0.4


# %% histories that were cleared


def test_rollback_of_merged_world_adds_nothing_back(mini_world: World):
    other = World()
    with other.modify_world():
        other.add_kinematic_structure_entity(Body(name=PrefixedName("other_root")))
    mini_world.merge_world(other)

    other.rollback_modification_blocks(len(other.modification_history))

    assert other.kinematic_structure_entities == []


def test_rollback_to_before_replace_with_raises(mini_world: World):
    replacement = World()
    with replacement.modify_world():
        replacement.add_kinematic_structure_entity(Body(name=PrefixedName("new_root")))
    version = mini_world.get_world_model_manager().version

    mini_world._replace_with(replacement)

    with pytest.raises(RollbackBeforeClearError):
        mini_world.rollback_to_version(version)


def test_rollback_after_replace_with_reverts_later_blocks(mini_world: World):
    replacement = World()
    with replacement.modify_world():
        replacement.add_kinematic_structure_entity(Body(name=PrefixedName("new_root")))
    mini_world._replace_with(replacement)
    version = mini_world.get_world_model_manager().version
    entities = list(mini_world.kinematic_structure_entities)
    with mini_world.modify_world():
        mini_world.add_connection(
            FixedConnection(
                parent=mini_world.root, child=Body(name=PrefixedName("later"))
            )
        )

    mini_world.rollback_to_version(version)

    assert mini_world.kinematic_structure_entities == entities


def test_merging_world_with_itself_raises(mini_world: World):
    with pytest.raises(WorldMergedWithItselfError):
        mini_world.merge_world(mini_world)
