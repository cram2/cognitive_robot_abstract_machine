"""
Tests of what opening and closing :meth:`World.modify_world` blocks does to the world:

which degrees of freedom survive a nested block, what a block that modifies nothing
leaves behind, and how the fields the model is built from are changed by record.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
    SetDofHasHardwareInterface,
)


# %% degrees of freedom across nested blocks


def test_nested_block_keeps_degree_of_freedom_of_outer_block(
    world_with_two_bodies: tuple[World, Body, Body],
):
    world, parent, child = world_with_two_bodies
    degree_of_freedom = DegreeOfFreedom(name=PrefixedName("hinge"))
    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
        world.add_degree_of_freedom(degree_of_freedom)
        with world.modify_world():
            world.add_kinematic_structure_entity(child)
        world.add_connection(
            RevoluteConnection(
                parent=parent, child=child, axis=Vector3.Z(), raw_dof=degree_of_freedom
            )
        )

    assert world.degrees_of_freedom == [degree_of_freedom]


# %% blocks that modify nothing


def test_empty_block_keeps_model_version(mini_world: World):
    version = mini_world.get_world_model_manager().version

    with mini_world.modify_world():
        pass

    assert mini_world.get_world_model_manager().version == version


def test_empty_block_is_not_recorded(mini_world: World):
    history = list(mini_world.get_world_model_manager().model_modification_blocks)

    with mini_world.modify_world():
        pass

    assert mini_world.get_world_model_manager().model_modification_blocks == history


# %% moving the origin of a connection


def test_updating_connection_origin_is_recorded(mini_world: World):
    connection = mini_world.connections[0]

    with mini_world.modify_world():
        connection.update_parent_T_connection(
            parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
        )

    last_block = mini_world.get_world_model_manager().model_modification_blocks[-1]
    assert [type(modification) for modification in last_block] == [
        AttributeUpdateModification
    ]


def test_updating_connection_origin_moves_child(mini_world: World):
    connection = mini_world.connections[0]
    parent_T_connection = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)

    with mini_world.modify_world():
        connection.update_parent_T_connection(parent_T_connection=parent_T_connection)

    np.testing.assert_allclose(
        mini_world.compute_forward_kinematics_np(connection.parent, connection.child),
        parent_T_connection.to_np(),
    )


def test_rollback_of_connection_origin_update_restores_it(mini_world: World):
    connection = mini_world.connections[0]
    origin_before = connection.parent_T_connection_expression.to_np()
    version = mini_world.get_world_model_manager().version
    with mini_world.modify_world():
        connection.update_parent_T_connection(
            parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0)
        )

    mini_world.rollback_to_version(version)

    np.testing.assert_allclose(
        connection.parent_T_connection_expression.to_np(), origin_before
    )


# %% recorded changes of model fields


def test_updating_connection_origin_in_another_frame_raises(mini_world: World):
    connection = mini_world.connections[0]

    with pytest.raises(ReferenceFrameMismatchError):
        with mini_world.modify_world():
            connection.update_parent_T_connection(
                parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, reference_frame=connection.child
                )
            )


def test_setting_hardware_interface_is_recorded(mini_world: World):
    with mini_world.modify_world():
        mini_world.set_dofs_has_hardware_interface(mini_world.degrees_of_freedom, True)

    last_block = mini_world.get_world_model_manager().model_modification_blocks[-1]
    assert [type(modification) for modification in last_block] == [
        SetDofHasHardwareInterface
    ]
