import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
import pytest
from numpy.testing import assert_raises
from typing_extensions import Tuple

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    DuplicateWorldEntityError,
    MismatchingPublishChangesAttribute,
    UsageError,
    MissingWorldModificationContextError,
    DofNotInWorldStateError,
    WrongWorldModelVersion,
    NonMonotonicTimeError,
    BrokenWorldModificationHistoryError,
    WorldEntityNotFoundError,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Milk,
    Drawer,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap

# from semantic_digital_twin.spatial_types.math import rotation_matrix_from_rpy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
)
from semantic_digital_twin.testing import StateChangeCounter, world_setup
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Actuator,
    WorldEntityWithClassBasedID,
    WorldEntityWithID,
)
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
)
from semantic_digital_twin.world_description.world_state import (
    WorldStateTrajectory,
    WorldState,
)
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)


def test_create_with_root_body_names_the_root_from_a_plain_string():
    world = World.create_with_root_body("kitchen")
    assert world.root.name == PrefixedName("kitchen")


def test_create_with_root_body_defaults_the_root_name_to_map():
    world = World.create_with_root_body()
    assert world.root.name == PrefixedName("map")


def test_force_root_name_renames_the_root_body():
    world = World.create_with_root_body("map")
    world.force_root_name(PrefixedName("odom"))
    assert world.root.name == PrefixedName("odom")


def test_force_root_name_preserves_root_identity_and_children(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    root_id = world.root.id
    bf_connection_before = world.get_connection(world.root, bf)

    world.force_root_name(PrefixedName("new_root", prefix="world"))

    assert world.root.id == root_id
    assert world.root.name == PrefixedName("new_root", prefix="world")
    assert world.get_connection(world.root, bf) is bf_connection_before
    assert world.get_kinematic_structure_entity_by_name(PrefixedName("bf")) is bf


def test_force_root_name_records_an_attribute_update_modification():
    world = World.create_with_root_body("map")
    root_id = world.root.id

    world.force_root_name(PrefixedName("odom"))

    last_block = world._model_manager.model_modification_blocks[-1]
    attribute_updates = [
        modification
        for modification in last_block.modifications
        if isinstance(modification, AttributeUpdateModification)
        and modification.entity_id == root_id
    ]
    assert len(attribute_updates) == 1


def test_set_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    c1: PrismaticConnection = world.get_connection(l1, l2)
    c1.position = 1.0
    assert c1.position == 1.0
    c2: RevoluteConnection = world.get_connection(r1, r2)
    c2.position = 1337
    assert c2.position == 1337
    c3: Connection6DoF = world.get_connection(world.root, bf)
    transform = RotationMatrix.from_rpy(1, 0, 0).to_np()
    transform[0, 3] = 69
    c3.origin = HomogeneousTransformationMatrix(
        data=transform, reference_frame=world.root, child_frame=bf
    )
    assert np.allclose(world.compute_forward_kinematics_np(world.root, bf), transform)

    JointState.from_mapping({c1: 2}).apply_to(world)
    assert c1.position == 2.0

    transform[0, 3] += c1.position
    assert np.allclose(l2.global_transform.to_np(), transform)


def test_construction(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world.validate()
    assert len(world.connections) == 5
    assert len(world.kinematic_structure_entities) == 6
    assert world.state.positions[0] == 0
    assert world.get_connection(l1, l2).dof.id == world.get_connection(r1, r2).dof.id


def test_chain_of_bodies(world_setup):
    world, _, l2, _, _, _ = world_setup
    result = world.compute_chain_of_kinematic_structure_entities(
        root=world.root, tip=l2
    )
    result = [x.name for x in result]
    assert result == [
        PrefixedName(name="root", prefix="world"),
        PrefixedName(name="bf", prefix=None),
        PrefixedName(name="l1", prefix=None),
        PrefixedName(name="l2", prefix=None),
    ]


def test_chain_of_connections(world_setup):
    world, _, l2, _, _, _ = world_setup
    result = world.compute_chain_of_connections(root=world.root, tip=l2)
    result = [x.name for x in result]
    assert result == [
        PrefixedName(name="root_T_bf", prefix=None),
        PrefixedName(name="bf_T_l1", prefix=None),
        PrefixedName(name="l1_T_l2", prefix=None),
    ]


def test_split_chain_of_bodies(world_setup):
    world, _, l2, _, _, r2 = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r2, tip=l2)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [PrefixedName(name="r2", prefix=None), PrefixedName(name="r1", prefix=None)],
        [PrefixedName(name="bf", prefix=None)],
        [PrefixedName(name="l1", prefix=None), PrefixedName(name="l2", prefix=None)],
    )


def test_split_chain_of_bodies_adjacent1(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r2, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [PrefixedName(name="r2", prefix=None)],
        [PrefixedName(name="r1", prefix=None)],
        [],
    )


def test_split_chain_of_bodies_adjacent2(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r1, tip=r2)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [],
        [PrefixedName(name="r1", prefix=None)],
        [PrefixedName(name="r2", prefix=None)],
    )


def test_split_chain_of_bodies_identical(world_setup):
    world, _, _, _, r1, _ = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r1, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([], [PrefixedName(name="r1", prefix=None)], [])


def test_split_chain_of_connections(world_setup):
    world, _, l2, _, _, r2 = world_setup
    result = world.compute_split_chain_of_connections(root=r2, tip=l2)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [
            PrefixedName(name="r1_T_r2", prefix=None),
            PrefixedName(name="bf_T_r1", prefix=None),
        ],
        [
            PrefixedName(name="bf_T_l1", prefix=None),
            PrefixedName(name="l1_T_l2", prefix=None),
        ],
    )


def test_split_chain_of_connections_adjacent1(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_connections(root=r2, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([PrefixedName(name="r1_T_r2", prefix=None)], [])


def test_split_chain_of_connections_adjacent2(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_connections(root=r1, tip=r2)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([], [PrefixedName(name="r1_T_r2", prefix=None)])


def test_split_chain_of_connections_identical(world_setup):
    world, _, _, _, r1, _ = world_setup
    result = world.compute_split_chain_of_connections(root=r1, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([], [])


@pytest.mark.skip(
    reason="readding of 1dof connection broken because reference to dof is lost"
)
def test_nested_with_blocks_illegal_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup

    with world.modify_world():
        connection1 = world.get_connection(l1, l2)
        world.remove_connection(connection1)
        with world.modify_world():
            connection2 = world.get_connection(r1, r2)
            world.remove_connection(connection2)
        world.add_connection(connection1)
        world.add_connection(connection2)


# %% batching state changes


def test_batching_state_changes_announces_them_once(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    counter = StateChangeCounter(_world=world)

    with world.batch_state_changes():
        prismatic_connection.position = 1.0
        prismatic_connection.position = 2.0

    assert counter.count == 1
    assert prismatic_connection.position == 2.0


def test_state_changes_outside_a_batch_are_announced_individually(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    counter = StateChangeCounter(_world=world)

    prismatic_connection.position = 1.0
    prismatic_connection.position = 2.0

    assert counter.count == 2


def test_batching_without_a_state_change_announces_nothing(world_setup):
    world, _, _, _, _, _ = world_setup
    counter = StateChangeCounter(_world=world)

    with world.batch_state_changes():
        pass

    assert counter.count == 0


def test_nested_batches_of_state_changes_announce_them_once(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    counter = StateChangeCounter(_world=world)

    with world.batch_state_changes():
        prismatic_connection.position = 1.0
        with world.batch_state_changes():
            prismatic_connection.position = 2.0
        assert counter.count == 0

    assert counter.count == 1


def test_batching_state_changes_forwards_its_publish_changes(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    counter = StateChangeCounter(_world=world)

    with world.batch_state_changes(publish_changes=False):
        prismatic_connection.position = 1.0

    assert counter.count == 1
    assert counter.publish_changes_of_last_call is False


def test_a_nested_batch_disagreeing_about_publishing_is_rejected(world_setup):
    world, _, _, _, _, _ = world_setup

    with pytest.raises(MismatchingPublishChangesAttribute):
        with world.batch_state_changes(publish_changes=False):
            with world.batch_state_changes(publish_changes=True):
                pass


def test_a_batch_interrupted_by_an_error_still_announces_its_changes(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    counter = StateChangeCounter(_world=world)

    with pytest.raises(ZeroDivisionError):
        with world.batch_state_changes():
            prismatic_connection.position = 1.0
            raise ZeroDivisionError

    assert counter.count == 1
    assert prismatic_connection.position == 1.0


def test_compute_fk_connection6dof(world_setup):
    world, _, _, bf, _, _ = world_setup
    fk = world.compute_forward_kinematics_np(world.root, bf)
    np.testing.assert_array_equal(fk, np.eye(4))

    connection: Connection6DoF = world.get_connection(world.root, bf)

    world.state[connection.x.id].position = 1.0
    world.state[connection.qw.id].position = 0
    world.state[connection.qz.id].position = 1
    world.notify_state_change()
    fk = world.compute_forward_kinematics_np(world.root, bf)
    np.testing.assert_array_equal(
        fk,
        [
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def test_compute_fk(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    fk = world.compute_forward_kinematics_np(l2, r2)
    np.testing.assert_array_equal(fk, np.eye(4))

    connection: PrismaticConnection = world.get_connection(r1, r2)

    state_memory_id = id(world.state._data)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()
    assert state_memory_id == id(world.state._data)
    fk = world.compute_forward_kinematics_np(l2, r2)
    assert np.allclose(
        fk,
        np.array(
            [
                [0.540302, -0.841471, 0.0, -1.0],
                [0.841471, 0.540302, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_compute_ik(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    target = np.array(
        [
            [0.540302, -0.841471, 0.0, -1.0],
            [0.841471, 0.540302, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    joint_state = world.compute_inverse_kinematics(
        l2, r2, HomogeneousTransformationMatrix(target, reference_frame=l2)
    )
    for joint, state in joint_state.items():
        world.state[joint.id].position = state
    world.notify_state_change()
    assert np.allclose(world.compute_forward_kinematics_np(l2, r2), target, atol=1e-3)


def test_compute_fk_expression(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()
    fk = world.compute_forward_kinematics_np(r2, l2)
    fk_expr = world.compose_forward_kinematics_expression(r2, l2)
    fk2 = fk_expr.evaluate()
    np.testing.assert_array_almost_equal(fk, fk2)


def test_apply_control_commands(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    state_memory_id = id(world.state._data)
    connection: PrismaticConnection = world.get_connection(r1, r2)
    cmd = np.array([100.0, 0, 0, 0, 0, 0, 0, 0])
    dt = 0.1
    world.apply_control_commands(cmd, dt, Derivatives.jerk)
    assert world.state[connection.dof.id].jerk == 100.0
    assert world.state[connection.dof.id].acceleration == 100.0 * dt
    assert world.state[connection.dof.id].velocity == 100.0 * dt * dt
    assert world.state[connection.dof.id].position == 100.0 * dt * dt * dt
    # the state should reuse the same memory
    assert state_memory_id == id(world.state._data)


def test_compute_relative_pose(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(l1, l2)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()

    pose = HomogeneousTransformationMatrix(reference_frame=l2)
    relative_pose = world.transform(pose, l1)
    expected_pose = HomogeneousTransformationMatrix(
        [
            [1.0, 0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose.to_np())


def test_compute_relative_pose_both(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world.get_connection(world.root, bf).origin = HomogeneousTransformationMatrix(
        data=np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        reference_frame=world.root,
        child_frame=bf,
    )
    world.notify_state_change()

    pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, reference_frame=bf)
    relative_pose = world.transform(pose, world.root)
    # Rotation is 90 degrees around z-axis, translation is 1 along x-axis
    expected_pose = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose)


def test_compute_relative_pose_only_translation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(l1, l2)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()

    pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0, reference_frame=l2)
    relative_pose = world.transform(pose, l1)
    expected_pose = np.array(
        [
            [1.0, 0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose)


def test_compute_relative_pose_only_rotation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: RevoluteConnection = world.get_connection(r1, r2)
    world.state[connection.dof.id].position = np.pi / 2  # 90 degrees
    world.notify_state_change()

    pose = HomogeneousTransformationMatrix(reference_frame=r2)
    relative_pose = world.transform(pose, r1)
    expected_pose = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose)


def test_add_semantic_annotation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    v = SemanticAnnotation(name=PrefixedName("muh"))
    with world.modify_world():
        world.add_semantic_annotation(v)
        world.add_semantic_annotation(v)
    assert world.get_semantic_annotation_by_name(v.name) == v


def test_duplicate_semantic_annotation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    v = SemanticAnnotation(name=PrefixedName("muh"))
    with world.modify_world():
        world.add_semantic_annotation(v)
        world.semantic_annotations.append(v)
    with pytest.raises(DuplicateWorldEntityError):
        world.get_semantic_annotation_by_name(v.name)


def test_all_kinematic_structure_entities_have_uuid(world_setup):
    world, _, _, _, _, _ = world_setup
    uuids = {kse.id for kse in world.kinematic_structure_entities}

    assert len(uuids) == len(world.kinematic_structure)


def test_all_degree_of_freedom_have_uuid(world_setup):
    world, _, _, _, _, _ = world_setup
    uuids = {dof.id for dof in world.degrees_of_freedom}

    assert len(uuids) == len(world.degrees_of_freedom)


def test_merge_world(world_setup, pr2_world_copy):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_copy.get_kinematic_structure_entity_by_name("base_link")
    r_gripper_tool_frame = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_copy.get_connection(
        torso_lift_link,
        pr2_world_copy.get_kinematic_structure_entity_by_name("r_shoulder_pan_link"),
    )

    l_shoulder_pan_joint = pr2_world_copy.get_connection(
        torso_lift_link,
        pr2_world_copy.get_kinematic_structure_entity_by_name("l_shoulder_pan_link"),
    )

    world.merge_world(pr2_world_copy)

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert l_shoulder_pan_joint in world.connections
    assert torso_lift_link in world.bodies
    assert r_shoulder_pan_joint in world.connections


def test_merge_with_connection(world_setup, pr2_world_copy):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_copy.get_kinematic_structure_entity_by_name("base_link")
    r_gripper_tool_frame = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_copy.get_connection(
        torso_lift_link,
        pr2_world_copy.get_kinematic_structure_entity_by_name("r_shoulder_pan_link"),
    )

    pose = np.eye(4)
    pose[0, 3] = 1.0

    origin = HomogeneousTransformationMatrix(pose)

    connection = pr2_world_copy.get_connection_by_name(
        PR2Joint.LEFT_GRIPPER_LEFT_FINGER
    )
    connection_dof_id = connection.dof.id
    pr2_world_copy.state[connection.dof.id].position = 0.55
    pr2_world_copy.notify_state_change()
    expected_fk = pr2_world_copy.compute_forward_kinematics(
        connection.parent, connection.child
    ).to_np()

    new_connection = FixedConnection(
        parent=world.root,
        child=pr2_world_copy.root,
        parent_T_connection_expression=origin,
    )

    world.merge_world(pr2_world_copy, new_connection)
    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert new_connection in world.connections
    assert torso_lift_link in world.bodies
    assert r_shoulder_pan_joint in world.connections
    assert world.state[connection_dof_id].position == 0.55
    assert world.compute_forward_kinematics_np(world.root, base_link)[
        0, 3
    ] == pytest.approx(1.0, abs=1e-6)
    actual_fk = world.compute_forward_kinematics(
        connection.parent, connection.child
    ).to_np()
    assert np.allclose(actual_fk, expected_fk)


def test_merge_world_with_connection_worldless_parent():
    world1 = World()
    b1 = Body(name=PrefixedName("b1"))

    world2 = World()
    b2 = Body(name=PrefixedName("b2"))

    with world2.modify_world():
        world2.add_body(b2)

    b1_C_b2 = FixedConnection(parent=b1, child=b2)
    world1.merge_world(world2, b1_C_b2)

    assert b1_C_b2 in world1.connections
    assert b2 in world1.bodies


def test_merge_world_with_connection_worldless_child():
    world1 = World()
    b1 = Body(name=PrefixedName("b1"))

    world2 = World()
    b2 = Body(name=PrefixedName("b2"))

    with world2.modify_world():
        world2.add_body(b2)

    b2_C_b1 = FixedConnection(parent=b2, child=b1)
    world1.merge_world(world2, b2_C_b1)

    assert b2_C_b1 in world1.connections
    assert b2 in world1.bodies


def test_merge_with_pose(world_setup, pr2_world_copy):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_copy.get_kinematic_structure_entity_by_name("base_link")
    r_gripper_tool_frame = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_copy.get_connection(
        torso_lift_link,
        pr2_world_copy.get_kinematic_structure_entity_by_name("r_shoulder_pan_link"),
    )

    pose = np.eye(4)
    pose[0, 3] = 1.0  # Translate along x-axis

    world.merge_world_at_pose(pr2_world_copy, HomogeneousTransformationMatrix(pose))

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert torso_lift_link in world.bodies
    assert r_shoulder_pan_joint in world.connections
    assert world.compute_forward_kinematics_np(world.root, base_link)[
        0, 3
    ] == pytest.approx(1.0, abs=1e-6)


def test_merge_with_pose_rotation(world_setup, pr2_world_copy):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_copy.get_kinematic_structure_entity_by_name("base_link")
    r_gripper_tool_frame = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_copy.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_copy.get_connection(
        torso_lift_link,
        pr2_world_copy.get_kinematic_structure_entity_by_name("r_shoulder_pan_link"),
    )

    # Rotation is 90 degrees around z-axis, translation is 1 along x-axis
    pose = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    world.merge_world_at_pose(pr2_world_copy, HomogeneousTransformationMatrix(pose))

    base_footprint = world.get_kinematic_structure_entity_by_name("base_footprint")

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert torso_lift_link in world.bodies
    assert r_shoulder_pan_joint in world.connections
    fk_base = world.compute_forward_kinematics_np(world.root, base_footprint)
    assert fk_base[0, 3] == pytest.approx(1.0, abs=1e-6)
    assert fk_base[1, 3] == pytest.approx(1.0, abs=1e-6)
    assert fk_base[2, 3] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_almost_equal(
        RotationMatrix.from_rpy(0, 0, np.pi / 2).to_np()[:3, :3],
        fk_base[:3, :3],
        decimal=6,
    )


def test_merge_in_empty_world(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup

    empty_world = World()

    assert empty_world.root is None
    empty_world.merge_world(world)
    assert empty_world.root is not None


def test_remove_connection(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection = world.get_connection(l1, l2)
    num_dofs = len(world.degrees_of_freedom)
    with world.modify_world():
        world.remove_connection(connection)
        world.remove_kinematic_structure_entity(l2)
    assert connection not in world.connections
    # dof should still exist because it was a mimic connection, so the number didn't change.
    assert num_dofs == len(world.degrees_of_freedom)

    with world.modify_world():
        world.remove_connection(world.get_connection(r1, r2))
        new_connection = FixedConnection(r1, r2)
        world.add_connection(new_connection)

    with pytest.raises(AssertionError):
        with world.modify_world():
            # if you remove a connection, the child must be connected some other way or deleted
            world.remove_connection(world.get_connection(r1, r2))


def test_remove_branch_from_world(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    parent_connection = world.get_connection(bf, r1)
    inner_connection = world.get_connection(r1, r2)

    world.remove_branch_from_world(r1)

    # The whole r-branch and the connection attaching it to its parent are gone.
    assert r1 not in world.kinematic_structure_entities
    assert r2 not in world.kinematic_structure_entities
    assert parent_connection not in world.connections
    assert inner_connection not in world.connections

    # The rest of the world is untouched.
    for remaining in (world.root, bf, l1, l2):
        assert remaining in world.kinematic_structure_entities


def test_kinematic_structure_entity_hash(world_setup):
    _, l1, _, _, _, _ = world_setup
    assert hash(l1) == hash(l1.id)


def test_connection_hash(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    for connection in world.connections:
        print(type(connection))
        assert hash(connection) == hash((connection.parent, connection.child))


def test_degree_of_freedom_hash(world_setup):
    world, _, _, _, _, _ = world_setup
    dof = world.degrees_of_freedom[0]
    assert hash(dof) == hash(dof.id)


def test_copy_world(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    assert l2 in world_copy.bodies
    l2_copy = world_copy.get_kinematic_structure_entity_by_id(l2.id)
    assert id(l2) != id(l2_copy)
    original_bf_con = bf.parent_connection
    assert original_bf_con in world_copy.connections
    copy_connection = world_copy.get_connection(
        original_bf_con.parent, original_bf_con.child
    )
    assert id(copy_connection) != id(original_bf_con)
    bf.parent_connection.origin = HomogeneousTransformationMatrix(
        data=np.array([[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        reference_frame=bf.parent_connection.parent,
        child_frame=bf,
    )
    assert (
        float(
            world_copy.get_kinematic_structure_entity_by_name(
                "bf"
            ).global_transform.to_np()[0, 3]
        )
        == 0.0
    )
    assert float(bf.global_transform.to_np()[0, 3]) == 1.5

    assert set(world_copy._world_entity_hash_table.keys()) == set(
        world._world_entity_hash_table.keys()
    )


def test_copy_big_world():
    pr2_world = URDFParser.from_file(
        file_path="package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    PR2.from_world(pr2_world)
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "coraplex",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()

    apartment_world.merge_world(pr2_world)
    apartment_world_copy = deepcopy(apartment_world)

    assert set(apartment_world._world_entity_hash_table.keys()) == set(
        apartment_world_copy._world_entity_hash_table.keys()
    )


def test_copy_world_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()
    world_copy = deepcopy(world)

    assert world.get_connection(r1, r2).position == 1.0
    assert world_copy.get_connection(r1, r2).position == 1.0


def test_merge_state_skips_dofs_missing_in_self():
    world = World()
    shared_dof, other_only_dof = uuid4(), uuid4()

    state_self = WorldState(_world=world)
    state_self._add_dof(shared_dof)
    state_other = WorldState(_world=world)
    state_other._add_dof(shared_dof)
    state_other._add_dof(other_only_dof)
    state_other[shared_dof].position = 4.2
    state_other[other_only_dof].position = 13.37

    with pytest.raises(DofNotInWorldStateError):
        state_self.merge_state(state_other)


def test_world_state_keys_does_not_expose_internal_list(world_setup):
    world, *_ = world_setup
    state = world.state
    length_before = len(state)

    keys = state.keys()
    try:
        keys.append(uuid4())
    except AttributeError:
        pass

    assert len(state) == length_before


def test_world_state_equality_is_order_independent():
    world = World()  # only needed for the lock
    dof_a, dof_b = uuid4(), uuid4()

    state_1 = WorldState(_world=world)
    state_1._add_dof(dof_a)
    state_1._add_dof(dof_b)
    state_2 = WorldState(_world=world)
    state_2._add_dof(dof_b)
    state_2._add_dof(dof_a)

    for state in (state_1, state_2):
        state[dof_a].position = 1.0
        state[dof_b].position = 2.0

    assert state_1 == state_2


def test_world_str_contains_class_name():
    """
    world.py:486 uses self.__class__.name (the dataclass field default, None) instead of
    the class name, so every world stringifies as 'None v...'.
    """
    world = World()
    assert "World" in str(world)


def test_world_state_item_not_set_yet(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    new_dof = DegreeOfFreedom(name=PrefixedName("new_dof"))

    with pytest.raises(DofNotInWorldStateError):
        world.state[new_dof.id] = np.asarray([0, 0, 0, 0])

    with world.modify_world():
        world.add_degree_of_freedom(new_dof)
        world.state[new_dof.id] = np.asarray([1, 0, 0, 0])
        assert world.state[new_dof.id].position == 1.0


def test_match_index(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    for body in world.bodies:
        new_body = world_copy.get_kinematic_structure_entity_by_id(body.id)
        assert body.index == new_body.index


def test_copy_dof(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    for dof in world.degrees_of_freedom:
        new_dof = world_copy.get_degree_of_freedom_by_id(dof.id)
        assert dof.id == new_dof.id
        assert dof.limits.lower == new_dof.limits.lower
        assert dof.limits.upper == new_dof.limits.upper


def test_copy_pr2_world_state_reset(pr2_world_state_reset):
    pr2_world_state_reset.state[
        pr2_world_state_reset.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position = 0.3
    pr2_world_state_reset.notify_state_change()
    pr2_copy = deepcopy(pr2_world_state_reset)


def test_copy_pr2_world_state_reset_connection_origin(pr2_world_state_reset):
    pr2_world_state_reset.notify_state_change()
    pr2_copy = deepcopy(pr2_world_state_reset)

    for body in pr2_world_state_reset.bodies:
        pr2_body = pr2_world_state_reset.get_kinematic_structure_entity_by_id(body.id)
        pr2_copy_body = pr2_copy.get_kinematic_structure_entity_by_id(body.id)
        np.testing.assert_array_almost_equal(
            pr2_body.global_transform.to_np(),
            pr2_copy_body.global_transform.to_np(),
            decimal=4,
        )


def test_world_same_body_but_different_in_memory(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    for body in world_copy.bodies:
        assert body in world.bodies
        original_body = world.get_kinematic_structure_entity_by_id(body.id)
        assert id(body) != id(original_body)
    for connection in world_copy.connections:
        assert connection in world.connections
        original_connection = world.get_connection(connection.parent, connection.child)
        assert id(connection) != id(original_connection)
    for dof_id in world_copy.state:
        copy_dof = world_copy.get_degree_of_freedom_by_id(dof_id)
        assert copy_dof in world.degrees_of_freedom
        original_dof = world.get_degree_of_freedom_by_id(dof_id)
        assert id(copy_dof) != id(original_dof)


def test_copy_pr2(pr2_world_state_reset):
    pr2_world_state_reset.state[
        pr2_world_state_reset.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position = 0.3
    pr2_world_state_reset.notify_state_change()
    pr2_copy = deepcopy(pr2_world_state_reset)
    assert pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "head_tilt_link"
    ).global_transform.to_np()[2, 3] == pytest.approx(1.472, abs=1e-3)
    assert pr2_copy.get_kinematic_structure_entity_by_name(
        "head_tilt_link"
    ).global_transform.to_np()[2, 3] == pytest.approx(1.472, abs=1e-3)


def test_copy_connections(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    for connection in pr2_world_state_reset.connections:
        pr2_copy_connection = pr2_copy.get_connection_by_name(connection.name)
        assert connection.name == pr2_copy_connection.name
        np.testing.assert_array_almost_equal(
            connection.origin.to_np(), pr2_copy_connection.origin.to_np(), decimal=3
        )
    pr2_copy.state[
        pr2_copy.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position = 0.3
    pr2_copy.notify_state_change()
    original_torso_state = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.TORSO_LIFT
    ).origin
    copied_and_updated_torso_state = pr2_copy.get_connection_by_name(
        PR2Joint.TORSO_LIFT
    ).origin

    assert_raises(
        AssertionError,
        np.testing.assert_array_almost_equal,
        original_torso_state,
        copied_and_updated_torso_state,
    )


def test_omnidrive_translation_dofs_get_translation_limits():
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    base = Body(name=PrefixedName("base", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(base)
        drive = OmniDrive.create_with_dofs(
            world=world,
            parent=root,
            child=base,
            translation_velocity_limits=0.6,
            rotation_velocity_limits=0.5,
        )
        world.add_connection(drive)

    assert drive.x_velocity.limits.upper.velocity == pytest.approx(0.6)
    assert drive.y_velocity.limits.upper.velocity == pytest.approx(0.6)
    assert drive.yaw.limits.upper.velocity == pytest.approx(0.5)


def test_bug_05_has_collision_respects_volume_threshold():
    """
    world_entity.py:487-497: Body.has_collision documents and accepts volume/ surface
    thresholds but ignores them entirely.
    """
    tiny_body = Body(name=PrefixedName("tiny", prefix="review"))
    collision = Box(
        scale=Scale(0.001, 0.001, 0.001),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=tiny_body),
    )
    tiny_body.collision = ShapeCollection([collision], reference_frame=tiny_body)
    # volume = 1e-9 m^3, far below the documented default threshold of 1.001e-6 m^3
    assert tiny_body.has_collision() is False


def create_body_with_box(name: str, scale: Scale) -> Body:
    """
    A body whose only collision shape is a box of the given scale.
    """
    body = Body(name=PrefixedName(name, prefix="review"))
    collision = Box(
        scale=scale,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
    )
    body.collision = ShapeCollection([collision], reference_frame=body)
    return body


def test_shape_of_sufficient_volume_needs_no_mesh(monkeypatch):
    """
    A shape big enough to be checked is recognised from its own volume, without building
    the mesh that only stands in for it.
    """
    body = create_body_with_box("chunky", Scale(0.1, 0.1, 0.1))

    def refuse_to_build_mesh(self):
        raise AssertionError("the mesh was built to answer has_collision")

    monkeypatch.setattr(Box, "mesh", property(refuse_to_build_mesh))

    assert body.has_collision() is True


def test_flat_shape_is_recognised_by_its_surface():
    """
    A shape without volume still counts as collision geometry if its surface is large
    enough, which is the case the surface threshold exists for.
    """
    flat_body = create_body_with_box("flat", Scale(1.0, 1.0, 0.0))

    assert flat_body.collision[0].volume == 0
    assert flat_body.has_collision() is True


def test_copy_two_times(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    pr2_copy_2 = deepcopy(pr2_copy)
    pr2_copy_3 = deepcopy(pr2_copy_2)
    for connection in pr2_world_state_reset.connections:
        pr2_copy_connection = pr2_copy_2.get_connection_by_name(connection.name)
        pr2_copy_3_connection = pr2_copy_3.get_connection_by_name(connection.name)
        assert connection.name == pr2_copy_connection.name
        assert connection.name == pr2_copy_3_connection.name


def test_copy_drawer(apartment_world_copy):
    [handle] = apartment_world_copy.get_semantic_annotations_by_type(Handle)
    [drawer] = apartment_world_copy.get_semantic_annotations_by_type(Drawer)

    apartment_copy = deepcopy(apartment_world_copy)
    copied_handle = apartment_copy.get_semantic_annotation_by_name(handle.name)
    copied_drawer = apartment_copy.get_semantic_annotation_by_name(drawer.name)
    assert copied_handle == handle
    assert copied_drawer == drawer


def test_copy_id(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    for body in pr2_world_state_reset.bodies:
        assert body.id == pr2_copy.get_kinematic_structure_entity_by_name(body.name).id


def test_world_entity_with_class_id():
    @dataclass(eq=False)
    class A(WorldEntityWithClassBasedID): ...

    @dataclass(eq=False)
    class B(WorldEntityWithClassBasedID): ...

    a_1 = A()
    a_2 = A()
    b_1 = B()
    assert a_1 == a_2
    assert a_1 != b_1


def test_copy_reference_frames_shape(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    for body in pr2_world_state_reset.bodies:
        copy_body = pr2_copy.get_kinematic_structure_entity_by_name(body.name)
        if len(body.collision.shapes) > 0:
            assert (
                body.collision.shapes[0].origin.reference_frame._world
                is not copy_body.collision.shapes[0].origin.reference_frame._world
            )


def test_set_omni_after_copy(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    assert (
        type(pr2_copy.get_body_by_name("base_footprint").parent_connection) == OmniDrive
    )

    pr2_copy.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(10, 10, 0)
    )
    pr2_copy.notify_state_change()

    np.testing.assert_array_almost_equal(
        pr2_copy.get_body_by_name("base_footprint")
        .global_transform.to_position()
        .to_np(),
        np.array([10.0, 10.0, 0.0, 1.0]),
    )


def test_add_entity_with_duplicate_name(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    body_duplicate = Body(name=PrefixedName("l1"))
    connection = FixedConnection(parent=l1, child=body_duplicate)
    with world.modify_world():
        world.add_kinematic_structure_entity(body_duplicate)
        world.add_connection(connection)


def test_overwrite_dof_limits(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connections_by_type(
        PrismaticConnection
    )[0]
    assert connection.dof.limits.lower.velocity == -1
    assert connection.dof.limits.upper.velocity == 1

    new_limits = DerivativeMap(0.69, 0.42, 1337, 23)

    connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits * -1, new_upper_limits=new_limits
    )
    assert connection.dof.limits.lower.position == -new_limits.position
    assert connection.dof.limits.upper.position == new_limits.position
    assert connection.dof.limits.lower.velocity == -new_limits.velocity
    assert connection.dof.limits.upper.velocity == new_limits.velocity
    assert connection.dof.limits.lower.acceleration == -new_limits.acceleration
    assert connection.dof.limits.upper.acceleration == new_limits.acceleration
    assert connection.dof.limits.lower.jerk == -new_limits.jerk
    assert connection.dof.limits.upper.jerk == new_limits.jerk

    new_limits2 = DerivativeMap(3333, 3333, 3333, 3333)

    connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits2 * -1, new_upper_limits=new_limits2
    )
    assert connection.dof.limits.lower.position == -new_limits.position
    assert connection.dof.limits.upper.position == new_limits.position
    assert connection.dof.limits.lower.velocity == -new_limits.velocity
    assert connection.dof.limits.upper.velocity == new_limits.velocity
    assert connection.dof.limits.lower.acceleration == -new_limits.acceleration
    assert connection.dof.limits.upper.acceleration == new_limits.acceleration
    assert connection.dof.limits.lower.jerk == -new_limits.jerk
    assert connection.dof.limits.upper.jerk == new_limits.jerk


def test_overwrite_dof_limits_mimic(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connections_by_type(
        PrismaticConnection
    )[0]
    with world.modify_world():
        body = Body(name=PrefixedName("muh"))
        mimic_connection = PrismaticConnection(
            parent=bf,
            child=body,
            offset=23,
            multiplier=-2,
            axis=Vector3(0, 0, 1),
            raw_dof=connection.raw_dof,
        )
        world.add_body(body)
        world.add_connection(mimic_connection)

    # when the multiplier is negative, the vel limits shouldn't be swapped
    assert np.isclose(
        mimic_connection.dof.limits.lower.velocity,
        connection.dof.limits.lower.velocity * 2,
    )
    assert np.isclose(
        mimic_connection.dof.limits.upper.velocity,
        connection.dof.limits.upper.velocity * 2,
    )

    new_limits = DerivativeMap(0.69, 0.42, 1337, 23)

    with pytest.raises(UsageError):
        mimic_connection.dof._overwrite_dof_limits(
            new_lower_limits=new_limits * -1, new_upper_limits=new_limits
        )

    mimic_connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits * -1, new_upper_limits=new_limits
    )

    # Check that limits are correctly applied with negative multiplier and offset
    # Position limits: swapped due to negative multiplier, then scaled and offset applied
    # Lower becomes: new_limits.position * (-2) + 23 = 0.69 * (-2) + 23 = -1.38 + 23 = 21.62
    # Upper becomes: (new_limits * -1).position * (-2) + 23 = -0.69 * (-2) + 23 = 1.38 + 23 = 24.38
    assert np.isclose(mimic_connection.dof.limits.lower.position, 21.62)
    assert np.isclose(mimic_connection.dof.limits.upper.position, 24.38)

    # Velocity limits: only multiplier applied (no offset), but absolute value for limits
    # Since we're dealing with limits, velocity should use abs(multiplier) = 2
    assert np.isclose(
        mimic_connection.dof.limits.lower.velocity, (new_limits * -1).velocity * 2
    )
    assert np.isclose(
        mimic_connection.dof.limits.upper.velocity, new_limits.velocity * 2
    )

    # Acceleration limits: only multiplier applied (no offset), absolute value for limits
    assert np.isclose(
        mimic_connection.dof.limits.lower.acceleration,
        (new_limits * -1).acceleration * 2,
    )
    assert np.isclose(
        mimic_connection.dof.limits.upper.acceleration, new_limits.acceleration * 2
    )

    # Jerk limits: only multiplier applied (no offset), absolute value for limits
    assert np.isclose(
        mimic_connection.dof.limits.lower.jerk, (new_limits * -1).jerk * 2
    )
    assert np.isclose(mimic_connection.dof.limits.upper.jerk, new_limits.jerk * 2)

    # limits are only applied if the new ones are lower
    new_limits2 = DerivativeMap(3333, 3333, 3333, 3333)

    mimic_connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits2 * -1, new_upper_limits=new_limits2
    )

    assert np.isclose(mimic_connection.dof.limits.lower.position, 21.62)
    assert np.isclose(mimic_connection.dof.limits.upper.position, 24.38)

    assert np.isclose(
        mimic_connection.dof.limits.lower.velocity, (new_limits * -1).velocity * 2
    )
    assert np.isclose(
        mimic_connection.dof.limits.upper.velocity, new_limits.velocity * 2
    )

    assert np.isclose(
        mimic_connection.dof.limits.lower.acceleration,
        (new_limits * -1).acceleration * 2,
    )
    assert np.isclose(
        mimic_connection.dof.limits.upper.acceleration, new_limits.acceleration * 2
    )

    assert np.isclose(
        mimic_connection.dof.limits.lower.jerk, (new_limits * -1).jerk * 2
    )
    assert np.isclose(mimic_connection.dof.limits.upper.jerk, new_limits.jerk * 2)


def test_missing_world_modification_context(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(MissingWorldModificationContextError):
        world.add_semantic_annotation(Handle(root=l1))


def test_dof_removal_simple():
    world = World()
    body1 = Body(name=PrefixedName("body1"))
    body2 = Body(name=PrefixedName("body2"))
    with world.modify_world():
        c = RevoluteConnection.create_with_dofs(
            world=world, parent=body1, child=body2, axis=Vector3.Z()
        )
        world.add_connection(c)
    with world.modify_world():
        world.remove_connection(c)

        c2 = RevoluteConnection.create_with_dofs(
            world=world, parent=body1, child=body2, axis=Vector3.Z()
        )
        world.add_connection(c2)


def test_dof_removal():
    world = World()
    body1 = Body(name=PrefixedName("body1"))
    with world.modify_world():
        world.add_body(body1)

    world2 = World()
    body2 = Body(name=PrefixedName("body2"))
    with world2.modify_world():
        world2.add_body(body2)

    world.merge_world(world2)

    with world.modify_world():
        new_body2 = world.get_kinematic_structure_entity_by_id(body2.id)
        world.remove_connection(new_body2.parent_connection)

        c_root_bf = OmniDrive.create_with_dofs(
            parent=body1, child=new_body2, world=world
        )
        world.add_connection(c_root_bf)


def test_actuators(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup

    connection: PrismaticConnection = world.get_connection(r1, r2)
    dof = connection.dof
    actuator = Actuator(
        name=PrefixedName("actuator"),
    )
    actuator.add_dof(dof)
    with world.modify_world():
        world.add_actuator(actuator)

    assert actuator in world.actuators
    assert world.get_actuator_by_id(actuator.id) == actuator


def test_add_body_hash():
    world = World()
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_body(body)

    assert hash(body) in world._world_entity_hash_table

    with world.modify_world():
        world.remove_kinematic_structure_entity(body)
    assert hash(body) not in world._world_entity_hash_table


def test_world_state_trajectory(world_setup, tmp_path):
    world, l1, l2, bf, r1, r2 = world_setup

    time = 1337.0

    connection: PrismaticConnection = world.get_connection(r1, r2)
    dof_uuid = connection.dof.id

    traj = WorldStateTrajectory.from_world_state(world.state, time)
    cmd = np.array([100.0, 0, 0, 0, 0, 0, 0, 0])
    dt = 0.1

    # Verify initial state
    assert len(traj.times) == 1
    assert traj.times[0] == time
    assert traj.data.shape[0] == 1  # One timestep
    assert traj.data.shape[1] == 4  # Four derivatives (pos, vel, acc, jerk)
    assert traj.data.shape[2] == len(world.state)  # Number of DOFs

    # Store initial state for comparison
    initial_state = deepcopy(world.state)

    for i in range(10):
        time += dt
        world.apply_control_commands(cmd, dt, Derivatives.jerk)
        traj.append(world.state, time)

    # Verify final trajectory structure
    assert len(traj.times) == 11  # Initial + 10 appended states
    assert traj.data.shape[0] == 11  # 11 timesteps
    assert traj.data.shape[1] == 4  # Four derivatives
    assert traj.data.shape[2] == len(world.state)  # Number of DOFs

    # Verify time progression
    expected_times = np.array([1337.0 + i * dt for i in range(11)])
    np.testing.assert_allclose(traj.times, expected_times)

    # Verify that the trajectory captures state changes
    # The first DOF should have changed due to jerk command
    assert not np.allclose(traj.data[0, :, 0], traj.data[-1, :, 0])  # First DOF changed
    assert np.allclose(
        traj.data[0, :, 1:], initial_state._data[:, 1:]
    )  # Other DOFs unchanged initially

    plotter = WorldStateTrajectoryPlotter()
    plotter.world_state_trajectory = traj
    plotter.plot_trajectory(str(tmp_path / "traj.pdf"))

    # Verify world version consistency
    assert traj._world_version == world.get_world_model_manager().version

    # Verify that trajectory data matches current world state
    np.testing.assert_allclose(traj.data[-1, :, :], world.state._data)

    # verify that the state increased on each step
    previous = initial_state[dof_uuid]
    for time, data in list(traj.items())[1:]:
        next = data[dof_uuid]
        assert next.position > previous.position
        assert next.velocity > previous.velocity
        assert next.acceleration > previous.acceleration
        previous = next

    with pytest.raises(NonMonotonicTimeError):
        traj.append(world.state, time - dt)

    world._notify_model_change()
    with pytest.raises(WrongWorldModelVersion):
        traj.append(world.state, time + dt)


def test_merge_into_empty_world(world_setup):
    world, _, _, _, _, _ = world_setup
    world2 = deepcopy(world)
    with world2.modify_world():
        world2.clear()
    world2.merge_world(world)


def test_reattach_child_to_new_parent(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    # Initial state: l2 is child of l1 via PrismaticConnection
    old_child_global_pose = l2.global_transform
    assert l2.parent_connection.parent == l1
    assert isinstance(l2.parent_connection, PrismaticConnection)

    with world.modify_world():
        world.move_branch_with_fixed_connection(new_parent=bf, branch_root=l2)

    # New state: l2 is child of bf via FixedConnection
    assert l2.parent_connection.parent == bf
    assert isinstance(l2.parent_connection, FixedConnection)
    assert l2 in world.compute_child_kinematic_structure_entities(bf)
    assert l2 not in world.compute_child_kinematic_structure_entities(l1)
    new_child_global_pose = l2.global_transform
    assert np.allclose(old_child_global_pose, new_child_global_pose)


def test_move_branch_preserves_connection_type_and_pose():
    """
    move_branch re-parents a branch keeping its connection type and global pose.

    Covers both the FixedConnection branch (which silently passed an invalid ``_world``
    kwarg before) and the Connection6DoF branch.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    new_parent = Body(name=PrefixedName("new_parent"))
    fixed_child = Body(name=PrefixedName("fixed_child"))
    free_child = Body(name=PrefixedName("free_child"))
    with world.modify_world():
        for body in [root, new_parent, fixed_child, free_child]:
            world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=new_parent,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, y=2.0, yaw=0.5
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=fixed_child,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.3, z=0.4
                ),
            )
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=free_child, world=world)
        )

    # FixedConnection branch (this is the path the `_world=self` bug used to break).
    fixed_child_pose = fixed_child.global_transform
    with world.modify_world():
        world.move_branch(fixed_child, new_parent)
    assert fixed_child.parent_kinematic_structure_entity == new_parent
    assert isinstance(fixed_child.parent_connection, FixedConnection)
    assert np.allclose(fixed_child.global_transform, fixed_child_pose)

    # Connection6DoF branch: type preserved, pose preserved.
    free_child_pose = free_child.global_transform
    with world.modify_world():
        world.move_branch(free_child, new_parent)
    assert free_child.parent_kinematic_structure_entity == new_parent
    assert isinstance(free_child.parent_connection, Connection6DoF)
    assert np.allclose(free_child.global_transform, free_child_pose)


def test_memoization_clears_only_last_modification_block():
    world = World()
    b1 = Body(name=PrefixedName("b1"))
    with world.modify_world():
        world.add_body(b1)

    b2 = Body(name=PrefixedName("b2"))
    b1_C_b2 = FixedConnection(parent=b2, child=b1)

    with world.modify_world():

        assert world.root == b1

        with world.modify_world():
            world.add_body(b2)

        assert world.root == b1

        world.add_connection(b1_C_b2)

    assert world.root == b2


def test_move_branch_offline_preserves_connection_type_and_pose():
    """
    The offline (enable_unsafe_inside_world_block) path of move_branch must, like the
    online path, preserve a FixedConnection and a Connection6DoF and keep the global
    pose.

    The offline path is what the semantic-annotation mounts use, and it runs inside an
    already-open modification block.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    new_parent = Body(name=PrefixedName("new_parent"))
    fixed_child = Body(name=PrefixedName("fixed_child"))
    free_child = Body(name=PrefixedName("free_child"))
    with world.modify_world():
        for body in [root, new_parent, fixed_child, free_child]:
            world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=new_parent,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, y=2.0, yaw=0.5
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=fixed_child,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.3, z=0.4
                ),
            )
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=free_child, world=world)
        )

    fixed_child_pose = fixed_child.global_transform
    free_child_pose = free_child.global_transform
    with world.modify_world():
        world.move_branch(
            fixed_child, new_parent, enable_unsafe_inside_world_block=True
        )
        world.move_branch(free_child, new_parent, enable_unsafe_inside_world_block=True)

    assert fixed_child.parent_kinematic_structure_entity == new_parent
    assert isinstance(fixed_child.parent_connection, FixedConnection)
    assert np.allclose(fixed_child.global_transform, fixed_child_pose)

    assert free_child.parent_kinematic_structure_entity == new_parent
    assert isinstance(free_child.parent_connection, Connection6DoF)
    assert np.allclose(free_child.global_transform, free_child_pose)


def test_move_branch_preserves_active_connection(world_setup):
    """
    move_branch keeps an active 1-DOF connection (e.g. a RevoluteConnection) intact - same type and
    same degree of freedom - instead of collapsing it to a FixedConnection, while preserving the pose.
    """
    world, l1, l2, bf, r1, r2 = world_setup
    old_connection = r2.parent_connection
    assert isinstance(old_connection, RevoluteConnection)
    old_dof_id = old_connection.dof.id
    old_pose = r2.global_transform

    with world.modify_world():
        world.move_branch(r2, bf)

    assert r2.parent_kinematic_structure_entity == bf
    assert isinstance(r2.parent_connection, RevoluteConnection)
    assert r2.parent_connection.dof.id == old_dof_id
    assert np.allclose(r2.global_transform, old_pose)


# %% re-parenting a free-floating branch


def create_world_with_free_floating_child() -> tuple[World, Body, Body]:
    """
    Builds a world where ``free_child`` hangs off the root by a :class:`Connection6DoF`
    and ``new_parent`` sits elsewhere under the root.

    :return: The world, the free-floating child and the body to re-parent it onto.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    new_parent = Body(name=PrefixedName("new_parent"))
    free_child = Body(name=PrefixedName("free_child"))
    with world.modify_world():
        for body in [root, new_parent, free_child]:
            world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=new_parent,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, y=2.0, yaw=0.5
                ),
            )
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=free_child, world=world)
        )
    free_child.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.3, y=-0.7, z=0.2, yaw=1.1, reference_frame=root
    )
    return world, free_child, new_parent


def test_move_branch_preserves_free_connection_degrees_of_freedom():
    """
    move_branch keeps a Connection6DoF's degrees of freedom instead of replacing them
    with fresh ones, while preserving the branch's global pose.

    Re-using the degrees of freedom is what keeps the world state layout stable across
    the move.
    """
    world, free_child, new_parent = create_world_with_free_floating_child()
    old_dof_ids = [dof.id for dof in free_child.parent_connection.passive_dofs]
    old_pose = free_child.global_transform

    with world.modify_world():
        world.move_branch(free_child, new_parent)

    assert free_child.parent_kinematic_structure_entity == new_parent
    assert isinstance(free_child.parent_connection, Connection6DoF)
    assert [dof.id for dof in free_child.parent_connection.passive_dofs] == old_dof_ids
    assert np.allclose(free_child.global_transform, old_pose)


def test_move_branch_keeps_world_state_array_identity():
    """
    move_branch of a Connection6DoF branch must not reallocate the world state array.

    Compiled functions bind memory views of that array, so replacing it silently
    detaches them from the live state.
    """
    world, free_child, new_parent = create_world_with_free_floating_child()
    state_data_before_move = world.state._data

    with world.modify_world():
        world.move_branch(free_child, new_parent)

    assert world.state._data is state_data_before_move


def test_move_branch_resets_free_connection_derivatives():
    """
    move_branch leaves the re-used degrees of freedom without any residual motion.

    The world integrates every degree of freedom, including passive ones, so a leftover
    velocity would make the re-parented branch drift away from its new parent.
    """
    world, free_child, new_parent = create_world_with_free_floating_child()
    for dof in free_child.parent_connection.passive_dofs:
        world.state[dof.id].velocity = 0.3
        world.state[dof.id].acceleration = 0.2
        world.state[dof.id].jerk = 0.1

    with world.modify_world():
        world.move_branch(free_child, new_parent)

    for dof in free_child.parent_connection.passive_dofs:
        assert world.state[dof.id].velocity == 0
        assert world.state[dof.id].acceleration == 0
        assert world.state[dof.id].jerk == 0


def test_reset_state_context(pr2_world_state_reset):
    state_copy = pr2_world_state_reset.state._data.copy()
    with pr2_world_state_reset.reset_state_context():
        pr2_world_state_reset.get_body_by_name(
            "base_footprint"
        ).parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            10, 10, 0
        )
    assert np.allclose(state_copy, pr2_world_state_reset.state._data)


def test_copy_for_world():

    w1 = World()
    w2 = World()
    b1_uuid = UUID(int=1)
    b1_w1 = Body(
        name=PrefixedName("b1"),
        collision=ShapeCollection([Box(scale=Scale())]),
        id=b1_uuid,
    )

    milk = Milk(root=b1_w1)
    with w1.modify_world():
        w1.add_body(b1_w1)
        w1.add_semantic_annotation(milk)

    b1_w2 = b1_w1.copy_for_world(w2)

    assert b1_w2.id == b1_w1.id
    assert b1_w2.name == b1_w1.name
    assert b1_w2.collision == b1_w1.collision

    with w2.modify_world():
        w2.add_body(b1_w2)
    copied_milk = milk.copy_for_world(w2)

    assert copied_milk.root == b1_w2


def make_bodies(*names: tuple) -> list:
    return [Body(name=PrefixedName(name, prefix)) for name, prefix in names]


def test_suggest_typo_in_string_name():
    bodies = make_bodies(("torso_lift_link", "pr2"), ("head_pan_link", "pr2"))
    suggestions = World._suggest_world_entity_names("torso_lft_link", bodies)
    assert suggestions == [PrefixedName("torso_lift_link", "pr2")]


def test_suggest_typo_in_prefixed_name():
    bodies = make_bodies(("torso_lift_link", "pr2"), ("head_pan_link", "pr2"))
    suggestions = World._suggest_world_entity_names(
        PrefixedName("torso_lft_link", "pr2"), bodies
    )
    assert suggestions == [PrefixedName("torso_lift_link", "pr2")]


def test_suggest_same_bare_name_with_different_prefix():
    bodies = make_bodies(("base_link", "pr2"), ("head_pan_link", "pr2"))
    suggestions = World._suggest_world_entity_names(
        PrefixedName("base_link", "kitchen"), bodies
    )
    assert suggestions == [PrefixedName("base_link", "pr2")]


def test_suggest_returns_all_prefixes_of_exact_bare_name_match():
    bodies = make_bodies(("base_link", "pr2"), ("base_link", "kitchen"))
    suggestions = World._suggest_world_entity_names(
        PrefixedName("base_link", "unknown"), bodies
    )
    assert suggestions == [
        PrefixedName("base_link", "pr2"),
        PrefixedName("base_link", "kitchen"),
    ]


def test_suggest_exact_bare_name_matches_rank_before_fuzzy_matches():
    # the fuzzy candidate comes first in the iterable to prove ordering is by match
    # quality, not iteration order
    bodies = make_bodies(("torsoo", "b"), ("torso", "a"))
    suggestions = World._suggest_world_entity_names(PrefixedName("torso", "x"), bodies)
    assert suggestions == [PrefixedName("torso", "a"), PrefixedName("torsoo", "b")]


def test_suggest_fuzzy_match_expands_to_all_prefixes():
    bodies = make_bodies(("torso_lift_link", "pr2"), ("torso_lift_link", "pr2_copy"))
    suggestions = World._suggest_world_entity_names("torso_lft_link", bodies)
    assert suggestions == [
        PrefixedName("torso_lift_link", "pr2"),
        PrefixedName("torso_lift_link", "pr2_copy"),
    ]


def test_suggest_no_duplicates_when_exact_match_is_also_fuzzy_match():
    bodies = make_bodies(("torso", "pr2"))
    suggestions = World._suggest_world_entity_names(
        PrefixedName("torso", "wrong"), bodies
    )
    assert suggestions == [PrefixedName("torso", "pr2")]


def test_suggest_respects_default_max_suggestions():
    bodies = make_bodies(*[("base_link", prefix) for prefix in "abcde"])
    suggestions = World._suggest_world_entity_names(
        PrefixedName("base_link", "unknown"), bodies
    )
    assert suggestions == [
        PrefixedName("base_link", "a"),
        PrefixedName("base_link", "b"),
        PrefixedName("base_link", "c"),
    ]


def test_suggest_respects_explicit_max_suggestions():
    bodies = make_bodies(*[("base_link", prefix) for prefix in "abcde"])
    suggestions = World._suggest_world_entity_names(
        PrefixedName("base_link", "unknown"), bodies, max_suggestions=2
    )
    assert len(suggestions) == 2


def test_suggest_no_similar_names_returns_empty():
    bodies = make_bodies(("alpha", None), ("beta", None))
    assert World._suggest_world_entity_names("quaternion", bodies) == []


def test_suggest_empty_iterable_returns_empty():
    assert World._suggest_world_entity_names("anything", []) == []
    assert (
        World._suggest_world_entity_names(PrefixedName("anything", "prefix"), []) == []
    )


def test_not_found_error_contains_suggestions(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(WorldEntityNotFoundError) as exc_info:
        world.get_kinematic_structure_entity_by_name("rooot")
    assert exc_info.value.suggestions == [PrefixedName("root", "world")]
    assert "Suggestion: did you mean" in str(exc_info.value)
    assert "world/root" in str(exc_info.value)


def test_not_found_error_with_wrong_prefix_suggests_existing_entity(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(WorldEntityNotFoundError) as exc_info:
        world.get_body_by_name(PrefixedName("l1", "wrong_prefix"))
    assert exc_info.value.suggestions == [l1.name]


def test_not_found_error_without_suggestions(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(WorldEntityNotFoundError) as exc_info:
        world.get_body_by_name("quaternion_flux_capacitor")
    assert exc_info.value.suggestions == []
    assert "Suggestion" not in str(exc_info.value)


def test_not_found_error_suggestions_for_connections(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(WorldEntityNotFoundError) as exc_info:
        world.get_connection_by_name("l1_T_l3")
    assert PrefixedName("l1_T_l2") in exc_info.value.suggestions


def test_clearing_the_world_detaches_connections():
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    child = Body(name=PrefixedName("child", prefix="review"))
    collision = Box(
        scale=Scale(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=child),
    )
    child.collision = ShapeCollection([collision], reference_frame=child)
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(parent=root, child=child)
        world.add_connection(connection)

    with world.modify_world():
        world._clear_world_entities()

    assert connection._world is None


def test_robot_velocity_limit_setup_does_not_touch_environment_joints():

    def _make_box_body(name: str, scale: Scale = Scale(1.0, 1.0, 1.0)) -> Body:
        body = Body(name=PrefixedName(name, prefix="review"))
        collision = Box(
            scale=scale,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
        )
        body.collision = ShapeCollection([collision], reference_frame=body)
        return body

    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    robot_base = _make_box_body("robot_base")
    robot_link = _make_box_body("robot_link")
    drawer_body = _make_box_body("drawer_body")

    env_limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(None, -10.0, None, None),
        upper=DerivativeMap(None, 10.0, None, None),
    )
    with world.modify_world():
        for b in [root, robot_base, robot_link, drawer_body]:
            world.add_kinematic_structure_entity(b)
        world.add_connection(FixedConnection(parent=root, child=robot_base))
        robot_joint = RevoluteConnection.create_with_dofs(
            world=world,
            parent=robot_base,
            child=robot_link,
            axis=Vector3.Z(reference_frame=robot_base),
        )
        world.add_connection(robot_joint)
        drawer_joint = PrismaticConnection.create_with_dofs(
            world=world,
            parent=root,
            child=drawer_body,
            axis=Vector3.X(reference_frame=root),
            dof_limits=env_limits,
        )
        world.add_connection(drawer_joint)

    MinimalRobot.from_branch_in_world(robot_base)

    # the environment joint does not belong to the robot and must keep its limits
    assert drawer_joint.raw_dof.limits.upper.velocity == pytest.approx(10.0)
    assert drawer_joint.raw_dof.limits.lower.velocity == pytest.approx(-10.0)


def test_get_semantic_annotation_by_id_raises_package_exception():
    world = World()
    with pytest.raises(WorldEntityNotFoundError):
        world.get_semantic_annotation_by_id(uuid4())


def test_failed_add_without_context_does_not_brick_the_world():
    world = World()
    body = Body(name=PrefixedName("body", prefix="review"))

    with pytest.raises(MissingWorldModificationContextError):
        world.add_kinematic_structure_entity(body)

    with world.modify_world():
        world.add_kinematic_structure_entity(body)

    assert body in world.bodies


def test_hash_table_lookup_survives_annotation_mutation():

    @dataclass(eq=False)
    class ReviewAnnotation(SemanticAnnotation):
        """
        Semantic annotation with a mutable entity list, for hash-stability tests.
        """

        parts: list[Body] = field(default_factory=list)

    def _make_box_body(name: str, scale: Scale = Scale(1.0, 1.0, 1.0)) -> Body:
        body = Body(name=PrefixedName(name, prefix="review"))
        collision = Box(
            scale=scale,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
        )
        body.collision = ShapeCollection([collision], reference_frame=body)
        return body

    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    child = _make_box_body("child")
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(parent=root, child=child)
        world.add_connection(connection)

    extra = _make_box_body("extra")
    with world.modify_world():
        world.add_kinematic_structure_entity(extra)
        world.add_connection(FixedConnection(parent=root, child=extra))
        annotation = ReviewAnnotation(
            name=PrefixedName("annotation", prefix="review"), parts=[child]
        )
        world.add_semantic_annotation(annotation)

    annotation.parts.append(extra)

    assert world._world_entity_hash_table.get(hash(annotation)) is annotation


def test_validation_still_works_with_python_optimize_flag():
    snippet = (
        "from semantic_digital_twin.world import World\n"
        "from semantic_digital_twin.world_description.world_entity import Body\n"
        "from semantic_digital_twin.datastructures.prefixed_name import PrefixedName\n"
        "world = World()\n"
        "# build an invalid world (two disconnected roots) behind the back of the\n"
        "# modification machinery, then validate it: validation must fail\n"
        "world.kinematic_structure.add_node(Body(name=PrefixedName('a')))\n"
        "world.kinematic_structure.add_node(Body(name=PrefixedName('b')))\n"
        "world.validate()\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", snippet],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0, (
        "an invalid world (two roots) passed validation under python -O:\n"
        + result.stdout
        + result.stderr
    )


def test_world_entity_equality_works_across_subclasses():
    @dataclass(eq=False)
    class EntityA(WorldEntityWithID):
        pass

    @dataclass(eq=False)
    class EntityB(EntityA):
        pass

    shared_id = uuid4()
    entity_a = EntityA(id=shared_id)
    entity_b = EntityB(id=shared_id)

    assert hash(entity_a) == hash(entity_b)
    assert entity_a == entity_b


def test_reset_state_context_restores_state_on_exception(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(l1, l2)
    connection.position = 0.0

    with pytest.raises(RuntimeError):
        with world.reset_state_context():
            connection.position = 1.0
            raise RuntimeError("simulated user error")

    assert connection.position == pytest.approx(0.0)


def test_broken_world_modification_history_after_exception_in_modification_block_is_raised():
    """
    world.py:205-219: when an exception escapes a modify_world block, the current
    modification block is discarded but the already-applied modifications are not rolled
    back.

    Replay-based operations (deepcopy, sync) then produce a different world than the
    original.
    """
    world = World()
    body_1 = Body(name=PrefixedName("body_1", prefix="review"))
    body_2 = Body(name=PrefixedName("body_2", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body_1)

    with pytest.raises(BrokenWorldModificationHistoryError):
        with world.modify_world():
            world.add_kinematic_structure_entity(body_2)
            world.add_connection(FixedConnection(parent=body_1, child=body_2))
            raise RuntimeError("simulated user error")


def test_memoized_queries_match_graph_after_exception():
    world = World()
    body_1 = Body(name=PrefixedName("body_1", prefix="review"))
    body_2 = Body(name=PrefixedName("body_2", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body_1)

    with pytest.raises(BrokenWorldModificationHistoryError):
        with world.modify_world():
            world.add_kinematic_structure_entity(body_2)
            raise RuntimeError("simulated user error")

    graph_names = {b.name.name for b in world.kinematic_structure.nodes()}
    memoized_names = {b.name.name for b in world.bodies}
    assert memoized_names == graph_names


def test_is_kinematic_structure_entity_in_world_by_name(world_setup):
    world, l1, *_ = world_setup
    assert world.is_kinematic_structure_entity_in_world_by_name("l1")
    assert not world.is_kinematic_structure_entity_in_world_by_name("nonexistent")


# %% moving a branch keeps both worlds replayable


def world_with_branch(branch_length: int) -> Tuple[World, Body]:
    """
    Build a world whose root carries one branch of the requested length.

    :param branch_length: Number of bodies hanging below the branch root.
    :return: The world and the root of its branch.
    """
    world = World(name="source")
    branch_root = Body(name=PrefixedName("branch_root"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(parent=Body(name=PrefixedName("root")), child=branch_root)
        )
        parent = branch_root
        for index in range(branch_length):
            child = Body(name=PrefixedName(f"branch_body_{index}"))
            world.add_connection(FixedConnection(parent=parent, child=child))
            parent = child
    return world, branch_root


def assert_rebuilds_to_same_structure(world: World, other_world: World) -> None:
    """
    Assert that replaying a world's own history onto an empty world reproduces it.

    Both worlds release their entities first, because a world loaded from the database
    replays a history whose entities belong to no world yet, and an entity can belong to
    at most one world.

    :param world: The world whose recorded history is replayed.
    :param other_world: The other side of the move, which must let its entities go.
    """
    entity_names = {entity.name.name for entity in world.kinematic_structure_entities}
    connection_count = len(world.connections)
    recorded_blocks = list(world.get_world_model_manager().model_modification_blocks)

    for world_to_release in (world, other_world):
        with world_to_release.modify_world():
            world_to_release._clear_world_entities()

    rebuilt = World(name=world.name)
    with rebuilt.modify_world():
        for block in recorded_blocks:
            block.apply(rebuilt)

    assert {
        entity.name.name for entity in rebuilt.kinematic_structure_entities
    } == entity_names
    assert len(rebuilt.connections) == connection_count


def test_world_a_multi_body_branch_moved_into_replays_from_its_own_history():
    """
    The world a branch is moved into records a history that starts from nothing.

    Every connection has to leave the world that holds it. Cleaning it up in the new
    world instead opens that world's history with the removal of a connection it never
    added, and the history no longer rebuilds it.
    """
    world, branch_root = world_with_branch(branch_length=2)
    new_world = world.move_branch_to_new_world(branch_root)

    assert_rebuilds_to_same_structure(new_world, world)


def test_world_a_multi_body_branch_moved_out_of_replays_from_its_own_history():
    """
    The world a branch is moved out of keeps a history that still rebuilds it.

    Entities may only leave after the connections naming them, otherwise the recorded
    removal of a connection refers to entities the replay has dropped.
    """
    world, branch_root = world_with_branch(branch_length=2)
    new_world = world.move_branch_to_new_world(branch_root)

    assert_rebuilds_to_same_structure(world, new_world)


def test_world_a_single_body_branch_moved_into_replays_from_its_own_history():
    """
    A branch of a single body detaches its connection before its body, and the world it
    moves into stays replayable.
    """
    world, branch_root = world_with_branch(branch_length=0)
    new_world = world.move_branch_to_new_world(branch_root)

    assert_rebuilds_to_same_structure(new_world, world)


def test_world_a_single_body_branch_moved_out_of_replays_from_its_own_history():
    """
    A branch of a single body leaves behind a world that stays replayable.
    """
    world, branch_root = world_with_branch(branch_length=0)
    new_world = world.move_branch_to_new_world(branch_root)

    assert_rebuilds_to_same_structure(world, new_world)


def test_world_does_not_record_removing_a_connection_it_does_not_hold():
    """
    A world only records the removal of a connection it actually holds.

    Recording it regardless is what lets a history open with the removal of something
    nothing ever added, which no replay can reproduce.
    """
    world, branch_root = world_with_branch(branch_length=1)
    connection = branch_root.parent_connection
    other_world = World(name="other")

    with other_world.modify_world():
        other_world.remove_connection(connection)

    recorded_modifications = [
        modification
        for block in other_world.get_world_model_manager().model_modification_blocks
        for modification in block
    ]
    assert recorded_modifications == []
    assert connection in world.connections


def recorded_modifications(world: World) -> list:
    """
    Flatten every modification recorded so far by a world's model manager.

    :param world: The world whose recorded history is inspected.
    :return: The modifications recorded across all of the world's modification blocks.
    """
    return [
        modification
        for block in world.get_world_model_manager().model_modification_blocks
        for modification in block
    ]


def test_world_does_not_record_removing_a_kinematic_structure_entity_it_does_not_hold():
    """
    A world only records the removal of a kinematic_structure_entity it actually holds.
    """
    world = World(name="source")
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_body(body)
    other_world = World(name="other")

    with other_world.modify_world():
        other_world.remove_kinematic_structure_entity(body)

    assert recorded_modifications(other_world) == []
    assert body in world.kinematic_structure_entities


def test_world_does_not_record_removing_a_degree_of_freedom_it_does_not_hold(
    world_setup,
):
    """
    A world only records the removal of a degree of freedom it actually holds.
    """
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    dof = connection.dof
    other_world = World(name="other")

    with other_world.modify_world():
        other_world.remove_degree_of_freedom(dof)

    assert recorded_modifications(other_world) == []
    assert dof in world.degrees_of_freedom


def test_world_does_not_record_removing_a_semantic_annotation_it_does_not_hold():
    """
    A world only records the removal of a semantic annotation it actually holds.
    """
    world = World(name="source")
    annotation = SemanticAnnotation(name=PrefixedName("annotation"))
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    other_world = World(name="other")

    with other_world.modify_world():
        other_world.remove_semantic_annotation(annotation)

    assert recorded_modifications(other_world) == []
    assert annotation in world.semantic_annotations


def test_world_does_not_record_removing_an_actuator_it_does_not_hold():
    """
    A world only records the removal of an actuator it actually holds.
    """
    world = World(name="source")
    actuator = Actuator(name=PrefixedName("actuator"))
    with world.modify_world():
        world.add_actuator(actuator)
    other_world = World(name="other")

    with other_world.modify_world():
        other_world.remove_actuator(actuator)

    assert recorded_modifications(other_world) == []
    assert actuator in world.actuators


def test_column_indices_locate_the_requested_degrees_of_freedom(world_setup):
    world, *_ = world_setup
    dofs = list(world.degrees_of_freedom)
    for index, dof in enumerate(dofs):
        world.state[dof.id].velocity = 0.1 * (index + 1)

    columns = world.state.column_indices(dofs)

    velocities = world.state.velocities
    assert [velocities[column] for column in columns] == pytest.approx(
        [0.1 * (index + 1) for index in range(len(dofs))]
    )


def test_column_indices_follow_the_requested_order(world_setup):
    world, *_ = world_setup
    dofs = list(world.degrees_of_freedom)

    assert (
        world.state.column_indices(list(reversed(dofs)))
        == world.state.column_indices(dofs)[::-1]
    )


def test_column_indices_of_degree_of_freedom_outside_the_state(world_setup):
    world, *_ = world_setup

    with pytest.raises(DofNotInWorldStateError):
        world.state.column_indices([DegreeOfFreedom(name=PrefixedName("new_dof"))])
