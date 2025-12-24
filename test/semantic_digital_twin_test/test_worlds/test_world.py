from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_raises

from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle

from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.exceptions import (
    AddingAnExistingSemanticAnnotationError,
    DuplicateWorldEntityError,
    DuplicateKinematicStructureEntityError,
    UsageError,
    MissingWorldModificationContextError,
    DofNotInWorldStateError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap

# from semantic_digital_twin.spatial_types.math import rotation_matrix_from_rpy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
)
from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    CollisionCheckingConfig,
    Actuator,
)


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
    c3.origin = transform
    assert np.allclose(world.compute_forward_kinematics_np(world.root, bf), transform)

    world.set_positions_1DOF_connection({c1: 2})
    assert c1.position == 2.0

    transform[0, 3] += c1.position
    assert np.allclose(l2.global_pose.to_np(), transform)


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

    state_memory_id = id(world.state.data)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()
    assert state_memory_id == id(world.state.data)
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
    state_memory_id = id(world.state.data)
    connection: PrismaticConnection = world.get_connection(r1, r2)
    cmd = np.array([100.0, 0, 0, 0, 0, 0, 0, 0])
    dt = 0.1
    world.apply_control_commands(cmd, dt, Derivatives.jerk)
    assert world.state[connection.dof.id].jerk == 100.0
    assert world.state[connection.dof.id].acceleration == 100.0 * dt
    assert world.state[connection.dof.id].velocity == 100.0 * dt * dt
    assert world.state[connection.dof.id].position == 100.0 * dt * dt * dt
    # the state should reuse the same memory
    assert state_memory_id == id(world.state.data)


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
    world.get_connection(world.root, bf).origin = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
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


def test_merge_world(world_setup, pr2_world_state_reset):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_link"
    )
    r_gripper_tool_frame = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_state_reset.get_connection(
        torso_lift_link,
        pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_shoulder_pan_link"
        ),
    )

    l_shoulder_pan_joint = pr2_world_state_reset.get_connection(
        torso_lift_link,
        pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "l_shoulder_pan_link"
        ),
    )

    world.merge_world(pr2_world_state_reset)

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert l_shoulder_pan_joint in world.connections
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world


def test_merge_with_connection(world_setup, pr2_world_state_reset):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_link"
    )
    r_gripper_tool_frame = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_state_reset.get_connection(
        torso_lift_link,
        pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_shoulder_pan_link"
        ),
    )

    pose = np.eye(4)
    pose[0, 3] = 1.0

    origin = HomogeneousTransformationMatrix(pose)

    connection = pr2_world_state_reset.get_connection_by_name(
        "l_gripper_l_finger_joint"
    )
    pr2_world_state_reset.state[connection.dof.id].position = 0.55
    pr2_world_state_reset.notify_state_change()
    expected_fk = pr2_world_state_reset.compute_forward_kinematics(
        connection.parent, connection.child
    ).to_np()

    new_connection = FixedConnection(
        parent=world.root,
        child=pr2_world_state_reset.root,
        parent_T_connection_expression=origin,
    )

    world.merge_world(pr2_world_state_reset, new_connection)
    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert new_connection in world.connections
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world
    assert world.state[connection.dof.id].position == 0.55
    assert world.compute_forward_kinematics_np(world.root, base_link)[
        0, 3
    ] == pytest.approx(1.0, abs=1e-6)
    actual_fk = world.compute_forward_kinematics(
        connection.parent, connection.child
    ).to_np()
    assert np.allclose(actual_fk, expected_fk)


def test_merge_with_pose(world_setup, pr2_world_state_reset):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_link"
    )
    r_gripper_tool_frame = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_state_reset.get_connection(
        torso_lift_link,
        pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_shoulder_pan_link"
        ),
    )

    pose = np.eye(4)
    pose[0, 3] = 1.0  # Translate along x-axis

    world.merge_world_at_pose(
        pr2_world_state_reset, HomogeneousTransformationMatrix(pose)
    )

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world
    assert world.compute_forward_kinematics_np(world.root, base_link)[
        0, 3
    ] == pytest.approx(1.0, abs=1e-6)


def test_merge_with_pose_rotation(world_setup, pr2_world_state_reset):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_link"
    )
    r_gripper_tool_frame = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    torso_lift_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    r_shoulder_pan_joint = pr2_world_state_reset.get_connection(
        torso_lift_link,
        pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_shoulder_pan_link"
        ),
    )
    base_footprint = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
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

    world.merge_world_at_pose(
        pr2_world_state_reset, HomogeneousTransformationMatrix(pose)
    )

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world
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
    bf.parent_connection.origin = np.array(
        [[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert (
        float(
            world_copy.get_kinematic_structure_entity_by_name("bf").global_pose.to_np()[
                0, 3
            ]
        )
        == 0.0
    )
    assert float(bf.global_pose.to_np()[0, 3]) == 1.5


def test_copy_world_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    world.state[connection.dof.id].position = 1.0
    world.notify_state_change()
    world_copy = deepcopy(world)

    assert world.get_connection(r1, r2).position == 1.0
    assert world_copy.get_connection(r1, r2).position == 1.0


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
        assert dof.lower_limits == new_dof.lower_limits
        assert dof.upper_limits == new_dof.upper_limits


def test_copy_pr2_world_state_reset(pr2_world_state_reset):
    pr2_world_state_reset.state[
        pr2_world_state_reset.get_degree_of_freedom_by_name("torso_lift_joint").id
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
            pr2_body.global_pose.to_np(), pr2_copy_body.global_pose.to_np(), decimal=4
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
        pr2_world_state_reset.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position = 0.3
    pr2_world_state_reset.notify_state_change()
    pr2_copy = deepcopy(pr2_world_state_reset)
    assert pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "head_tilt_link"
    ).global_pose.to_np()[2, 3] == pytest.approx(1.472, abs=1e-3)
    assert pr2_copy.get_kinematic_structure_entity_by_name(
        "head_tilt_link"
    ).global_pose.to_np()[2, 3] == pytest.approx(1.472, abs=1e-3)


def test_copy_connections(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    for connection in pr2_world_state_reset.connections:
        pr2_copy_connection = pr2_copy.get_connection_by_name(connection.name)
        assert connection.name == pr2_copy_connection.name
        np.testing.assert_array_almost_equal(
            connection.origin.to_np(), pr2_copy_connection.origin.to_np(), decimal=3
        )
    pr2_copy.state[
        pr2_copy.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position = 0.3
    pr2_copy.notify_state_change()

    assert_raises(
        AssertionError,
        np.testing.assert_array_almost_equal,
        pr2_world_state_reset.get_connection_by_name("torso_lift_joint").origin.to_np(),
        pr2_copy.get_connection_by_name("torso_lift_joint").origin.to_np(),
    )


def test_copy_two_times(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    pr2_copy_2 = deepcopy(pr2_copy)
    for connection in pr2_world_state_reset.connections:
        pr2_copy_connection = pr2_copy_2.get_connection_by_name(connection.name)
        assert connection.name == pr2_copy_connection.name


def test_copy_id(pr2_world_state_reset):
    pr2_copy = deepcopy(pr2_world_state_reset)
    for body in pr2_world_state_reset.bodies:
        assert body.id == pr2_copy.get_kinematic_structure_entity_by_name(body.name).id


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
        pr2_copy.get_body_by_name("base_footprint").global_pose.to_position().to_np(),
        np.array([10.0, 10.0, 0.0, 1.0]),
    )


def test_omni_merge(pr2_world_state_reset, apartment_world_setup):
    pr2_copy = deepcopy(pr2_world_state_reset)
    apartment_copy = deepcopy(apartment_world_setup)

    pr2_copy.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(10, 10, 0)
    )

    np.testing.assert_array_almost_equal(
        pr2_copy.get_body_by_name("base_footprint").global_pose.to_position().to_np(),
        np.array([10.0, 10.0, 0.0, 1.0]),
    )

    apartment_copy.merge_world(
        pr2_copy
    )  # , TransformationMatrix.from_xyz_rpy(1.4, 2, 0))
    np.testing.assert_array_almost_equal(
        apartment_copy.get_body_by_name("base_footprint")
        .global_pose.to_position()
        .to_np(),
        np.array([1.4, 2, 0, 1]),
    )

    apartment_copy.get_body_by_name("base_footprint").parent_connection.origin = (
        TransformationMatrix.from_xyz_rpy(10, 10, 0)
    )
    np.testing.assert_array_almost_equal(
        apartment_copy.get_body_by_name("base_footprint")
        .global_pose.to_position()
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
    assert connection.dof.lower_limits.velocity == -1
    assert connection.dof.upper_limits.velocity == 1

    new_limits = DerivativeMap([0.69, 0.42, 1337, 23])

    connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits * -1, new_upper_limits=new_limits
    )
    assert connection.dof.lower_limits.position == -new_limits.position
    assert connection.dof.upper_limits.position == new_limits.position
    assert connection.dof.lower_limits.velocity == -new_limits.velocity
    assert connection.dof.upper_limits.velocity == new_limits.velocity
    assert connection.dof.lower_limits.acceleration == -new_limits.acceleration
    assert connection.dof.upper_limits.acceleration == new_limits.acceleration
    assert connection.dof.lower_limits.jerk == -new_limits.jerk
    assert connection.dof.upper_limits.jerk == new_limits.jerk

    new_limits2 = DerivativeMap([3333, 3333, 3333, 3333])

    connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits2 * -1, new_upper_limits=new_limits2
    )
    assert connection.dof.lower_limits.position == -new_limits.position
    assert connection.dof.upper_limits.position == new_limits.position
    assert connection.dof.lower_limits.velocity == -new_limits.velocity
    assert connection.dof.upper_limits.velocity == new_limits.velocity
    assert connection.dof.lower_limits.acceleration == -new_limits.acceleration
    assert connection.dof.upper_limits.acceleration == new_limits.acceleration
    assert connection.dof.lower_limits.jerk == -new_limits.jerk
    assert connection.dof.upper_limits.jerk == new_limits.jerk


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
            dof_id=connection.dof_id,
        )
        world.add_body(body)
        world.add_connection(mimic_connection)

    # when the multiplier is negative, the vel limits shouldn't be swapped
    assert np.isclose(
        mimic_connection.dof.lower_limits.velocity,
        connection.dof.lower_limits.velocity * 2,
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.velocity,
        connection.dof.upper_limits.velocity * 2,
    )

    new_limits = DerivativeMap([0.69, 0.42, 1337, 23])

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
    assert np.isclose(mimic_connection.dof.lower_limits.position, 21.62)
    assert np.isclose(mimic_connection.dof.upper_limits.position, 24.38)

    # Velocity limits: only multiplier applied (no offset), but absolute value for limits
    # Since we're dealing with limits, velocity should use abs(multiplier) = 2
    assert np.isclose(
        mimic_connection.dof.lower_limits.velocity, (new_limits * -1).velocity * 2
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.velocity, new_limits.velocity * 2
    )

    # Acceleration limits: only multiplier applied (no offset), absolute value for limits
    assert np.isclose(
        mimic_connection.dof.lower_limits.acceleration,
        (new_limits * -1).acceleration * 2,
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.acceleration, new_limits.acceleration * 2
    )

    # Jerk limits: only multiplier applied (no offset), absolute value for limits
    assert np.isclose(
        mimic_connection.dof.lower_limits.jerk, (new_limits * -1).jerk * 2
    )
    assert np.isclose(mimic_connection.dof.upper_limits.jerk, new_limits.jerk * 2)

    # limits are only applied if the new ones are lower
    new_limits2 = DerivativeMap([3333, 3333, 3333, 3333])

    mimic_connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits2 * -1, new_upper_limits=new_limits2
    )

    assert np.isclose(mimic_connection.dof.lower_limits.position, 21.62)
    assert np.isclose(mimic_connection.dof.upper_limits.position, 24.38)

    assert np.isclose(
        mimic_connection.dof.lower_limits.velocity, (new_limits * -1).velocity * 2
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.velocity, new_limits.velocity * 2
    )

    assert np.isclose(
        mimic_connection.dof.lower_limits.acceleration,
        (new_limits * -1).acceleration * 2,
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.acceleration, new_limits.acceleration * 2
    )

    assert np.isclose(
        mimic_connection.dof.lower_limits.jerk, (new_limits * -1).jerk * 2
    )
    assert np.isclose(mimic_connection.dof.upper_limits.jerk, new_limits.jerk * 2)


def test_missing_world_modification_context(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(MissingWorldModificationContextError):
        world.add_semantic_annotation(Handle(body=l1))


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
        ...


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
        world.remove_connection(body2.parent_connection)

        c_root_bf = OmniDrive.create_with_dofs(parent=body1, child=body2, world=world)
        world.add_connection(c_root_bf)


def test_set_static_collision_config():
    w = World()

    with w.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        w.add_kinematic_structure_entity(b1)
        w.add_kinematic_structure_entity(b2)

        dof = DegreeOfFreedom(name=PrefixedName("dofyboi"))
        w.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            b1, b2, axis=Vector3.from_iterable([0, 0, 1]), dof_id=dof.id
        )
        w.add_connection(connection)

        collision_config = CollisionCheckingConfig(
            buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=4
        )
        connection.set_static_collision_config_for_direct_child_bodies(collision_config)


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
