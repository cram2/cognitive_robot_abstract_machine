from collections import defaultdict

import numpy as np
import pytest
from typing_extensions import List

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import DuplicateRobotAssignmentsError
from semantic_digital_twin.orm.ormatic_interface import *  # noqa
from semantic_digital_twin.reasoning.predicates import LeftOf
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import (
    PR2,
    PR2MobileBase,
    PR2Torso,
    PR2Neck,
    PR2LeftArm,
    PR2RightArm,
    PR2LeftGripper,
    PR2RightGripper,
    PR2LeftGripperLeftFinger,
    PR2LeftGripperRightFinger,
    PR2RightGripperLeftFinger,
    PR2RightGripperRightFinger,
    PR2KinectV1,
)
from semantic_digital_twin.robots.robot_parts import KinematicChain
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_computations.ik_solver import (
    MaxIterationsException,
    UnreachableException,
)
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    DifferentialDrive,
    PrismaticConnection,
    RevoluteConnection,
    ActiveConnection1DOF,
)


def test_compute_chain_of_bodies_pr2(pr2_world_state_reset):
    root_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )
    tip_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    real = pr2_world_state_reset.compute_chain_of_kinematic_structure_entities(
        root=root_link, tip=tip_link
    )
    real = [x.name for x in real]
    assert real == [
        PrefixedName(name="base_footprint", prefix="pr2"),
        PrefixedName(name="base_link", prefix="pr2"),
        PrefixedName(name="torso_lift_link", prefix="pr2"),
        PrefixedName(name="r_shoulder_pan_link", prefix="pr2"),
        PrefixedName(name="r_shoulder_lift_link", prefix="pr2"),
        PrefixedName(name="r_upper_arm_roll_link", prefix="pr2"),
        PrefixedName(name="r_upper_arm_link", prefix="pr2"),
        PrefixedName(name="r_elbow_flex_link", prefix="pr2"),
        PrefixedName(name="r_forearm_roll_link", prefix="pr2"),
        PrefixedName(name="r_forearm_link", prefix="pr2"),
        PrefixedName(name="r_wrist_flex_link", prefix="pr2"),
        PrefixedName(name="r_wrist_roll_link", prefix="pr2"),
        PrefixedName(name="r_gripper_palm_link", prefix="pr2"),
        PrefixedName(name="r_gripper_tool_frame", prefix="pr2"),
    ]


def test_compute_chain_of_connections_pr2(pr2_world_state_reset):
    root_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )
    tip_link = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    real = pr2_world_state_reset.compute_chain_of_connections(
        root=root_link, tip=tip_link
    )
    real = [x.name for x in real]
    assert real == [
        PrefixedName(name="base_footprint_joint", prefix="pr2"),
        PrefixedName(name="torso_lift_joint", prefix="pr2"),
        PrefixedName(name="r_shoulder_pan_joint", prefix="pr2"),
        PrefixedName(name="r_shoulder_lift_joint", prefix="pr2"),
        PrefixedName(name="r_upper_arm_roll_joint", prefix="pr2"),
        PrefixedName(name="r_upper_arm_joint", prefix="pr2"),
        PrefixedName(name="r_elbow_flex_joint", prefix="pr2"),
        PrefixedName(name="r_forearm_roll_joint", prefix="pr2"),
        PrefixedName(name="r_forearm_joint", prefix="pr2"),
        PrefixedName(name="r_wrist_flex_joint", prefix="pr2"),
        PrefixedName(name="r_wrist_roll_joint", prefix="pr2"),
        PrefixedName(name="r_gripper_palm_joint", prefix="pr2"),
        PrefixedName(name="r_gripper_tool_joint", prefix="pr2"),
    ]


def test_duplicate_robot_assignments_error_pr2(pr2_world_state_reset):
    [pr2] = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)
    assert pr2 is not None, "PR2 robot not found in world state reset"

    # Create another robot
    with pytest.raises(DuplicateRobotAssignmentsError):
        pr2_2 = PR2.from_world(pr2_world_state_reset)


def test_compute_chain_of_connections_error_pr2(pr2_world_state_reset):
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    with pytest.raises(AssertionError):
        pr2_world_state_reset.compute_chain_of_connections(root, tip)


def test_compute_split_chain_of_bodies_pr2(pr2_world_state_reset):
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_r_finger_tip_link"
    )
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_l_finger_tip_link"
    )
    chain1, connection, chain2 = (
        pr2_world_state_reset.compute_split_chain_of_kinematic_structure_entities(
            root, tip
        )
    )
    chain1 = [n.name.name for n in chain1]
    connection = [n.name.name for n in connection]
    chain2 = [n.name.name for n in chain2]
    assert chain1 == [
        "l_gripper_r_finger_tip_link",
        "l_gripper_r_finger_link",
    ]
    assert connection == ["l_gripper_palm_link"]
    assert chain2 == ["l_gripper_l_finger_link", "l_gripper_l_finger_tip_link"]


def test_get_split_chain_pr2(pr2_world_state_reset):
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_r_finger_tip_link"
    )
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_l_finger_tip_link"
    )
    chain1, chain2 = pr2_world_state_reset.compute_split_chain_of_connections(root, tip)
    chain1 = [n.name.name for n in chain1]
    chain2 = [n.name.name for n in chain2]
    assert chain1 == ["l_gripper_r_finger_tip_joint", "l_gripper_r_finger_joint"]
    assert chain2 == ["l_gripper_l_finger_joint", "l_gripper_l_finger_tip_joint"]


def test_compute_fk_np_pr2(pr2_world_state_reset):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_tool_frame"
    )
    pr2_world_state_reset.notify_state_change()
    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    # fk = pr2_world_state_reset.compose_forward_kinematics_expression(root, tip).evaluate()
    # print(pr2_world_state_reset.state.to_position_dict())
    np.testing.assert_array_almost_equal(
        fk,
        np.array(
            [
                [1.0, 0.0, 0.0, -0.0356],
                [0, 1.0, 0.0, -0.376],
                [0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_compute_fk_np_pr2_root_left_hand(pr2_world_state_reset):
    tip = pr2_world_state_reset.root
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_tool_frame"
    )

    connection = pr2_world_state_reset.get_connections_by_type(OmniDrive)[0]
    connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=1, yaw=1)

    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    np.testing.assert_array_almost_equal(
        fk,
        np.array(
            [
                [0.523, 0.815, 0.247, -1.692],
                [-0.841, 0.540, 0.0, 0.653],
                [-0.133, -0.208, 0.968, -0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
        decimal=3,
    )


def test_compute_fk_np_l_elbow_flex_joint_pr2(pr2_world_state_reset):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_elbow_flex_link"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_upper_arm_link"
    )

    fk_expr = pr2_world_state_reset.compose_forward_kinematics_expression(root, tip)
    fk2 = fk_expr.evaluate()

    np.testing.assert_array_almost_equal(
        fk2,
        np.array(
            [
                [0.988771, 0.0, -0.149438, 0.4],
                [0.0, 1.0, 0.0, 0.0],
                [0.149438, 0.0, 0.988771, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_compute_ik(pr2_world_state_reset):
    bf = pr2_world_state_reset.root
    eef = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    fk = pr2_world_state_reset.compute_forward_kinematics_np(bf, eef)
    fk[0, 3] -= 0.2
    joint_state = pr2_world_state_reset.compute_inverse_kinematics(
        bf, eef, HomogeneousTransformationMatrix(fk, reference_frame=bf)
    )
    for joint, state in joint_state.items():
        pr2_world_state_reset.state[joint.id].position = state
    pr2_world_state_reset.notify_state_change()
    actual_fk = pr2_world_state_reset.compute_forward_kinematics_np(bf, eef)
    assert np.allclose(actual_fk, fk, atol=1e-3)


def test_compute_ik_max_iter(pr2_world_state_reset):
    bf = pr2_world_state_reset.root
    eef = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    fk = pr2_world_state_reset.compute_forward_kinematics_np(bf, eef)
    fk[2, 3] = 10
    with pytest.raises(MaxIterationsException):
        pr2_world_state_reset.compute_inverse_kinematics(
            bf, eef, HomogeneousTransformationMatrix(fk, reference_frame=bf)
        )


def test_compute_ik_unreachable(pr2_world_state_reset):
    bf = pr2_world_state_reset.root
    eef = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    fk = pr2_world_state_reset.compute_forward_kinematics_np(bf, eef)
    fk[2, 3] = -1
    with pytest.raises(UnreachableException):
        pr2_world_state_reset.compute_inverse_kinematics(
            bf, eef, HomogeneousTransformationMatrix(fk, reference_frame=bf)
        )


def test_apply_control_commands_omni_drive_pr2(pr2_world_state_reset):
    omni_drive: OmniDrive = pr2_world_state_reset.get_connection_by_name(
        "odom_combined_T_base_footprint"
    )
    cmd = np.zeros((len(pr2_world_state_reset.degrees_of_freedom)), dtype=float)
    cmd[pr2_world_state_reset.state._index[omni_drive.x_velocity.id]] = 100
    cmd[pr2_world_state_reset.state._index[omni_drive.y_velocity.id]] = 100
    cmd[pr2_world_state_reset.state._index[omni_drive.yaw.id]] = 100
    dt = 0.1
    pr2_world_state_reset.apply_control_commands(cmd, dt, Derivatives.jerk)
    assert pr2_world_state_reset.state[omni_drive.yaw.id].jerk == 100.0
    assert pr2_world_state_reset.state[omni_drive.yaw.id].acceleration == 100.0 * dt
    assert pr2_world_state_reset.state[omni_drive.yaw.id].velocity == 100.0 * dt * dt
    assert (
        pr2_world_state_reset.state[omni_drive.yaw.id].position == 100.0 * dt * dt * dt
    )

    assert pr2_world_state_reset.state[omni_drive.x_velocity.id].jerk == 100.0
    assert (
        pr2_world_state_reset.state[omni_drive.x_velocity.id].acceleration == 100.0 * dt
    )
    assert (
        pr2_world_state_reset.state[omni_drive.x_velocity.id].velocity
        == 100.0 * dt * dt
    )
    assert pr2_world_state_reset.state[omni_drive.x_velocity.id].position == 0

    assert pr2_world_state_reset.state[omni_drive.y_velocity.id].jerk == 100.0
    assert (
        pr2_world_state_reset.state[omni_drive.y_velocity.id].acceleration == 100.0 * dt
    )
    assert (
        pr2_world_state_reset.state[omni_drive.y_velocity.id].velocity
        == 100.0 * dt * dt
    )
    assert pr2_world_state_reset.state[omni_drive.y_velocity.id].position == 0

    assert pr2_world_state_reset.state[omni_drive.x.id].jerk == 0.0
    assert pr2_world_state_reset.state[omni_drive.x.id].acceleration == 0.0
    assert pr2_world_state_reset.state[omni_drive.x.id].velocity == 0.0
    assert pr2_world_state_reset.state[omni_drive.x.id].position == 0.08951707486311977

    assert pr2_world_state_reset.state[omni_drive.y.id].jerk == 0.0
    assert pr2_world_state_reset.state[omni_drive.y.id].acceleration == 0.0
    assert pr2_world_state_reset.state[omni_drive.y.id].velocity == 0.0
    assert pr2_world_state_reset.state[omni_drive.y.id].position == 0.1094837581924854


def test_apply_control_commands_diff_drive(cylinder_bot_diff_world):
    diff_drive: DifferentialDrive = cylinder_bot_diff_world.get_connection_by_name(
        "map_T_bot"
    )
    cmd = np.zeros((len(cylinder_bot_diff_world.degrees_of_freedom)), dtype=float)
    cmd[cylinder_bot_diff_world.state._index[diff_drive.x_velocity.id]] = 100
    cmd[cylinder_bot_diff_world.state._index[diff_drive.yaw.id]] = 100
    dt = 0.1
    cylinder_bot_diff_world.apply_control_commands(cmd, dt, Derivatives.jerk)
    assert cylinder_bot_diff_world.state[diff_drive.yaw.id].jerk == 100.0
    assert cylinder_bot_diff_world.state[diff_drive.yaw.id].acceleration == 100.0 * dt
    assert cylinder_bot_diff_world.state[diff_drive.yaw.id].velocity == 100.0 * dt * dt
    assert (
        cylinder_bot_diff_world.state[diff_drive.yaw.id].position
        == 100.0 * dt * dt * dt
    )

    assert cylinder_bot_diff_world.state[diff_drive.x_velocity.id].jerk == 100.0
    assert (
        cylinder_bot_diff_world.state[diff_drive.x_velocity.id].acceleration
        == 100.0 * dt
    )
    assert (
        cylinder_bot_diff_world.state[diff_drive.x_velocity.id].velocity
        == 100.0 * dt * dt
    )
    assert cylinder_bot_diff_world.state[diff_drive.x_velocity.id].position == 0

    assert cylinder_bot_diff_world.state[diff_drive.x.id].jerk == 0.0
    assert cylinder_bot_diff_world.state[diff_drive.x.id].acceleration == 0.0
    assert cylinder_bot_diff_world.state[diff_drive.x.id].velocity == 0.0
    assert np.allclose(
        cylinder_bot_diff_world.state[diff_drive.x.id].position,
        100 * dt**3 * np.cos(100 * dt**3),
        atol=1e-3,
    )

    assert cylinder_bot_diff_world.state[diff_drive.y.id].jerk == 0.0
    assert cylinder_bot_diff_world.state[diff_drive.y.id].acceleration == 0.0
    assert cylinder_bot_diff_world.state[diff_drive.y.id].velocity == 0.0
    assert np.allclose(
        cylinder_bot_diff_world.state[diff_drive.y.id].position,
        100 * dt**3 * np.sin(100 * dt**3),
        atol=1e-3,
    )


def test_search_for_connections_of_type(pr2_world_state_reset: World):

    connections = pr2_world_state_reset.get_connections_by_type(OmniDrive)
    assert len(connections) == 1
    assert connections[0].name == PrefixedName(
        name="odom_combined_T_base_footprint", prefix="pr2"
    )
    assert (
        connections[0].parent
        == pr2_world_state_reset.root.child_kinematic_structure_entities[0]
    )
    assert connections[
        0
    ].child == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )

    connections = pr2_world_state_reset.get_connections_by_type(PrismaticConnection)
    assert len(connections) == 5
    assert connections[0].name == PrefixedName(name="torso_lift_joint", prefix="pr2")
    assert connections[
        0
    ].parent == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_link"
    )
    assert connections[
        0
    ].child == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "torso_lift_link"
    )
    assert connections[1].name == PrefixedName(
        name="r_gripper_motor_slider_joint", prefix="pr2"
    )
    assert connections[
        1
    ].parent == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_palm_link"
    )
    assert connections[
        1
    ].child == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_motor_slider_link"
    )
    assert connections[2].name == PrefixedName(name="r_gripper_joint", prefix="pr2")
    assert connections[
        2
    ].parent == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_r_finger_tip_link"
    )
    assert connections[
        2
    ].child == pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_l_finger_tip_frame"
    )

    connections = pr2_world_state_reset.get_connections_by_type(RevoluteConnection)
    assert len(connections) == 40


def test_pr2_semantic_annotation(pr2_world_state_reset):
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
    pr2.validate()
    # Ensure there are no loose bodies
    pr2_world_state_reset._notify_model_change()

    assert len(pr2.get_end_effectors()) == 2
    assert len(pr2.get_arms()) == 2
    assert len(pr2.get_sensors()) == 1
    assert pr2.torso.name.name == "PR2Torso"
    assert len(pr2.torso.neck.sensors) == 1
    assert pr2.left_arm and pr2.right_arm
    assert pr2.left_arm != pr2.right_arm
    assert pr2.get_sensors()[0] == pr2.get_default_camera()


def test_has_left_right_arm_mixin(pr2_world_state_reset):
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
    left_arm_chain = list(pr2.left_arm.bodies)
    right_arm_chain = list(pr2.right_arm.bodies)
    assert LeftOf(
        left_arm_chain[1].center_of_mass,
        right_arm_chain[1].center_of_mass,
        pr2.root.global_transform,
    )()


def test_kinematic_chains(pr2_world_state_reset):
    semantic_kinematic_chain_annotation: List[KinematicChain] = (
        pr2_world_state_reset.get_semantic_annotations_by_type(KinematicChain)
    )
    for chain in semantic_kinematic_chain_annotation:
        assert chain.root
        assert chain.tip


def test_tracy_semantic_annotation(tracy_world):
    tracy = tracy_world.get_semantic_annotations_by_type(Tracy)[0]

    tracy_world._notify_model_change()

    assert len(tracy.get_end_effectors()) == 2
    assert len(tracy.get_sensors()) == 1


def test_hsrb_semantic_annotation(_hsr_world_setup):
    hsrb = _hsr_world_setup.get_semantic_annotations_by_type(HSRB)[0]
    _hsr_world_setup._notify_model_change()

    assert len(hsrb.get_end_effectors()) == 1
    assert len(hsrb.get_arms()) == 1

    assert len(hsrb.get_sensors()) == 5
    assert hsrb.mobile_base.torso is not None


def test_pr2_tighten_dof_velocity_limits_of_1dof_connections(pr2_world_state_reset):
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    # set all joints to vel limit 1
    pr2.tighten_dof_velocity_limits_of_1dof_connections(defaultdict(lambda: 1))

    # try spacial case for specific joint
    new_limits = defaultdict(
        lambda: 0.5, {pr2._world.get_connection_by_name("head_pan_joint"): 23}
    )
    pr2.tighten_dof_velocity_limits_of_1dof_connections(new_limits)
    # if spacial case triggers, but the new limit is above the old one, nothing happens
    assert (
        pr2._world.get_connection_by_name("head_pan_joint").dof.limits.upper.velocity
        == 1
    )
    # new limit is applied to joint without spacial case
    assert (
        pr2._world.get_connection_by_name("head_tilt_joint").dof.limits.upper.velocity
        == 0.5
    )
    # non-spacial case where the old limit is below 1
    assert (
        pr2._world.get_connection_by_name("torso_lift_joint").dof.limits.upper.velocity
        == 0.013
    )


def test_pr2_tighten_dof_velocity_limits_proportionally(pr2_world_state_reset):
    """All limits scale by the same factor; the joint with the highest limit reaches maximum_velocity."""
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    connections_with_limits = [
        (connection, connection.raw_dof.limits.upper.velocity)
        for connection in pr2._world.get_connections_by_type(ActiveConnection1DOF)
        if connection.raw_dof.limits.upper.velocity is not None
    ]
    initial_maximum = max(velocity for _, velocity in connections_with_limits)
    maximum_velocity = initial_maximum / 10

    pr2.tighten_dof_velocity_limits_proportionally(maximum_velocity=maximum_velocity)

    for connection, initial_limit in connections_with_limits:
        expected = initial_limit * (maximum_velocity / initial_maximum)
        assert connection.raw_dof.limits.upper.velocity == pytest.approx(expected)
        # Symmetry: lower bound is the negative of the upper bound
        assert connection.raw_dof.limits.lower.velocity == pytest.approx(-expected)


def test_pr2_tighten_dof_velocity_limits_proportionally_preserves_ratios(
    pr2_world_state_reset,
):
    """The ratio between any two joints with different limits is identical before and after scaling."""
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    initial_limits = {
        connection: connection.raw_dof.limits.upper.velocity
        for connection in pr2._world.get_connections_by_type(ActiveConnection1DOF)
        if connection.raw_dof.limits.upper.velocity is not None
    }
    initial_maximum = max(initial_limits.values())

    distinct_pair = next(
        (
            (connection_a, connection_b)
            for connection_a in initial_limits
            for connection_b in initial_limits
            if abs(initial_limits[connection_a] - initial_limits[connection_b]) > 1e-10
        ),
        None,
    )

    connection_a, connection_b = distinct_pair
    before_ratio = initial_limits[connection_a] / initial_limits[connection_b]

    pr2.tighten_dof_velocity_limits_proportionally(maximum_velocity=initial_maximum / 2)

    after_ratio = (
        connection_a.raw_dof.limits.upper.velocity
        / connection_b.raw_dof.limits.upper.velocity
    )
    assert before_ratio == pytest.approx(after_ratio)


def test_pr2_tighten_dof_velocity_limits_proportionally_no_op_when_within_bounds(
    pr2_world_state_reset,
):
    """When maximum_velocity is at or above the current maximum, no limits are changed."""
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    initial_limits = {
        connection: connection.raw_dof.limits.upper.velocity
        for connection in pr2._world.get_connections_by_type(ActiveConnection1DOF)
        if connection.raw_dof.limits.upper.velocity is not None
    }
    initial_maximum = max(initial_limits.values())

    # Calling with the exact current maximum must be a no-op
    pr2.tighten_dof_velocity_limits_proportionally(maximum_velocity=initial_maximum)
    for connection, initial_limit in initial_limits.items():
        assert connection.raw_dof.limits.upper.velocity == pytest.approx(initial_limit)

    # Calling with a larger value must also be a no-op
    pr2.tighten_dof_velocity_limits_proportionally(maximum_velocity=100.0)
    for connection, initial_limit in initial_limits.items():
        assert connection.raw_dof.limits.upper.velocity == pytest.approx(initial_limit)


def test_split_chain_of_connections(pr2_world_state_reset):
    body1 = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_r_finger_link"
    )
    body2 = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "l_gripper_l_finger_link"
    )
    result = pr2_world_state_reset.compute_split_chain_of_connections(
        root=body1, tip=body2
    )
    result1_names = [c.name for c in result[0]]
    result2_names = [c.name for c in result[1]]
    chain1 = [
        PrefixedName(name="r_gripper_r_finger_joint", prefix="pr2"),
        PrefixedName(name="r_gripper_palm_joint", prefix="pr2"),
        PrefixedName(name="r_wrist_roll_joint", prefix="pr2"),
        PrefixedName(name="r_wrist_flex_joint", prefix="pr2"),
        PrefixedName(name="r_forearm_joint", prefix="pr2"),
        PrefixedName(name="r_forearm_roll_joint", prefix="pr2"),
        PrefixedName(name="r_elbow_flex_joint", prefix="pr2"),
        PrefixedName(name="r_upper_arm_joint", prefix="pr2"),
        PrefixedName(name="r_upper_arm_roll_joint", prefix="pr2"),
        PrefixedName(name="r_shoulder_lift_joint", prefix="pr2"),
        PrefixedName(name="r_shoulder_pan_joint", prefix="pr2"),
    ]

    chain2 = [
        PrefixedName(name="l_shoulder_pan_joint", prefix="pr2"),
        PrefixedName(name="l_shoulder_lift_joint", prefix="pr2"),
        PrefixedName(name="l_upper_arm_roll_joint", prefix="pr2"),
        PrefixedName(name="l_upper_arm_joint", prefix="pr2"),
        PrefixedName(name="l_elbow_flex_joint", prefix="pr2"),
        PrefixedName(name="l_forearm_roll_joint", prefix="pr2"),
        PrefixedName(name="l_forearm_joint", prefix="pr2"),
        PrefixedName(name="l_wrist_flex_joint", prefix="pr2"),
        PrefixedName(name="l_wrist_roll_joint", prefix="pr2"),
        PrefixedName(name="l_force_torque_adapter_joint", prefix="pr2"),
        PrefixedName(name="l_force_torque_joint", prefix="pr2"),
        PrefixedName(name="l_gripper_palm_joint", prefix="pr2"),
        PrefixedName(name="l_gripper_l_finger_joint", prefix="pr2"),
    ]
    assert result1_names == chain1
    assert result2_names == chain2


def test_robots_and_validate(supported_abstract_robots):
    for abstract_robot in supported_abstract_robots:
        print(f"Testing robot: {abstract_robot.__name__}")
        world = URDFParser.from_file(abstract_robot.get_ros_file_path()).parse()
        robot = abstract_robot.from_world(world)
        robot.validate()


def test_pr2_automatic_setup_correctly(pr2_world_state_reset):
    """
    Test that the PR2 instance correctly references all its parts in a consistent hierarchy,
    including mobile base, torso, arms, grippers, fingers, and sensors.
    Verifies that all parts reachable via robot._robot_parts are also reachable via the
    intended semantic attributes and have correct back-references.
    """
    # Retrieve the PR2 robot from the world
    robot = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    verified_parts = set()

    def verify_part(part, expected_type, parent_robot):
        assert isinstance(
            part, expected_type
        ), f"Part {part} is not of expected type {expected_type}"
        assert (
            part._robot == parent_robot
        ), f"Part {part} back-reference to robot is incorrect"
        verified_parts.add(part)
        return part

    # 1. PR2 -> MobileBase (via HasMobileBase mixin)
    mobile_base = verify_part(robot.mobile_base, PR2MobileBase, robot)

    # 2. MobileBase -> Torso (via HasTorso mixin)
    torso = verify_part(mobile_base.torso, PR2Torso, robot)
    assert robot.torso == torso, "PR2.torso property shortcut mismatch"

    # 3. Torso -> Neck (via HasNeck mixin)
    neck = verify_part(torso.neck, PR2Neck, robot)

    # 4. Neck -> Sensors (via HasSensors mixin)
    assert len(neck.sensors) == 1, "Neck should have exactly one sensor"
    verify_part(neck.sensors[0], PR2KinectV1, robot)

    # 5. Torso -> Arms (via HasLeftRightArm mixin)
    left_arm = verify_part(torso.left_arm, PR2LeftArm, robot)
    right_arm = verify_part(torso.right_arm, PR2RightArm, robot)
    assert robot.left_arm == left_arm, "PR2.left_arm property shortcut mismatch"
    assert robot.right_arm == right_arm, "PR2.right_arm property shortcut mismatch"

    # 6. Arms -> EndEffectors (via HasEndEffector mixin)
    left_gripper = verify_part(left_arm.end_effector, PR2LeftGripper, robot)
    right_gripper = verify_part(right_arm.end_effector, PR2RightGripper, robot)

    # 7. Grippers -> Fingers (via HasTwoFingers -> HasFingers mixin)
    assert (
        len(left_gripper.fingers) == 2
    ), "Left gripper should have exactly two fingers"
    l_l_finger = next(
        f for f in left_gripper.fingers if isinstance(f, PR2LeftGripperLeftFinger)
    )
    l_r_finger = next(
        f for f in left_gripper.fingers if isinstance(f, PR2LeftGripperRightFinger)
    )
    verify_part(l_l_finger, PR2LeftGripperLeftFinger, robot)
    verify_part(l_r_finger, PR2LeftGripperRightFinger, robot)

    assert (
        len(right_gripper.fingers) == 2
    ), "Right gripper should have exactly two fingers"
    r_l_finger = next(
        f for f in right_gripper.fingers if isinstance(f, PR2RightGripperLeftFinger)
    )
    r_r_finger = next(
        f for f in right_gripper.fingers if isinstance(f, PR2RightGripperRightFinger)
    )
    verify_part(r_l_finger, PR2RightGripperLeftFinger, robot)
    verify_part(r_r_finger, PR2RightGripperRightFinger, robot)

    # 8. Final Coverage Verification
    # Ensure that all robot parts discovered via automated introspection (robot._robot_parts)
    # have been explicitly checked in this test.
    all_discovered_parts = set(robot._robot_parts)

    missing_from_hierarchy = all_discovered_parts - verified_parts
    assert (
        not missing_from_hierarchy
    ), f"Some robot parts were discovered but not verified: {missing_from_hierarchy}"

    extra_in_hierarchy = verified_parts - all_discovered_parts
    assert (
        not extra_in_hierarchy
    ), f"Some verified parts were not in robot._robot_parts: {extra_in_hierarchy}"


def test_pr2_degrees_of_freedom_with_hardware_interface(pr2_world_state_reset):
    """
    Tests that the degrees_of_freedom_with_hardware_interface property
    correctly identifies all controlled joints of the PR2 robot.
    """
    robot = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
    dofs = robot.degrees_of_freedom_with_hardware_interface

    # PR2 has:
    # 7 DOFs per arm (14 total)
    # 2 DOFs for the neck (pan/tilt)
    # 1 DOF for the torso (lift)
    # Total = 17
    expected_dof_names = {
        "l_shoulder_pan_joint",
        "l_shoulder_lift_joint",
        "l_upper_arm_roll_joint",
        "l_elbow_flex_joint",
        "l_forearm_roll_joint",
        "l_wrist_flex_joint",
        "l_wrist_roll_joint",
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
        "torso_lift_joint",
        "head_pan_joint",
        "head_tilt_joint",
    }

    actual_dof_names = {dof.name.name for dof in dofs}

    assert (
        len(dofs) == 17
    ), f"Expected 17 DOFs with hardware interface, but got {len(dofs)}: {actual_dof_names}"

    assert (
        actual_dof_names == expected_dof_names
    ), f"Missing DOFs: {expected_dof_names - actual_dof_names}, Extra DOFs: {actual_dof_names - expected_dof_names}"
