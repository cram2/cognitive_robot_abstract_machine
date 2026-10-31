"""
Tests for simulating the PR2 physically in MuJoCo: the servos driving its joints and the
drive moving its base across the floor.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from typing_extensions import List

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import (
    MujocoBuilder,
    MujocoConnectionJoints,
    MujocoConnectionJoint,
    MujocoFixedConnectionJoints,
    MujocoOmniDriveJoints,
    MujocoSim,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    JointServo,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
    VelocityServo,
)


def _spawn_pr2(odom_T_robot_start: HomogeneousTransformationMatrix | None) -> PR2:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(PR2, odom_T_robot_start=odom_T_robot_start).spawn(world)


@pytest.fixture
def pr2() -> PR2:
    return _spawn_pr2(odom_T_robot_start=None)


def _drive_joints(builder: MujocoBuilder, pr2: PR2) -> list[mujoco.MjsJoint]:
    """
    :return: The joints the compiled model moves the PR2's base with.
    """
    base = next(body for body in builder.spec.bodies if body.name == pr2.root.name.name)
    return list(base.joints)


def _simulated_base_pose(simulation: MujocoSim, pr2: PR2) -> tuple[float, float, float]:
    """
    :return: The x, y and yaw the simulated base stands at.
    """
    simulator = simulation.simulator
    x, y, _ = simulator.get_body_position(body_name=pr2.root.name.name).result
    w, i, j, k = simulator.get_body_quaternion(body_name=pr2.root.name.name).result
    yaw = Rotation.from_quat([i, j, k, w]).as_euler("xyz")[2]
    return float(x), float(y), float(yaw)


# %% the servos the PR2 declares


def test_every_commanded_joint_the_fingers_and_the_drive_are_driven_by_one_servo(pr2):
    """
    Each joint a controller commands and the finger joint of each gripper are driven by
    exactly one position servo, and each degree of freedom the drive is commanded with
    by exactly one velocity servo.
    """
    world = pr2._world
    drive = pr2.drive
    commanded = {
        connection.raw_dof
        for connection in world.get_connections_of_branch(pr2.root)
        if isinstance(connection, ActiveConnection1DOF)
        and connection.has_hardware_interface
    }
    fingers = {
        arm.end_effector.thumb.root.parent_connection.raw_dof for arm in pr2.all_arms
    }

    def driven_by(servo_type: type) -> List[str]:
        return sorted(
            str(degree_of_freedom.name)
            for actuator in world.actuators
            if type(actuator) is servo_type
            for degree_of_freedom in actuator.dofs
        )

    assert len(driven_by(PositionServo)) + len(driven_by(VelocityServo)) == len(
        world.actuators
    )
    assert driven_by(PositionServo) == sorted(
        str(dof.name) for dof in commanded | fingers
    )
    assert driven_by(VelocityServo) == sorted(
        str(dof.name) for dof in (drive.x_velocity, drive.y_velocity, drive.yaw)
    )


def test_the_drive_servos_have_the_gains_the_base_declares(pr2):
    world = pr2._world
    drive = pr2.drive
    drive_servos = pr2.mobile_base.drive_servos

    gains_by_degree_of_freedom = {
        degree_of_freedom: actuator.gains
        for actuator in world.actuators
        for degree_of_freedom in actuator.dofs
    }

    assert gains_by_degree_of_freedom[drive.x_velocity] == (
        drive_servos.translation_gains
    )
    assert gains_by_degree_of_freedom[drive.y_velocity] == (
        drive_servos.translation_gains
    )
    assert gains_by_degree_of_freedom[drive.yaw] == drive_servos.rotation_gains


def test_a_servo_that_states_no_dynamics_keeps_those_of_the_joint(pr2):
    """
    A description such as an MJCF states a joint's dynamics itself, and a servo that
    states none leaves them as described.
    """
    connection = pr2.torso.active_connections[0]
    described = JointDynamics(armature=0.25)
    connection.dynamics = described

    pr2.torso._declare_servo(
        connection,
        JointServo(gains=ServoGains(stiffness=1.0, damping=1.0, torque_limit=1.0)),
    )

    assert connection.dynamics is described


def test_every_body_standing_on_the_base_carries_its_own_weight(pr2):
    for body in pr2.mobile_base.bodies:
        assert body.get_simulator_property_of_type(GravityCompensation) == (
            GravityCompensation(fraction=1.0)
        )


# %% the compiled model


def test_the_compiled_model_moves_the_base_along_x_and_y_and_about_z(pr2, tmp_path):
    builder = MujocoBuilder()
    builder.build_world(world=pr2._world, file_path=str(tmp_path / "scene.xml"))

    drive_joints = _drive_joints(builder, pr2)

    assert [(joint.type, list(joint.axis)) for joint in drive_joints] == [
        (mujoco.mjtJoint.mjJNT_SLIDE, [1.0, 0.0, 0.0]),
        (mujoco.mjtJoint.mjJNT_SLIDE, [0.0, 1.0, 0.0]),
        (mujoco.mjtJoint.mjJNT_HINGE, [0.0, 0.0, 1.0]),
    ]
    actuated_joints = {actuator.target for actuator in builder.spec.actuators}
    assert {joint.name for joint in drive_joints} <= actuated_joints


def test_the_drive_joints_are_named_after_the_robot_and_the_axis_they_move(
    pr2, tmp_path
):
    builder = MujocoBuilder()
    builder.build_world(world=pr2._world, file_path=str(tmp_path / "scene.xml"))

    assert [joint.name for joint in _drive_joints(builder, pr2)] == [
        "pr2_omni_drive_x",
        "pr2_omni_drive_y",
        "pr2_omni_drive_yaw",
    ]


def test_each_connection_is_built_by_the_realisation_for_its_type(pr2):
    world = pr2._world
    revolute = next(
        connection
        for connection in world.connections
        if type(connection) is RevoluteConnection
    )
    fixed = next(
        connection
        for connection in world.connections
        if isinstance(connection, FixedConnection)
    )

    assert type(MujocoConnectionJoints.of(pr2.drive, world)) is MujocoOmniDriveJoints
    assert type(MujocoConnectionJoints.of(revolute, world)) is MujocoConnectionJoint
    assert type(MujocoConnectionJoints.of(fixed, world)) is (
        MujocoFixedConnectionJoints
    )


def test_a_drive_without_servos_keeps_the_base_where_it_is_built(pr2, tmp_path):
    """
    A robot whose base is moved kinematically keeps it bolted to its ``odom`` in the
    compiled model.
    """
    world = pr2._world
    drive = pr2.drive
    with world.modify_world():
        for actuator in list(world.actuators):
            if set(actuator.dofs) & set(drive.active_dofs):
                world.remove_actuator(actuator)
    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    assert _drive_joints(builder, pr2) == []


# %% the physics


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_hold_the_parked_arms_and_the_raised_torso_up(pr2):
    """
    A joint with no servo would be written straight into MuJoCo and sag under its own
    weight between one control cycle and the next.
    """
    world = pr2._world
    for arm in pr2.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    pr2.torso.get_joint_state_by_type(TorsoState.HIGH).apply_to(world)
    world.notify_state_change()
    tool_frames = [arm.end_effector.tool_frame for arm in pr2.all_arms]
    parked = [
        world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]
        for tool_frame in tool_frames
    ]

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        simulated = [
            simulation.simulator.get_body_position(
                body_name=tool_frame.name.name
            ).result
            for tool_frame in tool_frames
        ]
    finally:
        simulation.stop_simulation()

    for simulated_position, parked_position in zip(simulated, parked):
        assert list(simulated_position) == pytest.approx(
            list(parked_position), abs=0.01
        )


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_drive_the_joints_to_a_commanded_configuration(pr2):
    world = pr2._world
    commanded = {
        PR2Joint.TORSO_LIFT: 0.2,
        PR2Joint.HEAD_PAN: 0.5,
        PR2Joint.HEAD_TILT: 0.3,
        PR2Joint.RIGHT_SHOULDER_PAN: -0.5,
        PR2Joint.RIGHT_SHOULDER_LIFT: 0.3,
        PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.4,
        PR2Joint.RIGHT_ELBOW_FLEX: -1.0,
        PR2Joint.RIGHT_FOREARM_ROLL: 0.5,
        PR2Joint.RIGHT_WRIST_FLEX: -0.6,
        PR2Joint.RIGHT_WRIST_ROLL: 0.7,
    }

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        for joint_name, position in commanded.items():
            connection = world.get_connection_by_name(joint_name)
            world.state[connection.raw_dof.id].position = position
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=3))
        reached = {
            joint_name: simulation.simulator.get_joint_value(joint_name).result
            for joint_name in commanded
        }
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx(commanded, abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_finger_servo_closes_the_gripper(pr2):
    world = pr2._world
    gripper = pr2.right_arm.end_effector
    gripper.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    world.notify_state_change()
    finger_joint = gripper.thumb.root.parent_connection
    closed = gripper.get_joint_state_by_type(GripperState.CLOSE)

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        closed.apply_to(world)
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=1))
        reached = simulation.simulator.get_joint_value(finger_joint.name.name).result
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx(
        world.state[finger_joint.raw_dof.id].position, abs=0.02
    )


def _drive_at(
    simulation: MujocoSim,
    pr2: PR2,
    base_velocity: tuple[float, float, float],
    duration: timedelta,
) -> None:
    """
    Command the drive with a velocity in the frame of the base for a while, then stop it
    and let it come to a standstill.

    :param base_velocity: The forward, sideways and angular velocity.
    """
    world = pr2._world
    drive = pr2.drive
    for degree_of_freedom, velocity in zip(
        (drive.x_velocity, drive.y_velocity, drive.yaw), base_velocity
    ):
        world.state[degree_of_freedom.id].velocity = velocity
    world.notify_state_change()
    simulation.step_simulation(duration)
    for degree_of_freedom in (drive.x_velocity, drive.y_velocity, drive.yaw):
        world.state[degree_of_freedom.id].velocity = 0.0
    world.notify_state_change()
    simulation.step_simulation(timedelta(seconds=0.5))


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_drive_moves_the_base_at_the_commanded_velocity(pr2):
    simulation = MujocoSim(world=pr2._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, pr2, (0.2, -0.1, 0.0), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, pr2)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((0.4, -0.2, 0.0), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_drive_is_commanded_in_the_frame_of_the_base():
    """
    A base facing along the ``odom``'s y-axis drives along it when commanded forward.
    """
    pr2 = _spawn_pr2(
        odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2)
    )

    simulation = MujocoSim(world=pr2._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, pr2, (0.2, 0.0, 0.0), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, pr2)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((0.0, 0.4, np.pi / 2), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_drive_turns_the_base_at_the_commanded_angular_velocity(pr2):
    simulation = MujocoSim(world=pr2._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, pr2, (0.0, 0.0, 0.3), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, pr2)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((0.0, 0.0, 0.6), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_odometry_is_where_the_simulated_base_got_to(pr2):
    """
    The drive's x, y and yaw are measured odometry, so the world holds where the
    simulated base actually is rather than where it was commanded to be.
    """
    world = pr2._world
    drive = pr2.drive

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, pr2, (0.2, 0.1, 0.3), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, pr2)
        odometry = tuple(
            world.state[degree_of_freedom.id].position
            for degree_of_freedom in (drive.x, drive.y, drive.yaw)
        )
    finally:
        simulation.stop_simulation()

    assert odometry == pytest.approx(reached, abs=1e-3)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_base_too_weak_to_move_does_not_report_the_pose_it_was_commanded_to(pr2):
    """
    A base whose servos cannot push it stays put when commanded to drive, as one held up
    by an obstacle would, and the odometry says so instead of following the command.
    """
    world = pr2._world
    drive = pr2.drive
    translation_servos = [
        actuator
        for actuator in world.actuators
        if set(actuator.dofs) & {drive.x_velocity, drive.y_velocity}
    ]
    for servo in translation_servos:
        servo.gains = replace(servo.gains, torque_limit=1e-6)

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, pr2, (0.2, 0.0, 0.0), timedelta(seconds=1))
        odometry_x = world.state[drive.x.id].position
    finally:
        simulation.stop_simulation()

    assert odometry_x == pytest.approx(0.0, abs=0.01)


# %% localization: the odom is not where the world's root is


def _spawn_localized_pr2(
    world_T_odom: HomogeneousTransformationMatrix,
    odom_T_robot_start: HomogeneousTransformationMatrix | None = None,
) -> PR2:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(
        PR2, world_T_odom=world_T_odom, odom_T_robot_start=odom_T_robot_start
    ).spawn(world)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_localized_base_starts_where_the_world_has_it():
    world_T_odom = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=-0.5, yaw=np.pi / 2
    )
    odom_T_robot_start = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, y=0.2, yaw=0.3
    )
    pr2 = _spawn_localized_pr2(world_T_odom, odom_T_robot_start)

    simulation = MujocoSim(world=pr2._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        reached = _simulated_base_pose(simulation, pr2)
    finally:
        simulation.stop_simulation()

    # x = 1 - 0.2, y = -0.5 + 0.5, yaw = pi / 2 + 0.3
    assert reached == pytest.approx((0.8, 0.0, np.pi / 2 + 0.3), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_localized_base_drives_along_its_odom_and_reports_odometry_in_it():
    """
    The odom is turned by a quarter against the world's root, so driving forward moves
    the base along the root's y-axis, while its odometry counts along the odom's x-axis.
    """
    world_T_odom = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=-0.5, yaw=np.pi / 2
    )
    pr2 = _spawn_localized_pr2(world_T_odom)
    world = pr2._world
    drive = pr2.drive

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, pr2, (0.2, 0.0, 0.0), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, pr2)
        odometry = tuple(
            world.state[degree_of_freedom.id].position
            for degree_of_freedom in (drive.x, drive.y, drive.yaw)
        )
        world_T_base = world.compute_forward_kinematics_np(world.root, pr2.root)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((1.0, -0.1, np.pi / 2), abs=0.02)
    assert odometry == pytest.approx((0.4, 0.0, 0.0), abs=0.02)
    assert world_T_base[:2, 3].tolist() == pytest.approx([1.0, -0.1], abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_base_starts_where_the_world_has_it():
    start = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, yaw=0.5)
    pr2 = _spawn_pr2(odom_T_robot_start=start)

    simulation = MujocoSim(world=pr2._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        reached = _simulated_base_pose(simulation, pr2)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((1.0, 2.0, 0.5), abs=0.02)
