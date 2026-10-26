"""
Tests for the Tiago described by MuJoCo's own model of it, which is the description that
makes its hand able to hold something.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tiago import TiagoMujoco
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)

pytestmark = pytest.mark.skipif(
    not TiagoMujoco.is_description_installed(),
    reason="The MuJoCo model of the Tiago is not installed",
)


@pytest.fixture
def mounted_tiago() -> TiagoMujoco:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(TiagoMujoco).spawn(world)


# %% reading the description


def test_the_robot_hangs_straight_off_the_drive_that_moves_it():
    """
    MJCF describes every body inside a world body, and a frame left above the robot's
    own root would sit between the robot and its drive, leaving it undriven.
    """
    robot = TiagoMujoco.from_description()

    assert robot.root is robot._world.root


def test_a_mounted_robot_is_moved_across_the_floor_by_its_own_drive(mounted_tiago):
    """
    The drive the mobile base declares attaches the robot to its odom, so it is the
    robot's root's own parent.
    """
    assert isinstance(mounted_tiago.drive, DifferentialDrive)
    assert mounted_tiago.drive is mounted_tiago.root.parent_connection


# %% what drives it


def test_every_joint_the_robot_moves_has_a_servo_to_move_it(mounted_tiago):
    """
    A physically simulated robot is moved by servos rather than placed, so each joint of
    the parts that carry an object is driven by one.
    """
    servoed = {
        degree_of_freedom.id
        for actuator in mounted_tiago._world.actuators
        if isinstance(actuator, PositionServo)
        for degree_of_freedom in actuator.dofs
    }
    carrying = [
        mounted_tiago.mobile_base.torso,
        *mounted_tiago.all_arms,
        *[arm.end_effector for arm in mounted_tiago.all_arms],
    ]
    undriven = [
        connection.name.name
        for part in carrying
        for connection in part.active_connections
        if isinstance(connection, ActiveConnection1DOF)
        and connection.raw_dof.id not in servoed
    ]

    assert undriven == []


def test_a_servo_keeps_the_joint_dynamics_its_description_states(mounted_tiago):
    """
    An MJCF states each joint's own damping and friction, so declaring a servo on that
    joint leaves them alone rather than replacing them with the servo's own.
    """
    as_described = MJCFParser(TiagoMujoco.get_ros_file_path()).parse()

    for connection in mounted_tiago.connections:
        if not isinstance(connection, ActiveConnection1DOF):
            continue
        described = as_described.get_connection_by_name(connection.name.name)
        assert connection.dynamics == described.dynamics


def test_a_servoed_robot_does_not_hold_up_its_own_weight(mounted_tiago):
    """
    The servos are sized to move the robot and what it carries, not to carry the robot,
    so the simulation carries its weight.
    """
    uncompensated = [
        body.name.name
        for body in mounted_tiago.mobile_base.torso.bodies
        if body.get_simulator_property_of_type(GravityCompensation) is None
    ]

    assert uncompensated == []


# %% the hand


def test_the_fingers_meet_at_the_same_point_however_wide_the_hand_stands(mounted_tiago):
    """
    The fingers slide rather than swing, so the point they meet at does not travel along
    the approach as they close -- which is what lets a grasp planned with the hand open
    be reached with it closing.
    """
    for arm in mounted_tiago.all_arms:
        assert arm.end_effector.travel_when_closing == 0.0


def test_opening_the_hand_frees_the_whole_stroke_of_both_fingers(mounted_tiago):
    """
    Both fingers carry the hand's opening, so the room between the pads grows by what
    the two of them are asked to travel.
    """
    for arm in mounted_tiago.all_arms:
        gripper = arm.end_effector
        closed = gripper.get_joint_state_by_type(GripperState.CLOSE)
        opened = gripper.get_joint_state_by_type(GripperState.OPEN)
        stroke = sum(
            abs(open_value - closed_value)
            for open_value, closed_value in zip(
                opened.target_values, closed.target_values
            )
        )

        closed.apply_to(mounted_tiago._world)
        when_closed = gripper.fingertips.separation
        opened.apply_to(mounted_tiago._world)

        # a micrometre, which is the single-precision the collision mesh is read at
        assert gripper.fingertips.separation == pytest.approx(
            when_closed + stroke, abs=1e-6
        )


# %% the physics


def _drive_at(
    tiago: TiagoMujoco,
    forward_velocity: float,
    angular_velocity: float,
    duration: timedelta,
) -> tuple[float, float, float]:
    """
    Command the drive for a while in a fresh simulation, every tick as a controller
    does, then stop it.

    :return: The x, y and yaw the simulated base stands at afterwards.
    """
    world = tiago._world
    drive = tiago.drive
    tick = timedelta(milliseconds=10)
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        for _ in range(round(duration / tick)):
            world.state[drive.x_velocity.id].velocity = forward_velocity
            world.state[drive.yaw.id].velocity = angular_velocity
            world.notify_state_change()
            simulation.step_simulation(tick)
        world.state[drive.x_velocity.id].velocity = 0.0
        world.state[drive.yaw.id].velocity = 0.0
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=0.5))
        simulator = simulation.simulator
        x, y, _ = simulator.get_body_position(body_name=tiago.root.name.name).result
        w, i, j, k = simulator.get_body_quaternion(
            body_name=tiago.root.name.name
        ).result
    finally:
        simulation.stop_simulation()
    yaw = Rotation.from_quat([i, j, k, w]).as_euler("xyz")[2]
    return float(x), float(y), float(yaw)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_hold_the_parked_arms_and_the_raised_torso_up(mounted_tiago):
    world = mounted_tiago._world
    for arm in mounted_tiago.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    mounted_tiago.mobile_base.torso.get_joint_state_by_type(TorsoState.HIGH).apply_to(
        world
    )
    world.notify_state_change()
    tool_frames = [arm.end_effector.tool_frame for arm in mounted_tiago.all_arms]
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
def test_the_finger_servos_close_the_gripper(mounted_tiago):
    world = mounted_tiago._world
    gripper = mounted_tiago.mobile_base.torso.right_arm.end_effector
    gripper.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
    world.notify_state_change()
    finger_joint = gripper.thumb.root.parent_connection

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        gripper.get_joint_state_by_type(GripperState.CLOSE).apply_to(world)
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=1))
        reached = simulation.simulator.get_joint_value(finger_joint.name.name).result
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx(
        world.state[finger_joint.raw_dof.id].position, abs=0.002
    )


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_base_drives_at_the_commanded_velocity(mounted_tiago):
    reached = _drive_at(mounted_tiago, 0.2, 0.0, timedelta(seconds=2))

    assert reached == pytest.approx((0.4, 0.0, 0.0), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_base_turns_at_the_commanded_angular_velocity(mounted_tiago):
    reached = _drive_at(mounted_tiago, 0.0, 0.3, timedelta(seconds=2))

    assert reached == pytest.approx((0.0, 0.0, 0.6), abs=0.02)
