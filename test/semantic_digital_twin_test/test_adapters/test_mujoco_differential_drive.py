"""
Tests for simulating a differential drive physically in MuJoCo: a base that only drives
along where it faces and turns about its vertical axis.
"""

from __future__ import annotations

from datetime import timedelta

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import (
    MujocoBuilder,
    MujocoConnectionJoints,
    MujocoDifferentialDriveJoints,
    MujocoSim,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import DifferentialDrive
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, VelocityServo

# %% fixtures


def _servoed_rover(start_yaw: float = 0.0) -> DifferentialDrive:
    """
    :param start_yaw: Where the base faces at the start, about the ``odom``'s z-axis.
    :return: The drive of a box-shaped base on a differential drive whose forward
        velocity and yaw are driven by velocity servos, with gains for the box's mass
        and inertia.
    """
    world = World()
    odom = Body(name=PrefixedName("odom", "rover"))
    base = Body(
        name=PrefixedName("base", "rover"),
        collision=ShapeCollection([Box(scale=Scale(0.5, 0.5, 0.3))]),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(odom)
        world.add_kinematic_structure_entity(base)
        drive = DifferentialDrive.create_with_dofs(
            world, odom, base, name=PrefixedName("drive", "rover")
        )
        world.add_connection(drive)
        for degree_of_freedom in drive.active_dofs:
            servo = VelocityServo(
                name=PrefixedName(f"{degree_of_freedom.name.name}_servo", "rover"),
                gains=(
                    VelocityServoGains(damping=50.0, torque_limit=50.0)
                    if degree_of_freedom == drive.yaw
                    else VelocityServoGains(damping=2000.0, torque_limit=1000.0)
                ),
            )
            servo.add_dof(degree_of_freedom)
            world.add_actuator(servo)
    world.state[drive.yaw.id].position = start_yaw
    world.notify_state_change()
    return drive


@pytest.fixture
def rover() -> DifferentialDrive:
    return _servoed_rover()


def _drive_joints(builder: MujocoBuilder, drive: DifferentialDrive) -> list:
    """
    :return: The joints the compiled model moves the base with.
    """
    base = next(
        body for body in builder.spec.bodies if body.name == drive.child.name.name
    )
    return list(base.joints)


def _drive_at(
    drive: DifferentialDrive,
    forward_velocity: float,
    angular_velocity: float,
    duration: timedelta,
) -> tuple[float, float, float]:
    """
    Command the drive for a while in a fresh simulation, every tick as a controller
    does, then stop it.

    :return: The x, y and yaw the simulated base stands at afterwards.
    """
    world = drive._world
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
        x, y, _ = simulator.get_body_position(body_name=drive.child.name.name).result
        w, i, j, k = simulator.get_body_quaternion(
            body_name=drive.child.name.name
        ).result
    finally:
        simulation.stop_simulation()
    yaw = Rotation.from_quat([i, j, k, w]).as_euler("xyz")[2]
    return float(x), float(y), float(yaw)


# %% the compiled model


def test_a_differential_drive_is_built_by_its_own_realisation(rover):
    assert type(MujocoConnectionJoints.of(rover, rover._world)) is (
        MujocoDifferentialDriveJoints
    )


def test_the_compiled_model_moves_the_base_along_x_and_y_and_about_z(rover, tmp_path):
    builder = MujocoBuilder()
    builder.build_world(world=rover._world, file_path=str(tmp_path / "scene.xml"))

    assert [
        (joint.name, joint.type, list(joint.axis))
        for joint in _drive_joints(builder, rover)
    ] == [
        ("rover_differential_drive_x", mujoco.mjtJoint.mjJNT_SLIDE, [1.0, 0.0, 0.0]),
        ("rover_differential_drive_y", mujoco.mjtJoint.mjJNT_SLIDE, [0.0, 1.0, 0.0]),
        ("rover_differential_drive_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0.0, 0.0, 1.0]),
    ]


def test_every_joint_the_base_moves_with_is_driven(rover, tmp_path):
    """
    The slide along y has no degree of freedom of the drive commanding it, and is driven
    all the same, so the base cannot slip sideways.
    """
    builder = MujocoBuilder()
    builder.build_world(world=rover._world, file_path=str(tmp_path / "scene.xml"))

    actuated_joints = {actuator.target for actuator in builder.spec.actuators}
    assert {joint.name for joint in _drive_joints(builder, rover)} <= actuated_joints


def test_a_drive_without_servos_keeps_the_base_where_it_is_built(rover, tmp_path):
    world = rover._world
    with world.modify_world():
        for actuator in list(world.actuators):
            world.remove_actuator(actuator)
    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    assert _drive_joints(builder, rover) == []


# %% the physics


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_base_drives_where_it_faces():
    """
    A base facing along the ``odom``'s y-axis drives along it when commanded forward.
    """
    rover = _servoed_rover(start_yaw=np.pi / 2)

    reached = _drive_at(rover, 0.2, 0.0, timedelta(seconds=2))

    assert reached == pytest.approx((0.0, 0.4, np.pi / 2), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_base_driving_while_it_turns_follows_an_arc(rover):
    """
    Driving forward while turning, the base follows the circle the forward and angular
    velocity describe, without slipping sideways off it.
    """
    forward_velocity, angular_velocity, seconds = 0.2, 0.5, 2.0
    turned = angular_velocity * seconds
    radius = forward_velocity / angular_velocity

    reached = _drive_at(
        rover, forward_velocity, angular_velocity, timedelta(seconds=seconds)
    )

    assert reached == pytest.approx(
        (radius * np.sin(turned), radius * (1 - np.cos(turned)), turned), abs=0.02
    )


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_odometry_is_where_the_simulated_base_got_to(rover):
    reached = _drive_at(rover, 0.2, 0.5, timedelta(seconds=2))
    odometry = tuple(
        rover._world.state[degree_of_freedom.id].position
        for degree_of_freedom in (rover.x, rover.y, rover.yaw)
    )

    assert odometry == pytest.approx(reached, abs=1e-3)
