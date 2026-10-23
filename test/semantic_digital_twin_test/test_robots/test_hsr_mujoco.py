"""
Tests for simulating the HSRB physically in MuJoCo: the servos driving its joints, the
springs in its fingers and the drive moving its base across the floor.
"""

from __future__ import annotations

from datetime import timedelta

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from typing_extensions import List

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint, HSRBSpringJoint
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
    VelocityServo,
)


@pytest.fixture
def hsrb() -> HSRB:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(HSRB).spawn(world)


def _simulated_base_pose(
    simulation: MujocoSim, hsrb: HSRB
) -> tuple[float, float, float]:
    """
    :return: The x, y and yaw the simulated base stands at.
    """
    simulator = simulation.simulator
    x, y, _ = simulator.get_body_position(body_name=hsrb.root.name.name).result
    w, i, j, k = simulator.get_body_quaternion(body_name=hsrb.root.name.name).result
    yaw = Rotation.from_quat([i, j, k, w]).as_euler("xyz")[2]
    return float(x), float(y), float(yaw)


def _command(hsrb: HSRB, positions: dict[str, float]) -> None:
    """
    Command joints of the HSRB to positions, as a controller does.

    :param positions: The position of each joint, by the joint's name.
    """
    world = hsrb._world
    for joint_name, position in positions.items():
        connection = world.get_connection_by_name(joint_name)
        world.state[connection.raw_dof.id].position = position
    world.notify_state_change()


def _hold_the_arm_out_sideways(hsrb: HSRB) -> dict[str, float]:
    """
    Command the arm into the pose it reaches beside the base in: lifted, swung level and
    rolled out to the side.

    :return: The commanded position of each arm joint, by the joint's name.
    """
    commanded = {
        HSRBJoint.ARM_LIFT: 0.41,
        HSRBJoint.ARM_FLEX: -1.46,
        HSRBJoint.ARM_ROLL: 1.2,
        HSRBJoint.WRIST_FLEX: -0.28,
        HSRBJoint.WRIST_ROLL: -1.22,
    }
    _command(hsrb, commanded)
    return commanded


# %% the hand's states


def test_the_open_hand_spreads_its_fingers_as_far_as_the_hand_motor_turns(hsrb):
    hand_motor = hsrb._world.get_connection_by_name(HSRBJoint.HAND_MOTOR)

    opened = hsrb.end_effector.get_joint_state_by_type(GripperState.OPEN)

    assert opened.target_values == [hand_motor.raw_dof.limits.upper.position] * len(
        opened.connections
    )


# %% the servos the HSRB declares


def test_every_commanded_joint_the_hand_and_the_drive_are_driven_by_one_servo(hsrb):
    """
    Each joint a controller commands, the hand's motor and the spring of each finger are
    driven by exactly one position servo, and each degree of freedom the drive is
    commanded with by exactly one velocity servo.
    """
    world = hsrb._world
    drive = hsrb.drive
    commanded = {
        connection.raw_dof
        for connection in world.get_connections_of_branch(hsrb.root)
        if isinstance(connection, ActiveConnection1DOF)
        and connection.has_hardware_interface
    }
    hand = {
        world.get_connection_by_name(joint_name).raw_dof
        for joint_name in [HSRBJoint.HAND_MOTOR, *HSRBSpringJoint]
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
    assert driven_by(PositionServo) == sorted(str(dof.name) for dof in commanded | hand)
    assert driven_by(VelocityServo) == sorted(
        str(dof.name) for dof in (drive.x_velocity, drive.y_velocity, drive.yaw)
    )


def test_the_drive_servos_have_the_gains_the_base_declares(hsrb):
    world = hsrb._world
    drive = hsrb.drive
    drive_servos = hsrb.mobile_base.drive_servos

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


def test_every_body_standing_on_the_base_carries_its_own_weight(hsrb):
    for body in hsrb.mobile_base.bodies:
        assert body.get_simulator_property_of_type(GravityCompensation) == (
            GravityCompensation(fraction=1.0)
        )


# %% the physics: the joints


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_hold_the_parked_arm_and_the_raised_torso_up(hsrb):
    """
    A joint with no servo would be written straight into MuJoCo and sag under its own
    weight between one control cycle and the next.
    """
    world = hsrb._world
    hsrb.mobile_base.torso.arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(
        world
    )
    hsrb.mobile_base.torso.get_joint_state_by_type(TorsoState.HIGH).apply_to(world)
    world.notify_state_change()
    tool_frame = hsrb.end_effector.tool_frame
    parked = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        simulated = simulation.simulator.get_body_position(
            body_name=tool_frame.name.name
        ).result
    finally:
        simulation.stop_simulation()

    assert list(simulated) == pytest.approx(list(parked), abs=0.01)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_drive_the_joints_to_a_commanded_configuration(hsrb):
    commanded = {
        HSRBJoint.ARM_LIFT: 0.3,
        HSRBJoint.ARM_FLEX: -1.0,
        HSRBJoint.ARM_ROLL: 0.5,
        HSRBJoint.WRIST_FLEX: -0.8,
        HSRBJoint.WRIST_ROLL: 0.7,
        HSRBJoint.HEAD_PAN: 0.5,
        HSRBJoint.HEAD_TILT: -0.3,
        HSRBJoint.HAND_MOTOR: 0.8,
    }

    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _command(hsrb, commanded)
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
def test_the_torso_rises_with_half_the_arm_lift(hsrb):
    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _command(hsrb, {HSRBJoint.ARM_LIFT: 0.4})
        simulation.step_simulation(timedelta(seconds=4))
        torso_lift = simulation.simulator.get_joint_value(HSRBJoint.TORSO_LIFT).result
    finally:
        simulation.stop_simulation()

    torso_connection = hsrb._world.get_connection_by_name(HSRBJoint.TORSO_LIFT)
    assert torso_lift == pytest.approx(0.4 * torso_connection.multiplier, abs=0.01)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_arm_comes_to_rest_held_out_sideways(hsrb):
    """
    A servo too stiff for the light links it turns keeps them ringing around the
    commanded position.
    """
    commanded = _hold_the_arm_out_sideways(hsrb)

    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=2))
        held = []
        for _ in range(100):
            simulation.step_simulation(timedelta(milliseconds=10))
            held.append(
                {
                    joint_name: simulation.simulator.get_joint_value(joint_name).result
                    for joint_name in commanded
                }
            )
    finally:
        simulation.stop_simulation()

    assert held == [pytest.approx(commanded, abs=0.02)] * len(held)


# %% the physics: holding something


def _carton_between_the_open_fingers(hsrb: HSRB, carton_scale: Scale) -> Body:
    """
    Hold the hand out level in front of the HSRB, open, and stand a carton on a
    support between its fingers, as a front grasp finds an object on a table.

    :param carton_scale: The carton's width, depth and height.
    :return: The carton.
    """
    world = hsrb._world
    _command(
        hsrb,
        {
            HSRBJoint.ARM_LIFT: 0.2,
            HSRBJoint.ARM_FLEX: 0.0,
            HSRBJoint.WRIST_FLEX: -np.pi / 2,
            HSRBJoint.HAND_MOTOR: 1.0,
        },
    )
    x, y, z = world.compute_forward_kinematics_np(
        world.root, hsrb.end_effector.tool_frame
    )[:3, 3]

    support = Body(name=PrefixedName("support"))
    support_scale = Scale(0.2, 0.2, 0.2)
    support.collision = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix(reference_frame=support),
                scale=support_scale,
            )
        ],
        reference_frame=support,
    )
    carton = Body(name=PrefixedName("carton"))
    carton.collision = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix(reference_frame=carton),
                scale=carton_scale,
            )
        ],
        reference_frame=carton,
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=support,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, y=y, z=z - carton_scale.z / 2 - support_scale.z / 2
                ),
            )
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=carton,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, y=y, z=z, reference_frame=world.root
                ),
            )
        )
        ContactParameters.create_for_grasped_object().apply_to([carton])
    return carton


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_closed_hand_keeps_a_carton_while_the_arm_lifts_it(hsrb):
    """
    MuJoCo's default friction cone lets a held object slide out under its own weight,
    so the simulation resolves friction exactly, as a demo carrying objects does.
    """
    carton = _carton_between_the_open_fingers(hsrb, Scale(0.06, 0.06, 0.2))
    lift = 0.15

    simulation = MujocoSim(
        world=hsrb._world,
        headless=True,
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
        impratio=10.0,
        noslip_iterations=10,
    )
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        _command(hsrb, {HSRBJoint.HAND_MOTOR: 0.0})
        simulation.step_simulation(timedelta(seconds=1))
        grasped_height = float(
            simulation.simulator.get_body_position(body_name=carton.name.name).result[2]
        )
        _command(hsrb, {HSRBJoint.ARM_LIFT: 0.2 + lift})
        simulation.step_simulation(timedelta(seconds=3))
        lifted_height = float(
            simulation.simulator.get_body_position(body_name=carton.name.name).result[2]
        )
    finally:
        simulation.stop_simulation()

    assert lifted_height - grasped_height == pytest.approx(lift, abs=0.01)


# %% the physics: the base


def _drive_at(
    simulation: MujocoSim,
    hsrb: HSRB,
    base_velocity: tuple[float, float, float],
    duration: timedelta,
) -> None:
    """
    Command the drive with a velocity in the frame of the base for a while, then stop it
    and let it come to a standstill.

    :param base_velocity: The forward, sideways and angular velocity.
    """
    world = hsrb._world
    drive = hsrb.drive
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
def test_the_drive_moves_the_base_at_the_commanded_velocity(hsrb):
    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, hsrb, (0.2, -0.1, 0.0), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, hsrb)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((0.4, -0.2, 0.0), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_drive_turns_the_base_at_the_commanded_angular_velocity(hsrb):
    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        _drive_at(simulation, hsrb, (0.0, 0.0, 0.3), timedelta(seconds=2))
        reached = _simulated_base_pose(simulation, hsrb)
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx((0.0, 0.0, 0.6), abs=0.02)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_base_stands_still_while_the_arm_swings(hsrb):
    """
    The arm swinging out shifts the robot's weight, which must not push a base that is
    commanded to stand.
    """
    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=0.5))
        standing = _simulated_base_pose(simulation, hsrb)
        _command(
            hsrb,
            {
                HSRBJoint.ARM_LIFT: 0.5,
                HSRBJoint.ARM_FLEX: -np.pi / 2,
                HSRBJoint.ARM_ROLL: 1.5,
            },
        )
        simulation.step_simulation(timedelta(seconds=3))
        after = _simulated_base_pose(simulation, hsrb)
    finally:
        simulation.stop_simulation()

    assert after == pytest.approx(standing, abs=0.005)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_base_stands_still_while_the_arm_is_held_out_sideways(hsrb):
    """
    The arm held out to the side must not turn a base that is commanded to stand.
    """
    _hold_the_arm_out_sideways(hsrb)

    simulation = MujocoSim(world=hsrb._world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=2))
        standing = _simulated_base_pose(simulation, hsrb)
        simulation.step_simulation(timedelta(seconds=5))
        after = _simulated_base_pose(simulation, hsrb)
    finally:
        simulation.stop_simulation()

    assert after == pytest.approx(standing, abs=0.001)
