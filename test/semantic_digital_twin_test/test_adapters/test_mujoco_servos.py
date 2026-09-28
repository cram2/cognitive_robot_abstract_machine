"""
Tests for the MuJoCo actuator a position servo becomes, and for a servoed joint being
driven rather than teleported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import mujoco
import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import (
    MujocoActuator,
    MujocoBuilder,
    MujocoSim,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, PositionServo

# %% the servo's actuator


def test_the_servo_actuator_is_a_pd_law_clamped_to_the_joints_range():
    gains = ServoGains(stiffness=100.0, damping=10.0, torque_limit=5.0)
    degree_of_freedom = DegreeOfFreedom(
        name=PrefixedName("hinge"),
        limits=DegreeOfFreedomLimits(
            DerivativeMap(position=-1.0), DerivativeMap(position=1.0)
        ),
    )
    servo = PositionServo(name=PrefixedName("hinge_servo"), gains=gains)
    servo.add_dof(degree_of_freedom)

    actuator = MujocoActuator.create_servo(servo)

    assert actuator.gain_type == mujoco.mjtGain.mjGAIN_FIXED
    assert actuator.gain_parameters[0] == gains.stiffness
    assert actuator.bias_type == mujoco.mjtBias.mjBIAS_AFFINE
    assert actuator.bias_parameters[:3] == [0.0, -gains.stiffness, -gains.damping]
    assert actuator.control_range == [-1.0, 1.0]
    assert actuator.force_range == [-gains.torque_limit, gains.torque_limit]


# %% a servoed pendulum


@dataclass
class PendulumWorld:
    """
    A world with one hinge, and the handles a test needs.
    """

    world: World
    """
    The world itself.
    """

    hinge: RevoluteConnection
    """
    The hinge.
    """

    mirrored_hinge: RevoluteConnection
    """
    A second hinge following the first with a negative multiplier, sharing its degree of
    freedom.
    """


def _pendulum_world() -> PendulumWorld:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        base = Body(name=PrefixedName("base"))
        world.add_connection(FixedConnection(parent=root, child=base))
        arm = Body(name=PrefixedName("arm"))
        mirrored_arm = Body(name=PrefixedName("mirrored_arm"))
        # the two arms hang at different heights so they never touch each other
        for body, height in ((arm, 0.0), (mirrored_arm, 0.1)):
            body.collision = ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=0.1, z=height, reference_frame=body
                        ),
                        scale=Scale(0.2, 0.02, 0.02),
                    )
                ],
                reference_frame=body,
            )
        degree_of_freedom = DegreeOfFreedom(
            name=PrefixedName("hinge"),
            limits=DegreeOfFreedomLimits(
                DerivativeMap(position=-1.0), DerivativeMap(position=1.0)
            ),
        )
        world.add_degree_of_freedom(degree_of_freedom)
        hinge = RevoluteConnection(
            name=PrefixedName("hinge"),
            parent=base,
            child=arm,
            axis=Vector3.Z(reference_frame=arm),
            raw_dof=degree_of_freedom,
        )
        world.add_connection(hinge)
        mirrored_hinge = RevoluteConnection(
            name=PrefixedName("mirrored_hinge"),
            parent=base,
            child=mirrored_arm,
            axis=Vector3.Z(reference_frame=mirrored_arm),
            raw_dof=degree_of_freedom,
            multiplier=-1.0,
        )
        world.add_connection(mirrored_hinge)
    return PendulumWorld(world=world, hinge=hinge, mirrored_hinge=mirrored_hinge)


def _servoed_pendulum_world() -> PendulumWorld:
    pendulum = _pendulum_world()
    servo = PositionServo(
        name=PrefixedName("hinge_servo"),
        gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=2.0),
    )
    servo.add_dof(pendulum.hinge.raw_dof)
    with pendulum.world.modify_world():
        pendulum.world.add_actuator(servo)
    return pendulum


def test_a_position_servo_becomes_a_servo_actuator_on_the_degree_of_freedoms_joint(
    tmp_path,
):
    pendulum = _servoed_pendulum_world()

    builder = MujocoBuilder()
    builder.build_world(world=pendulum.world, file_path=str(tmp_path / "scene.xml"))

    [servo] = builder.spec.actuators
    assert servo.name == "hinge_servo"
    assert servo.target == "hinge"
    assert servo.trntype == mujoco.mjtTrn.mjTRN_JOINT
    assert servo.gainprm[0] == 50.0
    assert list(servo.forcerange) == [-2.0, 2.0]


def test_a_degree_of_freedom_without_a_servo_gets_no_actuator(tmp_path):
    pendulum = _pendulum_world()

    builder = MujocoBuilder()
    builder.build_world(world=pendulum.world, file_path=str(tmp_path / "scene.xml"))

    assert builder.spec.actuators == []


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_servoed_joint_is_driven_towards_the_world_state_not_teleported():
    """
    Writing a servoed joint's position into the world hands the servo a set point:

    the joint then moves there through the physics, and the world keeps the set point
    rather than being overwritten with the position reached so far.
    """
    pendulum = _servoed_pendulum_world()
    world = pendulum.world
    set_point = 0.5

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulator = simulation.simulator
        world.state[pendulum.hinge.raw_dof.id].position = set_point
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=simulator.step_size))
        after_one_step = simulator.get_joint_value("hinge").result
        simulation.step_simulation(timedelta(seconds=2))
        settled = simulator.get_joint_value("hinge").result
    finally:
        simulation.stop_simulation()

    assert 0.0 < after_one_step < set_point
    assert settled == pytest.approx(set_point, abs=0.02)
    assert world.state[pendulum.hinge.raw_dof.id].position == set_point


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_starting_the_simulation_holds_a_servoed_joint_where_the_world_has_it():
    """
    A freshly reset simulation leaves every control input at zero, so a joint the world
    holds elsewhere would rush to zero the moment the physics starts.
    """
    pendulum = _servoed_pendulum_world()
    world = pendulum.world
    with world.modify_world():
        world.state[pendulum.hinge.raw_dof.id].position = 0.7
    world.notify_state_change()

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        held = simulation.simulator.get_joint_value("hinge").result
    finally:
        simulation.stop_simulation()

    assert held == pytest.approx(0.7, abs=0.02)
