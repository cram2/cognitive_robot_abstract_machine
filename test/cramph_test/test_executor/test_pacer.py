from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter

import numpy as np

import pytest

from cramph.exceptions import NonPositiveRealTimeFactorError
from cramph.executor import (
    NoPacing,
    RealTimePacer,
    SimulationPacer,
    SteppedSimulationPacer,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body, PositionServo

from ...pytest_environment import runs_in_continuous_integration


def test_simulation_pacer_timing_real_time(monkeypatch) -> None:
    pacer = SimulationPacer(real_time_factor=1.0)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert np.isclose(perf_counter() - start_time, 1.0, rtol=0.01)


def test_simulation_pacer_timing_2x(monkeypatch) -> None:
    pacer = SimulationPacer(real_time_factor=2.0)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    actual = perf_counter() - start_time
    assert np.isclose(actual, 0.5, rtol=0.01)


def test_simulation_pacer_timing_halfx(monkeypatch) -> None:
    pacer = SimulationPacer(real_time_factor=0.5)
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert np.isclose(perf_counter() - start_time, 2.0, rtol=0.01)


def test_no_pacing_does_not_wait() -> None:
    pacer = NoPacing()
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert perf_counter() - start_time < 0.01


def test_a_pacer_has_no_target_frequency_before_it_paces_a_loop() -> None:
    assert NoPacing().target_frequency is None


def test_real_time_pacer_holds_the_target_frequency() -> None:
    pacer = RealTimePacer()
    pacer.target_frequency = 50
    start_time = perf_counter()
    for i in range(50):
        pacer.sleep()
    assert np.isclose(perf_counter() - start_time, 1.0, rtol=0.01)


def test_a_simulation_cannot_be_configured_to_stand_still() -> None:
    with pytest.raises(NonPositiveRealTimeFactorError):
        SimulationPacer(real_time_factor=0.0)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_stepped_simulation_pacer_advances_the_physics_one_cycle_per_sleep() -> None:
    """
    A box dropped from a metre falls under the simulation's gravity exactly as far as
    the paced cycles add up to, so the physics and the loop stay in lockstep.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        box = Body(name=PrefixedName("box"))
        box.collision = ShapeCollection(
            [Box(origin=HomogeneousTransformationMatrix(), scale=Scale(0.1, 0.1, 0.1))],
            reference_frame=box,
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=root
                ),
            )
        )
    cycles, frequency = 25, 50

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        pacer = SteppedSimulationPacer(simulation)
        pacer.target_frequency = frequency
        for _ in range(cycles):
            pacer.sleep()
        height = simulation.simulator.get_body_position(body_name="box").result[2]
    finally:
        simulation.stop_simulation()

    fallen = 0.5 * 9.81 * (cycles / frequency) ** 2
    assert height == pytest.approx(1.0 - fallen, abs=0.01)


# %% waiting for the robot to come to a standstill


@dataclass
class JointThatStopsAfterAWhile:
    """
    A joint that travels a fixed distance per step for a while and then stays put, the
    way a servo settles or is held up by something it has run into.
    """

    steps_still_moving: int
    """
    How many more steps the joint travels before it stays put.
    """

    travel_per_step: float = 0.05
    """
    How far it travels per step while it is still moving.
    """

    position: float = 0.0
    """
    Where it stands now.
    """

    def step(self) -> None:
        """
        Let one step of physics pass.
        """
        if self.steps_still_moving == 0:
            return
        self.position += self.travel_per_step
        self.steps_still_moving -= 1


@dataclass
class ActuatorLengthsRead:
    """
    What a simulator answers a bulk read of actuator lengths with.
    """

    result: dict[str, float]
    """
    The length read per actuator name.
    """


@dataclass
class SimulationOfSettlingJoints:
    """
    Stands in for a stepped simulation, reporting joints that come to a standstill
    without any physics being computed.
    """

    joints: dict[str, JointThatStopsAfterAWhile]
    """
    The joints it reports, by name.
    """

    steps: int = 0
    """
    How many steps of physics it has been asked for.
    """

    @property
    def world(self) -> World:
        """
        :return: A world whose servos drive exactly the joints reported.
        """
        world = World()
        for joint_name in self.joints:
            servo = PositionServo(
                name=PrefixedName(f"{joint_name}_servo"),
                gains=ServoGains(stiffness=1.0, damping=1.0, torque_limit=1.0),
            )
            servo.add_dof(DegreeOfFreedom(name=PrefixedName(joint_name)))
            world.actuators.append(servo)
        return world

    @property
    def simulator(self) -> SimulationOfSettlingJoints:
        """
        :return: Itself, since it answers the reads a simulator would.
        """
        return self

    def step_simulation(self, duration: timedelta) -> None:
        """
        :param duration: Ignored; every call is one step of this stand-in.
        """
        self.steps += 1
        for joint in self.joints.values():
            joint.step()

    def get_actuators_lengths(self, actuator_names: list[str]) -> ActuatorLengthsRead:
        """
        :param actuator_names: The servos to read, each named after its joint.
        :return: Where the joint of each of them stands.
        """
        return ActuatorLengthsRead(
            result={
                actuator_name: self.joints[
                    actuator_name.removesuffix("_servo")
                ].position
                for actuator_name in actuator_names
            }
        )


def settling_pacer(
    simulation: SimulationOfSettlingJoints, frequency: float = 50, seconds: float = 10
) -> SteppedSimulationPacer:
    """
    :param simulation: The simulation to step.
    :param frequency: The rate the loop runs at.
    :param seconds: The simulated time the robot is given to come to a standstill.
    :return: A pacer driving that simulation.
    """
    pacer = SteppedSimulationPacer(
        simulation, standstill_time_limit=timedelta(seconds=seconds)
    )
    pacer.target_frequency = frequency
    return pacer


def test_waiting_ends_once_the_slowest_joint_has_stopped_moving() -> None:
    """
    The physics is stepped while anything is still travelling, and one step further,
    which is the step that shows nothing moved.
    """
    simulation = SimulationOfSettlingJoints(
        joints={
            "quick_joint": JointThatStopsAfterAWhile(steps_still_moving=2),
            "slow_joint": JointThatStopsAfterAWhile(steps_still_moving=6),
        }
    )

    settling_pacer(simulation).wait_for_standstill()

    assert simulation.steps == 7
    assert simulation.joints["slow_joint"].position == pytest.approx(0.3)


def test_a_robot_that_never_stops_moving_is_waited_for_only_so_long(caplog) -> None:
    simulation = SimulationOfSettlingJoints(
        joints={"restless_joint": JointThatStopsAfterAWhile(steps_still_moving=1000)}
    )

    settling_pacer(simulation, frequency=50, seconds=0.1).wait_for_standstill()

    assert simulation.steps == 5
    assert "still moving" in caplog.text


def test_a_simulation_without_servos_is_not_stepped_at_all() -> None:
    """
    A joint written straight into the world is already where it was put, so there is
    nothing to wait for.
    """
    simulation = SimulationOfSettlingJoints(joints={})

    settling_pacer(simulation).wait_for_standstill()

    assert simulation.steps == 0


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_waiting_covers_the_servos_moving_a_robots_base() -> None:
    """
    A base still rolling when its commanded velocity drops to zero is waited for like
    any other servoed joint, so the robot stands still once the wait is over.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    robot = RobotSpecification(PR2).spawn(world)
    drive = robot.drive

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        pacer = settling_pacer(simulation)
        world.state[drive.x_velocity.id].velocity = 0.3
        world.notify_state_change()
        for _ in range(25):
            pacer.sleep()
        world.state[drive.x_velocity.id].velocity = 0.0
        world.notify_state_change()
        pacer.wait_for_standstill()
        waited_until = simulation.simulator.get_body_position(
            body_name=robot.root.name.name
        ).result[0]
        simulation.step_simulation(timedelta(seconds=1))
        a_second_later = simulation.simulator.get_body_position(
            body_name=robot.root.name.name
        ).result[0]
    finally:
        simulation.stop_simulation()

    assert waited_until == pytest.approx(0.15, abs=0.02)
    assert a_second_later == pytest.approx(waited_until, abs=1e-3)
