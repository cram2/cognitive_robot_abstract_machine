"""
Tests for a mobile robot performing a plan in a physical simulation, where its own
servos move its joints and its drive moves its base across the floor.
"""

from __future__ import annotations

import pytest

from ...pytest_environment import runs_in_continuous_integration

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def simulated_hsrb() -> tuple[HSRB, MujocoSim]:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    hsrb = RobotSpecification(HSRB).spawn(world)
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    yield hsrb, simulation
    simulation.stop_simulation()


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_hsrb_raises_its_torso_and_drives_to_where_it_is_sent(simulated_hsrb):
    hsrb, simulation = simulated_hsrb
    world = hsrb._world
    goal = Pose.from_xyz_rpy(1.0, 0.5, 0.0, yaw=0.5, reference_frame=world.root)
    raised = hsrb.mobile_base.torso.get_joint_state_by_type(TorsoState.HIGH)
    plan = sequential(
        [
            ParkArmsAction(hsrb.all_arms),
            MoveTorsoAction(TorsoState.HIGH),
            NavigateAction(goal),
        ],
        context=Context(world=world, robot=hsrb, evaluate_conditions=False),
    ).plan

    with PhysicallySimulatedRobot(simulation=simulation):
        plan.perform()

    world_T_base = world.compute_forward_kinematics_np(world.root, hsrb.root)
    simulated_torso_lift = {
        connection.name.name: simulation.simulator.get_joint_value(
            connection.name.name
        ).result
        for connection in raised.connections
    }
    assert world_T_base[:3, 3].tolist() == pytest.approx(
        goal.position.to_np()[:3].tolist(), abs=0.02
    )
    assert simulated_torso_lift == pytest.approx(
        {
            connection.name.name: target
            for connection, target in zip(raised.connections, raised.target_values)
        },
        abs=0.02,
    )
