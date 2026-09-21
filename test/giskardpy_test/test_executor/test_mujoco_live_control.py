"""
Giskard's own control loop driving a physically simulated Tracy live: each control
cycle's command becomes the servos' set point through the world state, and the physics
steps in lockstep between cycles.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

from datetime import timedelta

import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = [
    pytest.mark.skipif(
        not tracy_installed(), reason="iai_tracy_description is not installed"
    ),
    pytest.mark.skipif(
        not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
    ),
]


@pytest.fixture
def parked_tracy() -> Tracy:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    robot = RobotSpecification(Tracy).spawn(world)
    for arm in robot.get_arms():
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    return robot


def test_the_simulated_arm_reaches_the_pose_giskard_commands_live(parked_tracy):
    """
    A goal ticked by Giskard against the world is reached by the arm in the physics, not
    merely in the world's own belief: every cycle's command is handed to the servos as
    their set point and the physics steps in between.
    """
    control_frequency = 50
    tick_limit = 500
    tracking_tolerance = 0.02
    world = parked_tracy._world
    tool_frame = parked_tracy.left_arm.end_effector.tool_frame
    start = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]
    goal_point = Point3(start[0], start[1], start[2] + 0.15, reference_frame=world.root)
    reach = CartesianPosition(
        name="reach", root_link=world.root, tip_link=tool_frame, goal_point=goal_point
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([reach, EndMotion.when_true(reach)])
    controller_config = QPControllerConfig(target_frequency=control_frequency)

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        executor = Executor(
            context=MotionStatechartContext(
                world=world, qp_controller_config=controller_config
            ),
            pacer=SteppedSimulationPacer(simulation),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(timeout=tick_limit)
        simulation.step_simulation(timedelta(seconds=1))
        simulated = numpy.array(
            simulation.simulator.get_body_position(
                body_name=tool_frame.name.name
            ).result
        )
    finally:
        simulation.stop_simulation()

    assert motion_statechart.is_end_motion()
    assert numpy.linalg.norm(simulated - goal_point.to_np()[:3]) <= tracking_tolerance
