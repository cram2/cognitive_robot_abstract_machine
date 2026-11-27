"""
Motion control of a world a physics moves, which leaves every degree of freedom outside
the robots to the physics.
"""

from __future__ import annotations

import pytest

from ..pytest_environment import runs_in_continuous_integration

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor, SteppedSimulationPacer
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.tasks.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

# %% a door the robot opens


@pytest.fixture
def hsrb_before_the_cracked_open_wardrobe(apartment_world_copy) -> World:
    """
    The HSRB in the apartment, in front of the wardrobe, out of reach of the handle of
    its cracked open left door.
    """
    world = apartment_world_copy
    handle = world.get_body_by_name("wardrobe_door_left_handle")
    handle.parent_connection.parent.parent_connection.position = 0.2
    standing_pose = Pose.from_xyz_rpy(x=-0.9, reference_frame=handle)
    RobotSpecification(
        HSRB,
        localization_pose=world.transform(standing_pose, world.root).homogeneous_matrix,
    ).spawn(world)
    return world


def test_the_controller_leaves_a_door_hinge_to_the_physics(
    hsrb_before_the_cracked_open_wardrobe,
):
    """
    The controller still commands the hinge, but only the physics, through the gripper
    pulling the door, may move it, so a door the gripper has not reached yet stays put.
    """
    world = hsrb_before_the_cracked_open_wardrobe
    [robot] = world.get_semantic_annotations_by_type(HSRB)
    handle = world.get_body_by_name("wardrobe_door_left_handle")
    hinge = handle.parent_connection.parent.parent_connection
    start = hinge.position
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        executor = StatechartExecutor(
            context=StatechartContext(world=world),
            pacer=SteppedSimulationPacer(simulation=simulation),
            extensions=[MotionControl()],
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(
            node=ArticulateGraspedMechanism(
                tip_link=robot.all_arms[0].end_effector.tool_frame,
                environment_link=handle,
                goal_position=hinge.dof.limits.upper.position,
            )
        )
        executor.compile(statechart=statechart)
        for _ in range(20):
            executor.tick()
            executor.pacer.sleep()
    finally:
        simulation.stop_simulation()

    assert world.state[hinge.dof.id].velocity != 0.0
    assert hinge.position == pytest.approx(start)
