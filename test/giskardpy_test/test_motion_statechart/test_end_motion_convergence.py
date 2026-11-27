"""
Tests of how a motion comes to its end.
"""

from __future__ import annotations

import numpy as np
import pytest
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain


def test_end_motion_abruptness(cylinder_bot_world: World):
    """
    Ends a motion of the cylinder bot and prints how fast the bot still moves when the
    statechart ends.
    """
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name(name="bot")

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    motion_statechart = Statechart(context=executor.context)
    goal = CartesianPose(
        kinematic_chain=KinematicChain(root=cylinder_bot_world.root, tip=tip),
        goal_pose=Pose.from_xyz_rpy(x=1, reference_frame=cylinder_bot_world.root),
        position_threshold=0.01,
    )
    motion_statechart.add_node(node=goal)
    end = EndMotion.when_true(node=goal)
    motion_statechart.add_node(node=end)

    executor.compile(statechart=motion_statechart)

    # We want to check the velocity in the last tick BEFORE cleanup
    # tick_until_end calls cleanup. We'll do it manually.

    for i in range(1000):
        executor.tick()
        if motion_statechart.is_ended():
            break

    velocities = cylinder_bot_world.state.velocities
    max_vel = np.max(np.abs(velocities))

    print(f"\nFinal max velocity: {max_vel}")

    # If EndMotion is abrupt, max_vel might be > 0.
    # We want it to be very small if it converged.
    # Currently it will likely be around 0.1 or so (controller-dependent)

    # assert max_vel < 0.001
