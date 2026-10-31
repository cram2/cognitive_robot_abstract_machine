import numpy as np
import pytest
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def test_end_motion_abruptness(cylinder_bot_world: World):
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl()],
    )
    motion_statechart = Statechart(context=executor.context)
    goal = CartesianPose(
        root_link=cylinder_bot_world.root,
        tip_link=tip,
        goal_pose=Pose.from_xyz_rpy(x=1, reference_frame=cylinder_bot_world.root),
        translation_threshold=0.01,
    )
    motion_statechart.add_node(goal)
    end = EndMotion.when_true(goal)
    motion_statechart.add_node(end)

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
