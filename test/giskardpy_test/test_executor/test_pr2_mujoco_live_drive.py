"""
Giskard driving a physically simulated PR2's base live: each control cycle's velocity
command becomes the drive servos' set point, and the odometry giskard reads back is
where the physics has moved the base.
"""

from __future__ import annotations

import numpy
import pytest
from scipy.spatial.transform import Rotation

from ...pytest_environment import runs_in_continuous_integration

from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)


@pytest.mark.parametrize(
    "world_T_odom",
    [
        HomogeneousTransformationMatrix(),
        HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=-0.5, yaw=numpy.pi / 2),
    ],
    ids=["odom_at_the_root", "localized_odom"],
)
def test_the_simulated_base_reaches_the_pose_giskard_commands_live(world_T_odom):
    """
    A base pose goal in the frame of the world's root is reached by the base in the
    physics, with the odom placed at the root or away from and turned against it.
    """
    control_frequency = 50
    tick_limit = 1000
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    pr2 = RobotSpecification(PR2, world_T_odom=world_T_odom).spawn(world)
    world_T_start = world.compute_forward_kinematics_np(world.root, pr2.root)
    goal = Pose.from_xyz_rpy(
        x=world_T_start[0, 3] + 0.5,
        y=world_T_start[1, 3] + 0.3,
        yaw=Rotation.from_matrix(world_T_start[:3, :3]).as_euler("xyz")[2] + 0.5,
        reference_frame=world.root,
    )
    drive_to_goal = CartesianPose(
        root_link=world.root, tip_link=pr2.root, goal_pose=goal
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([drive_to_goal, EndMotion.when_true(drive_to_goal)])

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        pacer = SteppedSimulationPacer(simulation)
        executor = Executor(
            context=MotionStatechartContext(
                world=world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=control_frequency
                ),
            ),
            pacer=pacer,
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(timeout=tick_limit)
        executor.set_velocity_acceleration_jerk_to_zero()
        world.notify_state_change()
        pacer.wait_for_standstill()
        x, y, _ = simulation.simulator.get_body_position(
            body_name=pr2.root.name.name
        ).result
        w, i, j, k = simulation.simulator.get_body_quaternion(
            body_name=pr2.root.name.name
        ).result
        world_T_believed = world.compute_forward_kinematics_np(world.root, pr2.root)
    finally:
        simulation.stop_simulation()

    goal_matrix = goal.to_np()
    simulated_yaw = Rotation.from_quat([i, j, k, w]).as_euler("xyz")[2]
    goal_yaw = Rotation.from_matrix(goal_matrix[:3, :3]).as_euler("xyz")[2]
    assert motion_statechart.is_end_motion()
    assert [x, y] == pytest.approx(goal_matrix[:2, 3].tolist(), abs=0.03)
    assert simulated_yaw == pytest.approx(goal_yaw, abs=0.03)
    assert world_T_believed[:2, 3].tolist() == pytest.approx([x, y], abs=0.005)
