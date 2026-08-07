"""
Standalone-controller coverage for Stretch.

This runs the real QP controller, behaviour tree and robot description in a closed loop
with no hardware, which is the closest offline proxy for whether a goal the demo sends
will actually be reached on the robot.
"""

import numpy as np
import pytest

from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchStandaloneInterface,
)
from giskardpy.middleware.ros2.utils.utils_for_tests import StretchTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.world_description.connections import DifferentialDrive


@pytest.fixture()
def default_joint_state():
    """
    A neutral arm configuration to seed each motion from.
    """
    return {
        "joint_lift": 0.5,
        "joint_arm_l0": 0.0,
        "joint_arm_l1": 0.0,
        "joint_arm_l2": 0.0,
        "joint_arm_l3": 0.0,
        "joint_wrist_yaw": 1.0,
        "joint_head_pan": 0.0,
        "joint_head_tilt": 0.0,
    }


@pytest.fixture()
def robot():
    tester = StretchTester()
    try:
        yield tester
    finally:
        tester.print_stats()


# %% controller setup


def test_standalone_interface_controls_the_base_drive(giskard: StretchTester):
    """
    The standalone interface registers the base drive it resolved from the world, so the
    controller can actually move the base rather than silently holding it.
    """
    drive = giskard.api.world.get_connections_by_type(DifferentialDrive)[0]

    assert drive.has_hardware_interface


def test_standalone_interface_controls_every_declared_joint(giskard: StretchTester):
    """
    Every joint the interface declares ends up controlled; a name that no longer matches
    the robot description would otherwise fail only once the robot is running.
    """
    interface = StretchStandaloneInterface()

    for joint_name in interface.controlled_joint_names(giskard.api.world):
        assert giskard.api.world.get_connection_by_name(
            joint_name
        ).has_hardware_interface


# %% goal convergence


LIFT_SEED_HEIGHT = 0.5
"""
Lift position the motions start from, matching ``default_joint_state``.
"""

LIFT_GOAL_HEIGHT = 0.9
"""
Lift position the motions drive to.
"""


def test_lift_goal_moves_the_tool_frame(giskard: StretchTester):
    """
    A joint goal reaches the controller and moves the arm: raising the lift raises the
    tool frame by the commanded amount.
    """
    height_before = giskard.compute_fk_pose(
        "base_link", "link_grasp_center"
    ).pose.position.z

    motion_statechart = MotionStatechart()
    motion_statechart.add_node(
        lift_goal := JointPositionList(
            goal_state=JointState.from_str_dict(
                {"joint_lift": LIFT_GOAL_HEIGHT}, giskard.api.world
            )
        )
    )
    motion_statechart.add_node(EndMotion.when_true(lift_goal))
    giskard.api.execute(motion_statechart)

    height_after = giskard.compute_fk_pose(
        "base_link", "link_grasp_center"
    ).pose.position.z
    np.testing.assert_allclose(
        height_after - height_before, LIFT_GOAL_HEIGHT - LIFT_SEED_HEIGHT, atol=1e-2
    )


def test_local_minimum_ends_a_motion(giskard: StretchTester):
    """
    The demo accepts a local minimum as success for its cartesian stage, so a motion
    ended only by :class:`LocalMinimumReached` has to terminate on its own and still
    arrive at the goal.
    """
    height_before = giskard.compute_fk_pose(
        "base_link", "link_grasp_center"
    ).pose.position.z

    motion_statechart = MotionStatechart()
    motion_statechart.add_node(
        JointPositionList(
            goal_state=JointState.from_str_dict(
                {"joint_lift": LIFT_GOAL_HEIGHT}, giskard.api.world
            )
        )
    )
    motion_statechart.add_node(
        local_minimum := LocalMinimumReached(joint_convergence_threshold=0.025)
    )
    motion_statechart.add_node(EndMotion.when_true(local_minimum))

    giskard.api.execute(motion_statechart)

    height_after = giskard.compute_fk_pose(
        "base_link", "link_grasp_center"
    ).pose.position.z
    np.testing.assert_allclose(
        height_after - height_before, LIFT_GOAL_HEIGHT - LIFT_SEED_HEIGHT, atol=5e-2
    )
