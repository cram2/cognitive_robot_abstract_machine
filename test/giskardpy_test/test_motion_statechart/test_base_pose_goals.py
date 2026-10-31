"""
Tests for the goals putting the base of a robot at a pose, each built for a robot the
same way, and moving the base only along the floor and about the vertical.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.goals.base_pose_goals import (
    MobileBasePoseGoal,
    DifferentialDriveBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.exceptions import RobotWithoutMobileBaseError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.drive_tasks import (
    BaseOrientation,
    BasePosition,
    DrivenBaseRotation,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% building a goal for a robot


def _target(world: World) -> Pose:
    """
    :return: A pose for the base, away from the origin of `world` and turned.
    """
    return Pose.from_xyz_rpy(1, 2, 0, yaw=0.5, reference_frame=world.root)


def test_a_holonomic_base_goal_moves_the_root_of_the_robot(
    pr2_world_copy: World,
) -> None:
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    target = _target(pr2_world_copy)

    goal = HolonomicBaseGoal.moving_base_of(robot, target, threshold=0.05)

    assert goal.mobile_base is robot.mobile_base
    assert goal.goal_pose is target
    assert goal.threshold == 0.05


def test_a_differential_drive_goal_drives_the_drive_of_the_robot(
    stretch_world_copy: World,
) -> None:
    robot = stretch_world_copy.get_semantic_annotations_by_type(Stretch)[0]
    target = _target(stretch_world_copy)

    goal = DifferentialDriveBaseGoal.moving_base_of(robot, target, threshold=0.05)

    assert goal.mobile_base is robot.mobile_base
    assert goal.goal_pose is target
    assert goal.threshold == 0.05


@pytest.mark.parametrize("goal_type", [HolonomicBaseGoal, DifferentialDriveBaseGoal])
def test_a_drive_goal_cannot_move_a_robot_without_a_mobile_base(
    self_collision_bot_world: World, goal_type: type[MobileBasePoseGoal]
) -> None:
    robot = self_collision_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]

    with pytest.raises(RobotWithoutMobileBaseError) as raised:
        goal_type.moving_base_of(robot, _target(self_collision_bot_world), 0.05)

    assert raised.value.robot is robot


def test_setting_the_odometry_writes_the_connection_above_the_robot(
    pr2_world_copy: World,
) -> None:
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    target = _target(pr2_world_copy)

    goal = SetOdometry.moving_base_of(robot, target, threshold=0.05)

    assert goal.odom_connection is robot.root.parent_connection
    np.testing.assert_allclose(goal.base_pose.to_np(), target.to_np())
    assert goal.base_pose.reference_frame is target.reference_frame


# %% moving the base along the floor


def _compiled(world: World, goal: StatechartNode) -> StatechartExecutor:
    """
    :return: An executor with motion control, compiled with `goal` ending the motion.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(goal)
    statechart.add_node(EndMotion.when_true(goal))
    executor.compile(statechart=statechart)
    return executor


def _equality_constraint_counts(goal: MobileBasePoseGoal) -> list[int]:
    """
    :return: How many equality constraints each task below `goal` writes, position
        tasks first.
    """
    tasks = [node for node in goal.descendants if isinstance(node, BasePosition)] + [
        node for node in goal.descendants if isinstance(node, DrivenBaseRotation)
    ]
    return [len(task.constraint_collection.equality_constraints) for task in tasks]


def test_a_holonomic_base_goal_constrains_only_x_y_and_yaw(
    cylinder_bot_world: World,
) -> None:
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    goal = HolonomicBaseGoal.moving_base_of(robot, _target(cylinder_bot_world), 0.01)

    _compiled(cylinder_bot_world, goal)

    assert _equality_constraint_counts(goal) == [2, 1]


def test_a_differential_drive_goal_constrains_only_x_y_and_yaw(
    cylinder_bot_diff_world: World,
) -> None:
    goal = DifferentialDriveBaseGoal(
        mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
            MobileBase
        )[0],
        goal_pose=_target(cylinder_bot_diff_world),
    )

    _compiled(cylinder_bot_diff_world, goal)

    assert _equality_constraint_counts(goal) == [2, 2, 1, 1, 1]


def test_a_holonomic_base_goal_reaches_the_pose_on_the_floor_below_its_target(
    cylinder_bot_world: World,
) -> None:
    """
    A target above the floor or tilted is reached once the base stands below it facing
    its way, which the base can do without lifting or tilting.
    """
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    target = Pose.from_xyz_rpy(
        1,
        -0.5,
        0.3,
        roll=0.2,
        pitch=-0.1,
        yaw=2.0,
        reference_frame=cylinder_bot_world.root,
    )
    goal = HolonomicBaseGoal.moving_base_of(robot, target, 0.01)
    executor = _compiled(cylinder_bot_world, goal)

    executor.tick_until_end()

    base_pose = robot.root.global_pose
    np.testing.assert_allclose(
        [float(base_pose.x), float(base_pose.y), float(base_pose.yaw)],
        [float(target.x), float(target.y), float(target.yaw)],
        atol=0.01,
    )


# %% turning the base


def _peak_yaw_per_tick_towards(
    world: World, yaw: float, reference_velocity: float = 0.05
) -> float:
    """
    Turns the base of the robot in `world` on the spot and measures how fast it turns.

    :param yaw: How far the goal is turned away from the current orientation of the
        base.
    :param reference_velocity: The reference velocity of the turn in rad/s, far enough
        below the velocity limit of the drive that the limit does not set the speed.
    :return: The largest angle the base turned within one control cycle.
    """
    base = world.get_semantic_annotations_by_type(MinimalRobot)[0].root
    turn = BaseOrientation(
        base=base,
        goal_pose=Pose.from_xyz_rpy(yaw=yaw, reference_frame=world.root),
        parent_T_goal_reference_frame=HomogeneousTransformationMatrix(),
        reference_velocity=reference_velocity,
    )
    executor = _compiled(world, turn)

    yaws = [float(base.global_pose.yaw)]
    while not executor.statechart.is_ended():
        executor.tick()
        yaws.append(float(base.global_pose.yaw))
    return float(np.max(np.abs(np.diff(np.unwrap(yaws)))))


def test_a_base_turns_as_fast_towards_a_far_away_yaw_as_towards_a_close_one(
    cylinder_bot_world: World, orientation_speed_ratio_tolerance: float
) -> None:
    """
    The angular speed towards a yaw goal is set by its reference velocity, not by how
    far away the goal is.
    """
    derivatives = Derivatives.range(Derivatives.position, Derivatives.jerk)
    initial_state = {
        derivative: cylinder_bot_world.state.get_derivative(derivative).copy()
        for derivative in derivatives
    }
    peak_speeds = []
    for yaw in (np.radians(30), np.radians(150)):
        for derivative in derivatives:
            cylinder_bot_world.state.set_derivative(
                derivative, initial_state[derivative]
            )
        cylinder_bot_world.notify_state_change()
        peak_speeds.append(_peak_yaw_per_tick_towards(cylinder_bot_world, yaw))

    assert peak_speeds[1] <= orientation_speed_ratio_tolerance * peak_speeds[0]
