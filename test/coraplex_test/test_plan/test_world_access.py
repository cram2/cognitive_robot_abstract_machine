"""
Tests that the nodes a pick-up runs into say what they change and read in the world.
"""

from __future__ import annotations

import pytest

from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.composites import Sequence
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    HoldPose,
)
from krrood_spatial_math.spatial_types import Pose2D
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from ...plan_running import run_plan

# %% a pick-up that ran


@pytest.fixture
def ran_pick_up(pr2_apartment_context):
    """
    The PR2 that drove to the counter and picked up the milk, the plan it ran, and the
    milk.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(
                planar_goal_pose=Pose2D.from_xy_yaw(
                    x=1.5, y=2.4, reference_frame=world.root
                )
            ),
            PickUpAction(
                grasp_candidate=milk.grasp_candidates()[0], arm=robot.right_arm
            ),
        ]
    )
    run_plan(plan, extensions)
    return robot, plan, milk


def first_of(holder, node_type: type):
    """
    :return: The first node of `node_type` below `holder`.
    """
    return next(node for node in holder.descendants if isinstance(node, node_type))


def planar_pose_of_the_base(robot) -> frozenset:
    """
    :return: The degrees of freedom placing the base of `robot` on the floor.
    """
    drive = robot.root.parent_connection
    return frozenset({drive.x, drive.y, drive.yaw})


# %% what the nodes declare


def test_setting_the_odometry_moves_the_degrees_of_freedom_of_the_drive(
    ran_pick_up,
) -> None:
    """
    Setting the odometry moves the degrees of freedom of the robot's drive.
    """
    robot, plan, milk = ran_pick_up
    set_odometry = first_of(plan, SetOdometry)

    assert set_odometry.compute_world_access().moved_degrees_of_freedom == frozenset(
        set_odometry.odom_connection.dofs
    )


def test_holding_the_base_holds_and_does_not_move_its_degrees_of_freedom(
    ran_pick_up,
) -> None:
    """
    Holding the base holds its degrees of freedom and moves none of them.
    """
    robot, plan, milk = ran_pick_up
    access = first_of(plan, HoldPose).compute_world_access()

    assert access.moved_degrees_of_freedom == frozenset()
    assert access.held_degrees_of_freedom >= planar_pose_of_the_base(robot)


def test_a_goal_bound_relative_to_the_robot_reads_where_the_base_stands(
    ran_pick_up,
) -> None:
    """
    A goal bound relative to the robot reads the degrees of freedom that place the base.
    """
    robot, plan, milk = ran_pick_up
    access = first_of(
        first_of(plan, ReachAction), CartesianPosition
    ).compute_world_access()

    assert access.read_degrees_of_freedom >= planar_pose_of_the_base(robot)
    assert not access.moved_degrees_of_freedom & planar_pose_of_the_base(robot)


def test_moving_a_branch_reattaches_its_root(ran_pick_up) -> None:
    """
    Moving a branch reattaches the root of that branch.
    """
    robot, plan, milk = ran_pick_up
    move_branch = first_of(plan, MoveBranch)

    assert move_branch.compute_world_access().reparented_frames == frozenset(
        {move_branch.branch_root}
    )
