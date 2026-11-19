"""
Tests that goals composed of other nodes arrange them as intended, either in parallel or
in a sequence.
"""

from __future__ import annotations

from cramph.composites import Parallel, Sequence
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.tasks.drive_tasks import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPoseStraight,
    CartesianOrientation,
    CartesianPose,
    CartesianPosition,
    CartesianPositionStraight,
    HoldOrientation,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from semantic_digital_twin.robots.robot_parts import MobileBase
from krrood_spatial_math.spatial_types import Pose3Dimensional
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% goals that are a parallel of their tasks


def _create_executor(world: World) -> StatechartExecutor:
    """
    :return: An executor with motion control acting in `world`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )


def test_cartesian_pose_is_a_parallel_of_its_position_and_orientation(
    cylinder_bot_world: World,
) -> None:
    """
    A pose goal is a parallel of a position and an orientation task, in that order.
    """
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        node=(
            pose := CartesianPose(
                kinematic_chain=KinematicChain(root=cylinder_bot_world.root, tip=tip),
                goal_pose=Pose3Dimensional.from_xyz_rpy(
                    x=0.1, reference_frame=cylinder_bot_world.root
                ),
            )
        )
    )
    statechart.add_node(node=EndMotion.when_true(node=pose))
    executor.compile(statechart=statechart)

    assert isinstance(pose, Parallel)
    assert [type(task) for task in pose.nodes] == [
        CartesianPosition,
        CartesianOrientation,
    ]


def test_nodes_handed_to_a_cartesian_pose_run_beside_its_tasks(
    cylinder_bot_world: World,
) -> None:
    """
    Whatever accompanies a pose, such as a task holding the orientation or a collision
    allowance, runs in the pose itself rather than in another parallel around it.
    """
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    pose = CartesianPose(
        kinematic_chain=KinematicChain(root=cylinder_bot_world.root, tip=tip),
        goal_pose=Pose3Dimensional.from_xyz_rpy(
            x=0.1, reference_frame=cylinder_bot_world.root
        ),
    )
    hold_orientation = HoldOrientation(
        kinematic_chain=KinematicChain(root=cylinder_bot_world.root, tip=tip)
    )
    pose.add_node(node=hold_orientation)
    statechart.add_node(node=pose)
    statechart.add_node(node=EndMotion.when_true(node=pose))
    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert [type(node) for node in pose.nodes] == [
        CartesianPosition,
        CartesianOrientation,
        HoldOrientation,
    ]
    assert hold_orientation.parent_node is pose
    assert pose.last_observation_state == ObservationStateValue.TRUE


def test_cartesian_pose_observes_its_goal_reached(
    cylinder_bot_world: World,
) -> None:
    """
    A pose goal observes true once both its tasks do.
    """
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        node=(
            pose := CartesianPose(
                kinematic_chain=KinematicChain(root=cylinder_bot_world.root, tip=tip),
                goal_pose=Pose3Dimensional.from_xyz_rpy(
                    x=0.1, reference_frame=cylinder_bot_world.root
                ),
            )
        )
    )
    statechart.add_node(node=EndMotion.when_true(node=pose))
    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert all(
        task.last_observation_state == ObservationStateValue.TRUE for task in pose.nodes
    )
    assert pose.last_observation_state == ObservationStateValue.TRUE


def test_cartesian_pose_straight_is_a_parallel_of_its_tasks(
    cylinder_bot_world: World,
) -> None:
    """
    A straight-line pose goal is a parallel of a straight-line position task and an
    orientation task, in that order.
    """
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        node=(
            pose := CartesianPoseStraight(
                kinematic_chain=KinematicChain(root=cylinder_bot_world.root, tip=tip),
                goal_pose=Pose3Dimensional.from_xyz_rpy(
                    x=0.1, reference_frame=cylinder_bot_world.root
                ),
            )
        )
    )
    statechart.add_node(node=EndMotion.when_true(node=pose))
    executor.compile(statechart=statechart)

    assert isinstance(pose, Parallel)
    assert [type(task) for task in pose.nodes] == [
        CartesianPositionStraight,
        CartesianOrientation,
    ]


# %% goals running their steps in a sequence


def test_differential_drive_goal_runs_its_steps_in_one_sequence(
    cylinder_bot_diff_world: World,
) -> None:
    executor = _create_executor(cylinder_bot_diff_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        node=(
            goal := DifferentialDriveBaseGoal(
                mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                    MobileBase
                )[0],
                goal_pose=Pose3Dimensional.from_xyz_rpy(
                    x=1, reference_frame=cylinder_bot_diff_world.root
                ),
            )
        )
    )
    statechart.add_node(node=EndMotion.when_true(node=goal))
    executor.compile(statechart=statechart)

    assert not isinstance(goal, Sequence)
    assert goal.nodes == [goal.sequence]
    assert isinstance(goal.sequence, Sequence)
    assert len(goal.sequence.nodes) == 3


def test_differential_drive_goal_is_a_step_without_an_attempt(
    cylinder_bot_diff_world: World,
) -> None:
    executor = _create_executor(cylinder_bot_diff_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        node=(
            plan := Sequence(
                nodes=[
                    goal := DifferentialDriveBaseGoal(
                        mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                            MobileBase
                        )[0],
                        goal_pose=Pose3Dimensional.from_xyz_rpy(
                            x=1, reference_frame=cylinder_bot_diff_world.root
                        ),
                    )
                ]
            )
        )
    )
    statechart.add_node(node=EndMotion.when_true(node=plan))
    executor.compile(statechart=statechart)

    assert plan.nodes == [goal]
