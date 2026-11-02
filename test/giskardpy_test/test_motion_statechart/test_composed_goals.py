import json

from cramph.composites import Parallel, Sequence
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.goals.base_pose_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.cartesian_goals import CartesianPoseStraight
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPosition,
    CartesianPositionStraight,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
    CartesianVelocityLimit,
)
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from ..motion_control_context import create_context_with_motion_control

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
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        pose := CartesianPose(
            root_link=cylinder_bot_world.root,
            tip_link=tip,
            goal_pose=Pose.from_xyz_rpy(x=0.1, reference_frame=cylinder_bot_world.root),
        )
    )
    statechart.add_node(EndMotion.when_true(pose))
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
    Whatever accompanies a pose, such as a speed cap or a collision allowance, runs in
    the pose itself rather than in another parallel around it.
    """
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    pose = CartesianPose(
        root_link=cylinder_bot_world.root,
        tip_link=tip,
        goal_pose=Pose.from_xyz_rpy(x=0.1, reference_frame=cylinder_bot_world.root),
    )
    speed_cap = CartesianPositionVelocityLimit(
        root_link=cylinder_bot_world.root, tip_link=tip
    )
    pose.add_node(speed_cap)
    statechart.add_node(pose)
    statechart.add_node(EndMotion.when_true(pose))
    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert [type(node) for node in pose.nodes] == [
        CartesianPosition,
        CartesianOrientation,
        CartesianPositionVelocityLimit,
    ]
    assert speed_cap.parent_node is pose
    assert pose.last_observation_state == ObservationStateValues.TRUE


def test_cartesian_pose_observes_its_goal_reached(
    cylinder_bot_world: World,
) -> None:
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        pose := CartesianPose(
            root_link=cylinder_bot_world.root,
            tip_link=tip,
            goal_pose=Pose.from_xyz_rpy(x=0.1, reference_frame=cylinder_bot_world.root),
        )
    )
    statechart.add_node(EndMotion.when_true(pose))
    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert all(
        task.last_observation_state == ObservationStateValues.TRUE
        for task in pose.nodes
    )
    assert pose.last_observation_state == ObservationStateValues.TRUE


def test_cartesian_pose_straight_is_a_parallel_of_its_tasks(
    cylinder_bot_world: World,
) -> None:
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    executor = _create_executor(cylinder_bot_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        pose := CartesianPoseStraight(
            root_link=cylinder_bot_world.root,
            tip_link=tip,
            goal_pose=Pose.from_xyz_rpy(x=0.1, reference_frame=cylinder_bot_world.root),
        )
    )
    statechart.add_node(EndMotion.when_true(pose))
    executor.compile(statechart=statechart)

    assert isinstance(pose, Parallel)
    assert [type(task) for task in pose.nodes] == [
        CartesianPositionStraight,
        CartesianOrientation,
    ]


def test_velocity_limit_has_its_two_limits_after_a_json_round_trip(
    cylinder_bot_world: World,
) -> None:
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    statechart = Statechart(
        context=create_context_with_motion_control(cylinder_bot_world)
    )
    statechart.add_node(
        CartesianVelocityLimit(root_link=cylinder_bot_world.root, tip_link=tip)
    )
    json_data = json.loads(json.dumps(statechart.to_json()))
    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_world)
    executor = _create_executor(cylinder_bot_world)
    statechart_copy = Statechart.from_json(
        json_data, context=executor.context, **tracker.create_kwargs()
    )
    (limit_copy,) = statechart_copy.top_level_nodes
    executor.compile(statechart=statechart_copy)

    assert isinstance(limit_copy, Parallel)
    assert [type(limit) for limit in limit_copy.nodes] == [
        CartesianPositionVelocityLimit,
        CartesianRotationVelocityLimit,
    ]


# %% goals running their steps in a sequence


def test_differential_drive_goal_runs_its_steps_in_one_sequence(
    cylinder_bot_diff_world: World,
) -> None:
    executor = _create_executor(cylinder_bot_diff_world)
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        goal := DifferentialDriveBaseGoal(
            mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                MobileBase
            )[0],
            goal_pose=Pose.from_xyz_rpy(
                x=1, reference_frame=cylinder_bot_diff_world.root
            ),
        )
    )
    statechart.add_node(EndMotion.when_true(goal))
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
        plan := Sequence(
            nodes=[
                goal := DifferentialDriveBaseGoal(
                    mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                        MobileBase
                    )[
                        0
                    ],
                    goal_pose=Pose.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_diff_world.root
                    ),
                )
            ]
        )
    )
    statechart.add_node(EndMotion.when_true(plan))
    executor.compile(statechart=statechart)

    assert plan.nodes == [goal]
