import json

import numpy as np

from cramph.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from cramph.composites import Sequence
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    StructureCopyMotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from cramph.statechart import Statechart
from giskardpy.motion_statechart.goals.base_pose_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from cramph.nodes_for_testing import ConstTrueNode
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.adapters.json_serializer import to_json, from_json
from krrood.symbolic_math.symbolic_math import (
    logic_and,
)
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

# %% motion nodes in JSON


def _create_executor(world: World) -> StatechartExecutor:
    """
    :return: An executor with simulated motion control acting in `world`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            )
        ],
    )


def test_to_json_joint_position_list(mini_world):
    connection = mini_world.connections[0]
    node = JointPositionList(
        goal_state=JointState.from_mapping({connection: 0.5}),
        threshold=0.5,
    )
    json_data = to_json(node)
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    tracker = WorldEntityWithIDKwargsTracker.from_world(mini_world)
    node_copy = from_json(new_json_data, **tracker.create_kwargs())
    assert node_copy.name == node.name
    assert node_copy.threshold == node.threshold
    assert node_copy.goal_state == node.goal_state


def test_executing_json_parsed_statechart(tmp_path):
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        tip2 = Body(name=PrefixedName("tip2"))
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(maximum_speed=1)
        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "a"),
            limits=limits,
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), raw_dof=dof
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"),
            limits=limits,
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), raw_dof=dof
        )
        world.add_connection(root_C_tip2)

    statechart = Statechart(context=_create_executor(world).context)

    task1 = JointPositionList(goal_state=JointState.from_mapping({root_C_tip: 0.5}))
    always_true = ConstTrueNode()
    statechart.add_node(always_true)
    statechart.add_node(task1)
    end = EndMotion()
    statechart.add_node(end)

    task1.start_condition = always_true.observes_true
    end.start_condition = logic_and(task1.observes_true, always_true.observes_true)

    json_data = statechart.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    executor = _create_executor(world)
    statechart_copy = Statechart.from_json(
        new_json_data, context=executor.context, **tracker.create_kwargs()
    )

    executor.compile(statechart=statechart_copy)

    task1_copy = statechart_copy.get_node_by_index(task1.index)
    end_copy = statechart_copy.get_node_by_index(end.index)
    assert task1_copy.observation_state == ObservationStateValues.UNKNOWN
    assert end_copy.observation_state == ObservationStateValues.UNKNOWN
    assert task1_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    statechart.draw(str(tmp_path / "muh.pdf"))
    executor.tick_until_end()
    statechart.draw(str(tmp_path / "muh.pdf"))
    assert task1_copy.observation_state == ObservationStateValues.TRUE
    assert end_copy.observation_state == ObservationStateValues.TRUE
    assert task1_copy.life_cycle_state == LifeCycleValues.RUNNING
    assert end_copy.life_cycle_state == LifeCycleValues.RUNNING


def test_cart_goal_simple(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )

    statechart = Statechart(context=_create_executor(pr2_world_state_reset).context)
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    statechart.add_node(cart_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = cart_goal.observes_true

    json_data = statechart.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(pr2_world_state_reset)
    kwargs = tracker.create_kwargs()
    executor = _create_executor(pr2_world_state_reset)
    statechart_copy = Statechart.from_json(
        new_json_data, context=executor.context, **kwargs
    )

    executor.compile(statechart=statechart_copy)
    executor.tick_until_end()

    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal, atol=cart_goal.translation_threshold)


def test_structure_copy_of_a_plain_statechart_keeps_the_motion_node_kinds(
    mini_world,
) -> None:
    connection = mini_world.connections[0]
    statechart = Statechart(context=_create_executor(mini_world).context)
    statechart.add_nodes(
        [
            task := JointPositionList(
                goal_state=JointState.from_mapping({connection: 0.5})
            ),
            monitor := LocalMinimumReached(),
            end := EndMotion.when_true(task),
        ]
    )

    statechart_copy = statechart.create_structure_copy()

    assert type(statechart_copy.get_node_by_index(task.index)) is Task
    assert type(statechart_copy.get_node_by_index(monitor.index)) is (
        StructureCopyMotionStatechartNode
    )
    assert type(statechart_copy.get_node_by_index(end.index)) is EndMotion


def test_nested_sequence_goal_json_round_trip_compilation() -> None:
    """
    A statechart with nested goals watched by a progress monitor can be deserialized and
    compiled.
    """
    statechart = Statechart(context=_create_executor(World()).context)
    leaf_node = ConstTrueNode(name="ConstTrue")
    child_sequence = Sequence(nodes=[leaf_node], name="SequentialNode")
    root = Sequence(nodes=[child_sequence], name="ActionNode")
    statechart.add_node(root)
    statechart.add_node(still_progressing := StillProgressing(monitored_node=root))
    statechart.add_node(still_progressing.cancel_motion())
    statechart.add_node(EndMotion.when_true(root))

    json_data = statechart.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    executor = _create_executor(World())
    statechart_copy = Statechart.from_json(new_json_data, context=executor.context)
    executor.compile(statechart=statechart_copy)


# %% statecharts sent after their goals expanded


def test_an_expanded_differential_drive_goal_reaches_its_pose_after_a_json_round_trip(
    cylinder_bot_diff_world: World,
) -> None:
    """
    A goal expands when it joins, so what is sent is its expanded steps, whose
    orientations are expressions over the base's forward kinematics.
    """
    goal_pose = Pose.from_xyz_rpy(x=0.5, reference_frame=cylinder_bot_diff_world.root)
    statechart = Statechart(context=_create_executor(cylinder_bot_diff_world).context)
    statechart.add_node(
        goal := DifferentialDriveBaseGoal(
            mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                MobileBase
            )[0],
            goal_pose=goal_pose,
            threshold=0.05,
        )
    )
    statechart.add_node(EndMotion.when_true(goal))
    new_json_data = json.loads(json.dumps(statechart.to_json()))
    statechart.context.cleanup()

    executor = _create_executor(cylinder_bot_diff_world)
    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_diff_world)
    statechart_copy = Statechart.from_json(
        new_json_data, context=executor.context, **tracker.create_kwargs()
    )
    goal_copy = statechart_copy.get_node_by_index(goal.index)
    executor.compile(statechart=statechart_copy)
    executor.tick_until_end(1000)

    assert [step.name for step in goal_copy.sequence.nodes] == [
        step.name for step in goal.sequence.nodes
    ]
    assert goal_copy.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_an_expanded_differential_drive_goal_binds_its_goal_frame_after_a_json_round_trip(
    cylinder_bot_diff_world: World,
) -> None:
    """
    The second leg is given relative to the base, so it ends where it should only if the
    receiver binds the goal frame when the leg starts, after the first leg moved the
    base.
    """
    bot = cylinder_bot_diff_world.get_body_by_name("bot")
    statechart = Statechart(context=_create_executor(cylinder_bot_diff_world).context)
    statechart.add_node(
        first_leg := DifferentialDriveBaseGoal(
            mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                MobileBase
            )[0],
            goal_pose=Pose.from_xyz_rpy(
                x=1, reference_frame=cylinder_bot_diff_world.root
            ),
            threshold=0.05,
        )
    )
    statechart.add_node(
        second_leg := DifferentialDriveBaseGoal(
            mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                MobileBase
            )[0],
            goal_pose=Pose.from_xyz_rpy(x=0.5, reference_frame=bot),
            threshold=0.05,
        )
    )
    second_leg.start_condition = first_leg.observes_true
    statechart.add_node(EndMotion.when_true(second_leg))
    new_json_data = json.loads(json.dumps(statechart.to_json()))
    statechart.context.cleanup()

    executor = _create_executor(cylinder_bot_diff_world)
    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_diff_world)
    statechart_copy = Statechart.from_json(
        new_json_data, context=executor.context, **tracker.create_kwargs()
    )
    second_leg_copy = statechart_copy.get_node_by_index(second_leg.index)
    executor.compile(statechart=statechart_copy)
    executor.tick_until_end(2000)

    assert second_leg_copy.life_cycle_state == LifeCycleValues.SUCCEEDED
    np.testing.assert_allclose(
        cylinder_bot_diff_world.compute_forward_kinematics_np(
            cylinder_bot_diff_world.root, bot
        )[:3, 3],
        [1.5, 0, 0],
        atol=2 * second_leg.threshold,
    )
