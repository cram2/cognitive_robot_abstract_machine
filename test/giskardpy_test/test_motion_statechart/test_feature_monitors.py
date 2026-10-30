from math import radians

from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.feature_monitors import (
    HeightMonitor,
    PerpendicularMonitor,
    DistanceMonitor,
    AngleMonitor,
)
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.feature_functions import (
    HeightGoal,
    DistanceGoal,
    AngleGoal,
    AlignPerpendicular,
)
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def _create_executor(world: World) -> StatechartExecutor:
    """
    :return: An executor with motion control acting in `world`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )


def _run(executor: StatechartExecutor, statechart: Statechart) -> None:
    """
    Compiles `statechart` with `executor` and ticks it until it ends.
    """
    executor.compile(statechart=statechart)
    executor.tick_until_end()


def test_height_monitor(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )
    tip_point = Point3(0, 0, 0, reference_frame=tip)
    reference_point = Point3(0, 0, 0, reference_frame=root)
    lower_limit, upper_limit = 0.3, 0.5

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = HeightGoal(
        root_link=root,
        tip_link=tip,
        tip_point=tip_point,
        reference_point=reference_point,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    monitor = HeightMonitor(
        root_link=root,
        tip_link=tip,
        tip_point=tip_point,
        reference_point=reference_point,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_distance_monitor(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )
    tip_point = Point3(0, 0, 0, reference_frame=tip)
    reference_point = Point3(0, 0, 0, reference_frame=root)
    lower_limit, upper_limit = 0.4, 0.6

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = DistanceGoal(
        root_link=root,
        tip_link=tip,
        tip_point=tip_point,
        reference_point=reference_point,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    monitor = DistanceMonitor(
        root_link=root,
        tip_link=tip,
        tip_point=tip_point,
        reference_point=reference_point,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_angle_monitor(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    tip_vector = Vector3.Y(reference_frame=tip)
    reference_vector = Vector3.X(reference_frame=root)
    lower_angle, upper_angle = radians(30), radians(32)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = AngleGoal(
        root_link=root,
        tip_link=tip,
        tip_vector=tip_vector,
        reference_vector=reference_vector,
        lower_angle=lower_angle,
        upper_angle=upper_angle,
    )
    monitor = AngleMonitor(
        root_link=root,
        tip_link=tip,
        tip_vector=tip_vector,
        reference_vector=reference_vector,
        lower_angle=lower_angle,
        upper_angle=upper_angle,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_perpendicular_monitor(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    tip_normal = Vector3.X(reference_frame=tip)
    reference_normal = Vector3.X(reference_frame=root)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = AlignPerpendicular(
        root_link=root,
        tip_link=tip,
        tip_normal=tip_normal,
        reference_normal=reference_normal,
    )
    monitor = PerpendicularMonitor(
        root_link=root,
        tip_link=tip,
        tip_normal=tip_normal,
        reference_normal=reference_normal,
    )
    statechart.add_nodes([drive, monitor])
    statechart.add_node(EndMotion.when_true(monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE
