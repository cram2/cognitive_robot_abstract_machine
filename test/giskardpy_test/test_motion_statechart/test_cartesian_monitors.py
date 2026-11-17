import pytest

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.cartesian_monitors import (
    PoseReached,
    PositionReached,
    OrientationReached,
    PointingAt,
    VectorsAligned,
    DistanceToLine,
)
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianOrientation,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain


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


def test_position_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(0.6, -0.3, 1.0, reference_frame=root)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPosition(
        kinematic_chain=KinematicChain(root=root, tip=tip), goal_point=goal_point
    )
    monitor = PositionReached(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_point=goal_point,
        goal_binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


@pytest.mark.parametrize(
    "goal_binding_policy",
    [GoalBindingPolicy.Bind_at_build, GoalBindingPolicy.Bind_on_start],
)
def test_position_reached_binding_policies(
    pr2_world_state_reset: World, goal_binding_policy: GoalBindingPolicy
):
    """
    The robot is driven toward a goal expressed relative to the tip.

    The monitor captures the same goal with the same policy and serves as the success
    condition, so the motion only ends once the monitor detects the goal was reached.
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(0.2, 0, 0, reference_frame=tip)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPosition(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_point=goal_point,
        goal_binding_policy=goal_binding_policy,
    )
    monitor = PositionReached(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_point=goal_point,
        goal_binding_policy=goal_binding_policy,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_orientation_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_orientation = RotationMatrix.from_axis_angle(
        AxisAngle(Vector3.Z(reference_frame=root), 0.3)
    )
    goal_orientation.reference_frame = root

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianOrientation(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_orientation=goal_orientation,
    )
    monitor = OrientationReached(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_orientation=goal_orientation,
        goal_binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_pose_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.6, y=-0.3, z=1.0, reference_frame=root
    )

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPose(
        kinematic_chain=KinematicChain(root=root, tip=tip), goal_pose=goal_pose
    )
    monitor = PoseReached(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_pose=goal_pose,
        goal_binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_pointing_at(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(2, 0, 0, reference_frame=root)
    pointing_axis = Vector3.X(reference_frame=tip)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = Pointing(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_point=goal_point,
        pointing_axis=pointing_axis,
    )
    monitor = PointingAt(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        goal_point=goal_point,
        pointing_axis=pointing_axis,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_vectors_aligned(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    reference_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.X(reference_frame=tip)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = AlignPlanes(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        reference_normal=reference_normal,
        tip_normal=tip_normal,
    )
    monitor = VectorsAligned(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        reference_normal=reference_normal,
        tip_normal=tip_normal,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_distance_to_line(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    center_point = Point3(0.6, -0.3, 1.0, reference_frame=root)
    line_axis = Vector3.Y(reference_frame=root)

    executor = _create_executor(pr2_world_state_reset)
    statechart = Statechart(context=executor.context)
    drive = CartesianPosition(
        kinematic_chain=KinematicChain(root=root, tip=tip), goal_point=center_point
    )
    monitor = DistanceToLine(
        kinematic_chain=KinematicChain(root=root, tip=tip),
        center_point=center_point,
        line_axis=line_axis,
        line_length=0.4,
    )
    statechart.add_nodes(nodes=[drive, monitor])
    statechart.add_node(node=EndMotion.when_true(node=monitor))

    _run(executor, statechart)

    assert monitor.observation_state == ObservationStateValues.TRUE
