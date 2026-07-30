import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.cartesian_monitors import (
    PoseReached,
    PositionReached,
    OrientationReached,
    PointingAt,
    VectorsAligned,
    DistanceToLine,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianOrientation,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.world import World


def _run(msc: MotionStatechart, world: World) -> None:
    kin_sim = Executor(MotionStatechartContext(world=world))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()


def test_position_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(0.6, -0.3, 1.0, reference_frame=root)

    msc = MotionStatechart()
    drive = CartesianPosition(root_link=root, tip_link=tip, goal_point=goal_point)
    monitor = PositionReached(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE


@pytest.mark.parametrize(
    "binding_policy",
    [GoalBindingPolicy.Bind_at_build, GoalBindingPolicy.Bind_on_start],
)
def test_position_reached_binding_policies(
    pr2_world_state_reset: World, binding_policy: GoalBindingPolicy
):
    """
    The robot is driven toward a goal expressed relative to the tip. The monitor captures the same
    goal with the same policy and serves as the end condition, so the motion only ends once the
    monitor detects the goal was reached.
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(0.2, 0, 0, reference_frame=tip)

    msc = MotionStatechart()
    drive = CartesianPosition(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        binding_policy=binding_policy,
    )
    monitor = PositionReached(
        root_link=root,
        tip_link=tip,
        goal_point=goal_point,
        binding_policy=binding_policy,
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_orientation_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_orientation = RotationMatrix.from_axis_angle(
        Vector3.Z(reference_frame=root), 0.3
    )
    goal_orientation.reference_frame = root

    msc = MotionStatechart()
    drive = CartesianOrientation(
        root_link=root, tip_link=tip, goal_orientation=goal_orientation
    )
    monitor = OrientationReached(
        root_link=root,
        tip_link=tip,
        goal_orientation=goal_orientation,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_pose_reached(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.6, y=-0.3, z=1.0, reference_frame=root
    )

    msc = MotionStatechart()
    drive = CartesianPose(root_link=root, tip_link=tip, goal_pose=goal_pose)
    monitor = PoseReached(
        root_link=root,
        tip_link=tip,
        goal_pose=goal_pose,
        binding_policy=GoalBindingPolicy.Bind_at_build,
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_pointing_at(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_point = Point3(2, 0, 0, reference_frame=root)
    pointing_axis = Vector3.X(reference_frame=tip)

    msc = MotionStatechart()
    drive = Pointing(
        root_link=root, tip_link=tip, goal_point=goal_point, pointing_axis=pointing_axis
    )
    monitor = PointingAt(
        root_link=root, tip_link=tip, goal_point=goal_point, pointing_axis=pointing_axis
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_vectors_aligned(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    goal_normal = Vector3.X(reference_frame=root)
    tip_normal = Vector3.X(reference_frame=tip)

    msc = MotionStatechart()
    drive = AlignPlanes(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    monitor = VectorsAligned(
        root_link=root, tip_link=tip, goal_normal=goal_normal, tip_normal=tip_normal
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE


def test_distance_to_line(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    center_point = Point3(0.6, -0.3, 1.0, reference_frame=root)
    line_axis = Vector3.Y(reference_frame=root)

    msc = MotionStatechart()
    drive = CartesianPosition(root_link=root, tip_link=tip, goal_point=center_point)
    monitor = DistanceToLine(
        root_link=root,
        tip_link=tip,
        center_point=center_point,
        line_axis=line_axis,
        line_length=0.4,
    )
    msc.add_nodes([drive, monitor])
    msc.add_node(EndMotion.when_true(monitor))

    _run(msc, pr2_world_state_reset)

    assert monitor.observation_state == ObservationStateValues.TRUE
