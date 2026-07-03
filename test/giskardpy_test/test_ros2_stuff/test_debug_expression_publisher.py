import krrood.symbolic_math.symbolic_math as sm

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.debug_expression_publisher import (
    DebugExpressionPublisher,
)
from giskardpy.motion_statechart.graph_node import DebugExpression
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World


def align_planes_statechart(world: World) -> MotionStatechart:
    root = world.root
    tip = world.get_body_by_name("bot")
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(
        AlignPlanes(
            root_link=root,
            tip_link=tip,
            goal_normal=Vector3.Z(reference_frame=root),
            tip_normal=Vector3.Z(reference_frame=tip),
            name="align",
        )
    )
    return motion_statechart


def build_align_planes_task(world: World) -> AlignPlanes:
    root = world.root
    tip = world.get_body_by_name("bot")
    task = AlignPlanes(
        root_link=root,
        tip_link=tip,
        goal_normal=Vector3.Z(reference_frame=root),
        tip_normal=Vector3.Z(reference_frame=tip),
        name="align",
    )
    artifacts = task.build(MotionStatechartContext(world=world))
    task._debug_expressions = artifacts.debug_expressions
    return task


def test_align_planes_sets_visualisation_frame(cylinder_bot_world):
    task = build_align_planes_task(cylinder_bot_world)

    current_normal = next(
        debug_expression
        for debug_expression in task.debug_expressions
        if debug_expression.name == "align/current_normal"
    )

    assert current_normal.expression.visualisation_frame is (
        cylinder_bot_world.get_body_by_name("bot")
    )


def test_attach_visualizes_spatial_debug_expressions_only(
    rclpy_node, cylinder_bot_world
):
    task = build_align_planes_task(cylinder_bot_world)
    task.debug_expressions.append(
        DebugExpression(name="scalar_only", expression=sm.Scalar(1.0))
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(task)

    publisher = DebugExpressionPublisher(world=cylinder_bot_world, node=rclpy_node)
    publisher.attach(motion_statechart)

    namespaces = {request.namespace for request in publisher._publisher._requests}
    assert namespaces == {"align/current_normal", "align/goal_normal"}


def test_vector_with_reference_frame_is_transformed_to_visualisation_frame(
    rclpy_node, cylinder_bot_world
):
    root = cylinder_bot_world.root
    bot = cylinder_bot_world.get_body_by_name("bot")
    vector = Vector3(x=1, y=0, z=0, reference_frame=root, visualisation_frame=bot)
    publisher = DebugExpressionPublisher(world=cylinder_bot_world, node=rclpy_node)

    request = publisher._to_request(DebugExpression(name="vector", expression=vector))

    assert request.spatial_type.reference_frame is bot


def test_attach_publisher_tracks_state_changes(rclpy_node, cylinder_bot_world):
    task = build_align_planes_task(cylinder_bot_world)
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(task)

    publisher = DebugExpressionPublisher(world=cylinder_bot_world, node=rclpy_node)
    publisher.attach(motion_statechart)

    assert publisher._publisher in cylinder_bot_world.state.state_change_callbacks


def test_executor_publishes_debug_expressions_when_enabled(
    rclpy_node, cylinder_bot_world
):
    executor = Ros2Executor(
        MotionStatechartContext(world=cylinder_bot_world),
        ros_node=rclpy_node,
        publish_debug_expressions=True,
    )

    executor.compile(align_planes_statechart(cylinder_bot_world))

    namespaces = {
        request.namespace
        for request in executor._debug_expression_publisher._publisher._requests
    }
    assert "align/current_normal" in namespaces


def test_executor_skips_debug_expressions_by_default(rclpy_node, cylinder_bot_world):
    executor = Ros2Executor(
        MotionStatechartContext(world=cylinder_bot_world),
        ros_node=rclpy_node,
    )

    executor.compile(align_planes_statechart(cylinder_bot_world))

    assert executor._debug_expression_publisher is None


def test_recompile_stops_previous_debug_expression_publisher(
    rclpy_node, cylinder_bot_world
):
    executor = Ros2Executor(
        MotionStatechartContext(world=cylinder_bot_world),
        ros_node=rclpy_node,
        publish_debug_expressions=True,
    )
    executor.compile(align_planes_statechart(cylinder_bot_world))
    previous_publisher = executor._debug_expression_publisher._publisher

    executor.compile(align_planes_statechart(cylinder_bot_world))

    callbacks = cylinder_bot_world.state.state_change_callbacks
    assert all(callback is not previous_publisher for callback in callbacks)
    current_publisher = executor._debug_expression_publisher._publisher
    assert current_publisher is not previous_publisher
    assert any(callback is current_publisher for callback in callbacks)
