"""
Tests visualizing the debug expressions of motion statechart nodes.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from visualization_msgs.msg import Marker, MarkerArray

import krrood.symbolic_math.symbolic_math as sm
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.exceptions import MissingExecutorExtensionError
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import EndedByOwner
from cramph.statechart import Statechart
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.model.world_config import EmptyWorld
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.debug_expression_publisher import (
    DebugExpressionPublishing,
    MarkerSubmission,
)
from giskardpy.motion_statechart.exceptions import MarkerPublishingThreadEndedError
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    MotionNodeArtifacts,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from krrood_spatial_math.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from ..motion_control_context import add_to_statechart_with_motion_control

# %% statecharts whose nodes register debug expressions


@dataclass(eq=False, repr=False, kw_only=True)
class NodeRegisteringDebugExpressions(EndedByOwner, MotionStatechartNode):
    """
    Registers the given debug expressions and nothing else, and runs until it is ended.
    """

    registered_debug_expressions: list[DebugExpression] = field(default_factory=list)
    """
    The debug expressions the node registers when it is built.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        return MotionNodeArtifacts(
            debug_expressions=list(self.registered_debug_expressions)
        )


@dataclass
class FakeMarkerPublisher:
    """
    Captures published marker arrays so tests can assert without a live RViz.
    """

    published: list[MarkerArray] = field(default_factory=list)
    """
    The published marker arrays, oldest first.
    """

    def publish(self, marker_array: MarkerArray) -> None:
        """
        Record `marker_array` instead of sending it.
        """
        self.published.append(marker_array)


@pytest.fixture()
def publishing(rclpy_node) -> Iterator[DebugExpressionPublishing]:
    """
    Publishes at every tick of :func:`create_executor`'s default tick duration, and has
    published everything before the ROS node is destroyed.
    """
    publishing = DebugExpressionPublishing(rclpy_node, publish_frequency=10)
    yield publishing
    publishing._marker_publisher.wait_until_published()


def create_executor(
    world: World, publishing: DebugExpressionPublishing, tick_duration: float = 0.1
) -> StatechartExecutor:
    """
    An executor that publishes debug expressions and nothing else.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world, tick_duration=tick_duration),
        extensions=[publishing],
    )


def point_debug_expression(name: str, world: World) -> DebugExpression:
    """
    A point whose x coordinate is the position of the first moving degree of freedom.
    """
    return DebugExpression(
        name=name,
        expression=Point3(
            x=world.active_degrees_of_freedom[0].variables.position,
            reference_frame=world.root,
        ),
    )


@dataclass
class NodesStartingInTurn:
    """
    A statechart in which :attr:`first` runs from the start and :attr:`second` only
    after a few ticks.
    """

    statechart: Statechart
    """
    The statechart holding the nodes.
    """

    first: NodeRegisteringDebugExpressions
    """
    Runs from the first tick.
    """

    second: NodeRegisteringDebugExpressions
    """
    Starts once a tick counter turned true.
    """

    @classmethod
    def build(cls, context: StatechartContext) -> NodesStartingInTurn:
        """
        :param context: The context the statechart is built in.
        :return: The statechart and its nodes.
        """
        statechart = Statechart(context=context)
        first = NodeRegisteringDebugExpressions(
            name="first",
            registered_debug_expressions=[
                point_debug_expression("first/point", context.world)
            ],
        )
        counter = CountTicks(name="counter", ticks=3)
        second = NodeRegisteringDebugExpressions(
            name="second",
            registered_debug_expressions=[
                point_debug_expression("second/point", context.world)
            ],
        )
        statechart.add_nodes([first, counter, second])
        second.start_condition = counter.observes_true
        return cls(statechart=statechart, first=first, second=second)


def align_planes_task(world: World) -> AlignPlanes:
    """
    A task aligning the z axis of the bot with the z axis of the root, named ``align``.
    """
    root = world.root
    tip = world.get_body_by_name("bot")
    return AlignPlanes(
        root_link=root,
        tip_link=tip,
        goal_normal=Vector3.Z(reference_frame=root),
        tip_normal=Vector3.Z(reference_frame=tip),
        name="align",
    )


def tick_until_running(
    executor: StatechartExecutor, node: MotionStatechartNode, tick_limit: int = 10
) -> None:
    """
    Tick until `node` runs.
    """
    for _ in range(tick_limit):
        executor.tick()
        if node.life_cycle_state == LifeCycleValues.RUNNING:
            return
    pytest.fail(f"{node.name} did not start within {tick_limit} ticks")


def published_namespaces(publishing: DebugExpressionPublishing) -> set[str]:
    """
    The namespaces of the markers that are shown once everything submitted is published.
    """
    publishing._marker_publisher.wait_until_published()
    return {
        identity.namespace
        for identity in publishing._marker_publisher.marker_publisher._published_marker_identities
    }


# %% importing without ROS


def test_ros_node_access_importable_without_rclpy(monkeypatch) -> None:
    """
    giskardpy.motion_statechart.ros_context must stay importable when rclpy is not
    installed.
    """
    modules_to_evict = [
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "giskardpy.motion_statechart.ros_context"
        or name == "giskardpy.motion_statechart.debug_expression_publisher"
    ]
    for name in modules_to_evict:
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "rclpy" or name.startswith("rclpy."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    importlib.import_module("giskardpy.motion_statechart.ros_context")


# %% collecting the debug expressions of each node


def test_align_planes_sets_visualisation_frame(cylinder_bot_world):
    """
    The current normal of the tip is drawn from the tip.
    """
    task = align_planes_task(cylinder_bot_world)
    add_to_statechart_with_motion_control(task, cylinder_bot_world)
    task._artifacts = task.build()

    current_normal = next(
        debug_expression
        for debug_expression in task.debug_expressions
        if debug_expression.name == "align/current_normal"
    )

    assert current_normal.expression.visualisation_frame is (
        cylinder_bot_world.get_body_by_name("bot")
    )


def test_compile_collects_the_spatial_debug_expressions_of_each_node(
    publishing, cylinder_bot_world
):
    """
    Debug expressions that are no spatial type are not visualized.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    statechart = Statechart(context=executor.context)
    node = NodeRegisteringDebugExpressions(
        name="node",
        registered_debug_expressions=[
            point_debug_expression("node/point", cylinder_bot_world),
            DebugExpression(name="node/scalar", expression=sm.Scalar(1.0)),
        ],
    )
    statechart.add_node(node)

    executor.compile(statechart)

    assert {
        collected_node: [visualization.namespace for visualization in visualizations]
        for collected_node, visualizations in publishing._visualizations_per_node.items()
    } == {node: ["node/point"]}


def test_vector_is_expressed_in_its_visualisation_frame(cylinder_bot_world):
    """
    A vector whose visualisation frame differs from its reference frame is drawn from
    its visualisation frame.
    """
    root = cylinder_bot_world.root
    bot = cylinder_bot_world.get_body_by_name("bot")
    vector = Vector3(x=1, y=0, z=0, reference_frame=root, visualisation_frame=bot)

    visualization = DebugExpressionPublishing._create_visualization(
        DebugExpression(name="vector", expression=vector), cylinder_bot_world
    )

    assert visualization.spatial_type.reference_frame is bot


# %% publishing the running nodes


def test_only_the_debug_expressions_of_running_nodes_are_published(
    publishing, cylinder_bot_world
):
    """
    Nodes that have not started yet show no markers.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    nodes = NodesStartingInTurn.build(executor.context)
    executor.compile(nodes.statechart)

    executor.tick()
    assert published_namespaces(publishing) == {"first/point"}

    tick_until_running(executor, nodes.second)
    assert published_namespaces(publishing) == {"first/point", "second/point"}


def test_the_shown_debug_expressions_are_compiled_again_only_when_they_change(
    publishing, cylinder_bot_world
):
    """
    The function evaluating the shown debug expressions is reused while the same nodes
    run.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    nodes = NodesStartingInTurn.build(executor.context)
    executor.compile(nodes.statechart)
    executor.tick()
    compiled_with_first_running = publishing._compiled_visualizations

    executor.tick()
    assert publishing._compiled_visualizations is compiled_with_first_running

    tick_until_running(executor, nodes.second)
    assert publishing._compiled_visualizations is not compiled_with_first_running


def test_markers_are_published_at_the_publish_frequency(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    The first tick publishes, then every tick that lies one publish period later.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    publishing.publish_frequency = 1 / (3 * executor.context.require_tick_duration())
    executor.compile(NodesStartingInTurn.build(executor.context).statechart)
    published_ticks = []
    monkeypatch.setattr(
        publishing._marker_publisher,
        "submit",
        lambda submission: published_ticks.append(executor.tick_count),
    )

    for _ in range(9):
        executor.tick()

    assert published_ticks == [1, 4, 7]


def test_markers_show_the_current_values_without_recompiling(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    The published markers follow the world state through the compiled function.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    executor.compile(NodesStartingInTurn.build(executor.context).statechart)
    executor.tick()
    fake = FakeMarkerPublisher()
    publishing._marker_publisher.marker_publisher.publisher = fake
    monkeypatch.setattr(
        SymbolicMathType, "evaluate", lambda self: pytest.fail("evaluate recompiles")
    )
    dof = cylinder_bot_world.active_degrees_of_freedom[0]

    cylinder_bot_world.state[dof.id].position = 1.5
    executor.tick()
    publishing._marker_publisher.wait_until_published()

    [sphere] = [
        marker
        for marker in fake.published[-1].markers
        if marker.ns == "first/point" and marker.type == Marker.SPHERE
    ]
    assert sphere.pose.position.x == 1.5


def test_the_end_of_a_run_deletes_the_markers(publishing, cylinder_bot_world):
    """
    No marker stays shown once a run stopped.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    executor.compile(NodesStartingInTurn.build(executor.context).statechart)
    executor.tick()

    executor.finish_run()

    assert published_namespaces(publishing) == set()


def test_compiling_again_forgets_the_nodes_of_the_previous_statechart(
    publishing, cylinder_bot_world
):
    """
    Only the nodes of the statechart compiled last are visualized.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    previous = NodesStartingInTurn.build(executor.context)
    executor.compile(previous.statechart)
    executor.tick()

    current = NodesStartingInTurn.build(executor.context)
    executor.compile(current.statechart)

    assert set(publishing._visualizations_per_node) == {current.first, current.second}


def test_a_submission_replaced_before_it_is_published_is_dropped(publishing):
    """
    Only the newest submission waiting for the publishing thread is published.
    """
    marker_publisher = publishing._marker_publisher
    fake = FakeMarkerPublisher()
    marker_publisher.marker_publisher.publisher = fake

    with marker_publisher._condition:
        marker_publisher.submit(
            MarkerSubmission(evaluated_visualizations=[], root_frame_name="dropped")
        )
        marker_publisher.submit(
            MarkerSubmission(evaluated_visualizations=[], root_frame_name="published")
        )
    marker_publisher.wait_until_published()

    assert len(fake.published) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_submitting_after_publishing_raised_raises(publishing, monkeypatch):
    """
    A publishing thread that ended does not leave the submitting thread waiting.
    """
    marker_publisher = publishing._marker_publisher

    def raise_error(evaluated_visualizations, root_frame_name) -> None:
        """
        Fail like a publisher whose ROS node was destroyed.
        """
        raise RuntimeError("publishing failed")

    monkeypatch.setattr(marker_publisher.marker_publisher, "publish", raise_error)
    marker_publisher.submit(
        MarkerSubmission(evaluated_visualizations=[], root_frame_name="map")
    )
    marker_publisher._thread.join()

    with pytest.raises(MarkerPublishingThreadEndedError):
        marker_publisher.submit(
            MarkerSubmission(evaluated_visualizations=[], root_frame_name="map")
        )


# %% enabling the publishing


def test_executor_with_motion_control_publishes_the_running_tasks(
    rclpy_node, publishing, cylinder_bot_world
):
    """
    The debug expressions of a task are published while motion control runs it.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[RosNodeAccess(rclpy_node), MotionControl(), publishing],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(align_planes_task(cylinder_bot_world))
    executor.compile(statechart)

    executor.tick()

    assert published_namespaces(publishing) == {
        "align/current_normal",
        "align/goal_normal",
    }


def test_executor_skips_debug_expressions_by_default(rclpy_node, cylinder_bot_world):
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[RosNodeAccess(rclpy_node), MotionControl()],
    )

    with pytest.raises(MissingExecutorExtensionError):
        executor.require_extension(DebugExpressionPublishing)


@pytest.mark.parametrize(
    ("debug_mode", "publish_debug_expressions", "expected_extension_types"),
    [
        (True, True, [DebugExpressionPublishing]),
        (True, False, []),
        (False, True, []),
    ],
)
def test_giskard_publishes_debug_expressions_only_in_debug_mode(
    rclpy_node, debug_mode, publish_debug_expressions, expected_extension_types
):
    """
    Debug expressions are published only when both switches are on.
    """
    server_config = GiskardServerConfig(
        publish_debug_expressions=publish_debug_expressions
    )
    server_config.debug_mode = debug_mode
    giskard = Giskard(
        world_config=EmptyWorld(),
        server_config=server_config,
        robot_interface_config=StandAloneRobotInterfaceConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    extension_types = [
        type(extension) for extension in giskard.create_debug_extensions(rclpy_node)
    ]

    assert extension_types == expected_extension_types
