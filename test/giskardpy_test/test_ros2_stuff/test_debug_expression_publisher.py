"""
Tests visualizing the debug expressions of motion statechart nodes.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
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
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart import debug_expression_publisher
from giskardpy.motion_statechart.debug_expression_publisher import (
    BackgroundMarkerPublisher,
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
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianOrientation
from giskardpy.motion_statechart.tasks.feature_functions import AlignPerpendicular
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from krrood_spatial_math.spatial_types import Point3, RotationMatrix, Vector3
from semantic_digital_twin.adapters.ros.visualization import (
    spatial_type_marker_renderer,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    EvaluatedSpatialTypeVisualization,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
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
    publishing = DebugExpressionPublishing(ros_node=rclpy_node, publish_frequency=10)
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
        statechart.add_nodes(nodes=[first, counter, second])
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


def cartesian_orientation_task(world: World) -> CartesianOrientation:
    """
    A task turning the bot about the z axis of the root.
    """
    return CartesianOrientation(
        root_link=world.root,
        tip_link=world.get_body_by_name("bot"),
        goal_orientation=RotationMatrix.from_rpy(yaw=1, reference_frame=world.root),
        name="orientation",
    )


def align_perpendicular_task(world: World) -> AlignPerpendicular:
    """
    A task turning the x axis of the bot perpendicular to the x axis of the root.
    """
    return AlignPerpendicular(
        root_link=world.root,
        tip_link=world.get_body_by_name("bot"),
        tip_normal=Vector3.X(reference_frame=world.get_body_by_name("bot")),
        reference_normal=Vector3.X(reference_frame=world.root),
        name="perpendicular",
    )


def marker_origins_in_root(markers: list[Marker], world: World) -> list[list[float]]:
    """
    Where each of `markers` starts, expressed in the root frame of `world`.
    """
    origins = []
    for marker in markers:
        root_T_frame = world.compute_forward_kinematics_np(
            world.root,
            world.get_kinematic_structure_entity_by_name(marker.header.frame_id),
        )
        position = marker.pose.position
        origins.append(
            (root_T_frame @ [position.x, position.y, position.z, 1])[:3].tolist()
        )
    return origins


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


def collected_namespaces(
    publishing: DebugExpressionPublishing, *nodes: MotionStatechartNode
) -> set[str]:
    """
    The marker namespaces of the debug expressions `publishing` collected of `nodes`.
    """
    return {
        visualization.namespace
        for node in nodes
        for visualization in publishing._visualizations_per_node[node]
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
    statechart.add_node(node=node)

    executor.compile(statechart=statechart)

    assert {
        collected_node: [visualization.label for visualization in visualizations]
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
        debug_expression=DebugExpression(name="vector", expression=vector),
        node=NodeRegisteringDebugExpressions(name="node"),
        world=cylinder_bot_world,
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
    executor.compile(statechart=nodes.statechart)

    executor.tick()
    assert published_namespaces(publishing) == collected_namespaces(
        publishing, nodes.first
    )

    tick_until_running(executor, nodes.second)
    assert published_namespaces(publishing) == collected_namespaces(
        publishing, nodes.first, nodes.second
    )


def test_running_nodes_whose_debug_expressions_share_a_name_publish_apart(
    publishing, cylinder_bot_world
):
    """
    Two nodes with the same name, such as two arms reaching for a pose at once, do not
    overwrite each other's markers.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        nodes=[
            NodeRegisteringDebugExpressions(
                name="reach",
                registered_debug_expressions=[
                    point_debug_expression("reach/point", cylinder_bot_world)
                ],
            )
            for _ in range(2)
        ]
    )
    executor.compile(statechart=statechart)

    executor.tick()

    assert len(published_namespaces(publishing)) == 2


def test_ticking_compiles_no_debug_expressions(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    The debug expressions are compiled with the statechart, so no tick compiles, not
    even one at which other nodes start running.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    nodes = NodesStartingInTurn.build(executor.context)
    executor.compile(statechart=nodes.statechart)

    def fail_compiling(**kwargs) -> None:
        """
        Fail the test instead of compiling during a tick.
        """
        pytest.fail("a tick compiled debug expressions")

    monkeypatch.setattr(
        debug_expression_publisher, "CompiledSpatialTypeVisualizations", fail_compiling
    )

    tick_until_running(executor, nodes.second)

    assert published_namespaces(publishing) == collected_namespaces(
        publishing, nodes.first, nodes.second
    )


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_publishing_builds_no_markers_after_compiling(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    The markers are built with the statechart, so publishing only writes new values into
    them, also for nodes that start running later.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    nodes = NodesStartingInTurn.build(executor.context)
    executor.compile(statechart=nodes.statechart)

    def fail_building_markers(**kwargs) -> Marker:
        """
        Stop the publishing thread instead of building a marker after compiling.
        """
        raise AssertionError("publishing built markers")

    monkeypatch.setattr(spatial_type_marker_renderer, "Marker", fail_building_markers)

    tick_until_running(executor, nodes.second)

    assert published_namespaces(publishing) == collected_namespaces(
        publishing, nodes.first, nodes.second
    )


def test_markers_are_published_at_the_publish_frequency(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    The first tick publishes, then every tick that lies one publish period later.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    publishing.publish_frequency = 1 / (3 * executor.context.require_tick_duration())
    executor.compile(statechart=NodesStartingInTurn.build(executor.context).statechart)
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
    nodes = NodesStartingInTurn.build(executor.context)
    executor.compile(statechart=nodes.statechart)
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
        if marker.ns in collected_namespaces(publishing, nodes.first)
        and marker.type == Marker.SPHERE
    ]
    assert sphere.pose.position.x == 1.5


def test_switched_off_publishing_neither_collects_nor_publishes(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    A statechart compiled while publishing is off costs no evaluation and shows no
    markers.
    """
    publishing.publishes_markers = False
    executor = create_executor(cylinder_bot_world, publishing)
    nodes = NodesStartingInTurn.build(executor.context)
    monkeypatch.setattr(
        publishing._marker_publisher,
        "submit",
        lambda submission: pytest.fail("switched off publishing submitted markers"),
    )

    executor.compile(statechart=nodes.statechart)
    tick_until_running(executor, nodes.second)
    executor.finish_run()

    assert publishing._visualizations_per_node == {}


def test_the_end_of_a_run_deletes_the_markers(publishing, cylinder_bot_world):
    """
    No marker stays shown once a run stopped.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    executor.compile(statechart=NodesStartingInTurn.build(executor.context).statechart)
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
    executor.compile(statechart=previous.statechart)
    executor.tick()

    current = NodesStartingInTurn.build(executor.context)
    executor.compile(statechart=current.statechart)

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
            submission=MarkerSubmission(evaluated_visualizations=[])
        )
        marker_publisher.submit(
            submission=MarkerSubmission(evaluated_visualizations=[])
        )
    marker_publisher.wait_until_published()

    assert len(fake.published) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_submitting_after_publishing_raised_raises(publishing, monkeypatch):
    """
    A publishing thread that ended does not leave the submitting thread waiting.
    """
    marker_publisher = publishing._marker_publisher

    def raise_error(
        *, evaluated_visualizations: list[EvaluatedSpatialTypeVisualization]
    ) -> None:
        """
        Fail like a publisher whose ROS node was destroyed.
        """
        raise RuntimeError("publishing failed")

    monkeypatch.setattr(marker_publisher.marker_publisher, "publish", raise_error)
    marker_publisher.submit(submission=MarkerSubmission(evaluated_visualizations=[]))
    marker_publisher._thread.join()

    with pytest.raises(MarkerPublishingThreadEndedError):
        marker_publisher.submit(
            submission=MarkerSubmission(evaluated_visualizations=[])
        )


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_waiting_for_publishes_returns_after_publishing_raised_with_a_submission_pending(
    rclpy_node, monkeypatch
):
    """
    A submission that arrives while the publish that ends the thread is in progress does
    not leave the waiting thread blocked.
    """
    marker_publisher = BackgroundMarkerPublisher(node=rclpy_node)
    publish_started = threading.Event()
    publish_may_raise = threading.Event()

    def raise_error_once_released(
        *, evaluated_visualizations: list[EvaluatedSpatialTypeVisualization]
    ) -> None:
        """
        Fail like a publisher whose ROS node was destroyed, once the test let it.
        """
        publish_started.set()
        publish_may_raise.wait()
        raise RuntimeError("publishing failed")

    monkeypatch.setattr(
        marker_publisher.marker_publisher, "publish", raise_error_once_released
    )
    marker_publisher.submit(submission=MarkerSubmission(evaluated_visualizations=[]))
    publish_started.wait()
    marker_publisher.submit(submission=MarkerSubmission(evaluated_visualizations=[]))
    publish_may_raise.set()
    waiting_thread = threading.Thread(
        target=marker_publisher.wait_until_published, daemon=True
    )

    waiting_thread.start()
    waiting_thread.join(timeout=5)

    assert not waiting_thread.is_alive()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_finishing_a_run_after_publishing_raised_releases_the_visualizations(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    The end of a run forgets the visualizations even when the publishing thread ended.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    executor.compile(statechart=NodesStartingInTurn.build(executor.context).statechart)
    marker_publisher = publishing._marker_publisher

    def raise_error(
        *, evaluated_visualizations: list[EvaluatedSpatialTypeVisualization]
    ) -> None:
        """
        Fail like a publisher whose ROS node was destroyed.
        """
        raise RuntimeError("publishing failed")

    monkeypatch.setattr(marker_publisher.marker_publisher, "publish", raise_error)
    executor.tick()
    marker_publisher._thread.join()

    executor.finish_run()

    assert publishing._visualizations_per_node == {}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_submitting_after_publishing_raised_names_the_cause(publishing, monkeypatch):
    """
    The error a submission raises once the publishing thread ended carries the exception
    that ended it.
    """
    marker_publisher = publishing._marker_publisher
    cause = RuntimeError("publishing failed")

    def raise_cause(
        *, evaluated_visualizations: list[EvaluatedSpatialTypeVisualization]
    ) -> None:
        """
        Fail like a publisher whose ROS node was destroyed.
        """
        raise cause

    monkeypatch.setattr(marker_publisher.marker_publisher, "publish", raise_cause)
    marker_publisher.submit(submission=MarkerSubmission(evaluated_visualizations=[]))
    marker_publisher._thread.join()

    with pytest.raises(MarkerPublishingThreadEndedError) as raised:
        marker_publisher.submit(
            submission=MarkerSubmission(evaluated_visualizations=[])
        )

    assert raised.value.__cause__ is cause


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_statechart_compiled_after_publishing_raised_publishes_again(
    publishing, cylinder_bot_world, monkeypatch
):
    """
    A failed publish ends the markers of the run it happened in, not those of the
    statecharts compiled after it.
    """
    executor = create_executor(cylinder_bot_world, publishing)
    executor.compile(statechart=NodesStartingInTurn.build(executor.context).statechart)
    failed_marker_publisher = publishing._marker_publisher

    def raise_error(
        *, evaluated_visualizations: list[EvaluatedSpatialTypeVisualization]
    ) -> None:
        """
        Fail like a publisher whose ROS node was destroyed.
        """
        raise RuntimeError("publishing failed")

    monkeypatch.setattr(
        failed_marker_publisher.marker_publisher, "publish", raise_error
    )
    executor.tick()
    failed_marker_publisher._thread.join()
    executor.finish_run()
    nodes = NodesStartingInTurn.build(executor.context)

    executor.compile(statechart=nodes.statechart)
    executor.tick()

    assert published_namespaces(publishing) == collected_namespaces(
        publishing, nodes.first
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
        extensions=[RosNodeAccess(ros_node=rclpy_node), MotionControl(), publishing],
    )
    statechart = Statechart(context=executor.context)
    task = align_planes_task(cylinder_bot_world)
    statechart.add_node(node=task)
    executor.compile(statechart=statechart)

    executor.tick()

    assert [
        visualization.label
        for visualization in publishing._visualizations_per_node[task]
    ] == ["align/current_normal", "align/goal_normal"]
    assert published_namespaces(publishing) == collected_namespaces(publishing, task)


@pytest.mark.parametrize(
    "create_task",
    [cartesian_orientation_task, align_planes_task, align_perpendicular_task],
)
def test_orientations_and_vectors_of_a_task_are_drawn_at_its_tip(
    rclpy_node, publishing, cylinder_bot_world, create_task
):
    """
    A task's debug markers that have no position of their own start at its tip, not at
    the origin of the frame they are expressed in.
    """
    drive = cylinder_bot_world.get_connections_by_type(OmniDrive)[0]
    cylinder_bot_world.state[drive.x.id].position = 1.0
    cylinder_bot_world.state[drive.y.id].position = 0.5
    cylinder_bot_world.state[drive.yaw.id].position = 0.3
    cylinder_bot_world.notify_state_change()
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[RosNodeAccess(ros_node=rclpy_node), MotionControl(), publishing],
    )
    statechart = Statechart(context=executor.context)
    task = create_task(cylinder_bot_world)
    statechart.add_node(node=task)
    executor.compile(statechart=statechart)
    fake = FakeMarkerPublisher()
    publishing._marker_publisher.marker_publisher.publisher = fake

    executor.tick()
    publishing._marker_publisher.wait_until_published()

    root_P_tip = cylinder_bot_world.compute_forward_kinematics_np(
        cylinder_bot_world.root, task.tip_link
    )[:3, 3]
    origins = marker_origins_in_root(fake.published[-1].markers, cylinder_bot_world)
    assert len(origins) > 0
    np.testing.assert_allclose(origins, [root_P_tip] * len(origins), atol=1e-9)


def test_executor_skips_debug_expressions_by_default(rclpy_node, cylinder_bot_world):
    """
    An executor without :class:`DebugExpressionPublishing` has no such extension.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[RosNodeAccess(ros_node=rclpy_node), MotionControl()],
    )

    with pytest.raises(MissingExecutorExtensionError):
        executor.require_extension(extension_type=DebugExpressionPublishing)
