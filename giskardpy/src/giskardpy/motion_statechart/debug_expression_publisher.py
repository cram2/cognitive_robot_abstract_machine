"""
Visualizes the debug expressions of running motion statechart nodes as RViz markers.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from rclpy.node import Node

from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from giskardpy.motion_statechart.exceptions import MarkerPublishingThreadEndedError
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    MotionStatechartNode,
)
from krrood_spatial_math.spatial_types import SpatialType, Vector3
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeVisualization,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    CompiledSpatialTypeVisualizations,
    EvaluatedSpatialTypePublisher,
    EvaluatedSpatialTypeVisualization,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% publishing on a thread of its own


@dataclass(kw_only=True)
class MarkerSubmission:
    """
    Evaluated visualizations handed over to be published together.
    """

    evaluated_visualizations: list[EvaluatedSpatialTypeVisualization]
    """
    The visualizations to show; the markers of all others are deleted.
    """


@dataclass(eq=False, kw_only=True)
class BackgroundMarkerPublisher(HasROS2Node):
    """
    Renders and publishes submitted visualizations on a daemon thread of its own, so the
    submitting thread never waits for it.

    Only the newest submission is published; a submission that is replaced before the
    thread takes it is dropped.
    """

    marker_publisher: EvaluatedSpatialTypePublisher = field(init=False)
    """
    Renders and publishes the submissions on :attr:`node`.
    """

    _submission: MarkerSubmission | None = field(init=False, default=None)
    """
    The submission that waits to be published.
    """

    _submission_being_published: MarkerSubmission | None = field(
        init=False, default=None
    )
    """
    The submission the thread took and publishes right now.
    """

    _condition: threading.Condition = field(
        init=False, default_factory=threading.Condition
    )
    """
    Guards :attr:`_submission`, :attr:`_submission_being_published` and
    :attr:`publishing_error`, and signals their changes.
    """

    publishing_error: BaseException | None = field(init=False, default=None)
    """
    The exception a publish raised, which ended the thread; nothing submitted is
    published from then on.

    None while the thread publishes.
    """

    _thread: threading.Thread = field(init=False)
    """
    Publishes the submissions until publishing raises.
    """

    def __post_init__(self):
        """
        Start the thread publishing the submissions.
        """
        self.marker_publisher = EvaluatedSpatialTypePublisher(node=self.node)
        self._thread = threading.Thread(
            target=self._publish_submissions,
            daemon=True,
            name="debug-marker-publisher",
        )
        self._thread.start()

    def submit(self, *, submission: MarkerSubmission) -> None:
        """
        Publish `submission` as soon as the thread is free, replacing a submission that
        still waits.

        :param submission: Visualizations whose rendered values nothing changes anymore.
        :raises MarkerPublishingThreadEndedError: If publishing raised before, caused by
            :attr:`publishing_error`.
        """
        with self._condition:
            if self.publishing_error is not None:
                raise MarkerPublishingThreadEndedError() from self.publishing_error
            self._submission = submission
            self._condition.notify_all()

    def wait_until_published(self) -> None:
        """
        Block until every submission was published or dropped and the thread holds none,
        or until publishing stopped.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self.publishing_error is not None
                    or (
                        self._submission is None
                        and self._submission_being_published is None
                    )
                )
            )

    def _publish_submissions(self) -> None:
        """
        Publish every submission the thread takes, until publishing raises.
        """
        while True:
            self._publish_next_submission()

    def _publish_next_submission(self) -> None:
        """
        Wait for a submission and publish it, releasing it afterwards even if publishing
        raises, in which case publishing stops.
        """
        with self._condition:
            self._condition.wait_for(lambda: self._submission is not None)
            submission = self._submission
            self._submission_being_published = submission
            self._submission = None
        try:
            self.marker_publisher.publish(
                evaluated_visualizations=submission.evaluated_visualizations
            )
        finally:
            with self._condition:
                self._submission_being_published = None
                self.publishing_error = sys.exception()
                self._condition.notify_all()


# %% the executor extension


@dataclass(kw_only=True)
class DebugExpressionPublishing(TickingExecutorExtension):
    """
    Publishes the spatial debug expressions of every running node as RViz markers.

    They are compiled with the statechart, evaluated on the control loop at
    :attr:`publish_frequency` and rendered and published on a thread of its own.

    .. warning:: Switch on :attr:`publishes_markers` only while debugging, because the
        evaluation slows down the control loop.
    """

    ros_node: Node
    """
    The ROS2 node used to create the marker publisher.
    """

    publish_frequency: float = 10.0
    """
    How often per second of control time the markers are published.
    """

    publishes_markers: bool = True
    """
    Whether the statecharts compiled from now on publish their debug expressions; while
    off, none are collected, compiled, evaluated or published.
    """

    _marker_publisher: BackgroundMarkerPublisher = field(init=False)
    """
    Renders and publishes the evaluated debug expressions off the control loop.
    """

    _visualizations_per_node: dict[
        MotionStatechartNode, list[SpatialTypeVisualization]
    ] = field(init=False, default_factory=dict)
    """
    The visualizations of the spatial debug expressions of each node that has any.
    """

    _compiled_visualizations: CompiledSpatialTypeVisualizations | None = field(
        init=False, default=None
    )
    """
    Evaluates the visualizations of every node in :attr:`_visualizations_per_node`, in
    its order; None while the current statechart publishes no markers.
    """

    _last_published_tick: int | None = field(init=False, default=None)
    """
    The tick count of the last publish, None if nothing was published since compiling.
    """

    def __post_init__(self):
        """
        Create the publisher whose thread publishes the debug markers.
        """
        self._marker_publisher = BackgroundMarkerPublisher(node=self.ros_node)

    def after_compile(self, executor: StatechartExecutor) -> None:
        """
        Collect and compile the spatial debug expressions of every node of the compiled
        statechart, replacing those of the previous one, and start publishing again if a
        publish raised before.
        """
        self._release_visualizations()
        if self._marker_publisher.publishing_error is not None:
            self._marker_publisher = BackgroundMarkerPublisher(node=self.ros_node)
        if not self.publishes_markers:
            return
        world = executor.context.world
        for node in executor.statechart.get_nodes_by_type(MotionStatechartNode):
            visualizations = [
                self._create_visualization(
                    debug_expression=debug_expression, node=node, world=world
                )
                for debug_expression in node.debug_expressions
                if isinstance(debug_expression.expression, SpatialType)
            ]
            if visualizations:
                self._visualizations_per_node[node] = visualizations
        if not self._visualizations_per_node:
            return
        self._compiled_visualizations = CompiledSpatialTypeVisualizations(
            visualizations=[
                visualization
                for visualizations in self._visualizations_per_node.values()
                for visualization in visualizations
            ],
            root_frame_name=str(world.root.name),
        )

    def after_tick(self, executor: StatechartExecutor) -> None:
        """
        Publish the debug expressions of the running nodes, if a publish period passed
        since the last publish.
        """
        if self._compiled_visualizations is None:
            return
        ticks_between_publishes = max(
            1,
            round(
                1 / (self.publish_frequency * executor.context.require_tick_duration())
            ),
        )
        if (
            self._last_published_tick is not None
            and executor.tick_count - self._last_published_tick
            < ticks_between_publishes
        ):
            return
        self._last_published_tick = executor.tick_count
        owning_nodes = [
            node
            for node, visualizations in self._visualizations_per_node.items()
            for _ in visualizations
        ]
        self._marker_publisher.submit(
            submission=MarkerSubmission(
                evaluated_visualizations=[
                    evaluated_visualization
                    for evaluated_visualization, node in zip(
                        self._compiled_visualizations.evaluate(),
                        owning_nodes,
                        strict=True,
                    )
                    if node.life_cycle_state == LifeCycleValues.RUNNING
                ]
            )
        )

    def after_run(self, executor: StatechartExecutor) -> None:
        """
        Delete the markers published since compiling, unless publishing stopped.
        """
        # Only a publish in progress can stop publishing, so this wait settles it.
        self._marker_publisher.wait_until_published()
        if (
            self._last_published_tick is not None
            and self._marker_publisher.publishing_error is None
        ):
            self._marker_publisher.submit(
                submission=MarkerSubmission(evaluated_visualizations=[])
            )
        self._release_visualizations()

    def _release_visualizations(self) -> None:
        """
        Forget the visualizations of the previous statechart.
        """
        self._visualizations_per_node = {}
        self._compiled_visualizations = None
        self._last_published_tick = None

    @staticmethod
    def _create_visualization(
        *, debug_expression: DebugExpression, node: MotionStatechartNode, world: World
    ) -> SpatialTypeVisualization:
        """
        :param debug_expression: A debug expression whose expression is a spatial type.
        :param node: The node that registered `debug_expression`.
        :param world: The world the expression is expressed in.
        :return: The visualization of `debug_expression`, labelled with its name, in a
            marker namespace no other node shares, and with a vector expressed in its
            visualisation frame so the arrow starts there.
        """
        expression = debug_expression.expression
        if (
            isinstance(expression, Vector3)
            and expression.visualisation_frame is not None
            and expression.reference_frame is not None
            and expression.visualisation_frame is not expression.reference_frame
        ):
            expression = world.transform(expression, expression.visualisation_frame)
        return SpatialTypeVisualization(
            spatial_type=expression,
            color=debug_expression.color,
            namespace=f"{debug_expression.name}#{node.index}",
            label=debug_expression.name,
        )
