from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

from rclpy.node import Node

from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeVisualization,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    SpatialTypePublisher,
)
from semantic_digital_twin.spatial_types.spatial_types import SpatialType, Vector3
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from semantic_digital_twin.world import World
from giskardpy.motion_statechart.graph_node import DebugExpression

if TYPE_CHECKING:
    from cramph.statechart import Statechart


@dataclass
class DebugExpressionPublisher:
    """
    Visualizes the debug expressions registered by motion statechart nodes.

    Harvests the spatial debug expressions of every compiled node and keeps them updated
    as the robot moves by delegating to a :class:`SpatialTypePublisher`.
    """

    world: World
    """The world whose state the debug expressions are evaluated against."""

    node: Node
    """
    The ROS2 node used to create the marker publisher.
    """

    _publisher: SpatialTypePublisher | None = field(init=False, default=None)
    """
    The underlying publisher that renders and republishes the debug expressions.
    """

    def attach(self, statechart: Statechart) -> None:
        """
        Register the spatial debug expressions of every node for live visualization.
        """
        requests = [
            self._to_request(debug_expression)
            for debug_expression in DebugExpression.collect_from(statechart)
            if isinstance(debug_expression.expression, SpatialType)
        ]
        if self._publisher is None:
            self._publisher = SpatialTypePublisher(node=self.node, _world=self.world)
        self._publisher.set_requests(requests)

    def _to_request(
        self, debug_expression: DebugExpression
    ) -> SpatialTypeVisualization:
        """
        Build a visualization request from a debug expression, anchoring vectors
        correctly.
        """
        expression = self._anchored_expression(debug_expression.expression)
        return SpatialTypeVisualization(
            spatial_type=expression,
            color=debug_expression.color,
            namespace=debug_expression.name,
            label=debug_expression.name,
        )

    def _anchored_expression(self, expression: SpatialType) -> SpatialType:
        """
        Express a vector in its visualisation frame so the rendered arrow points
        correctly.
        """
        if not isinstance(expression, Vector3):
            return expression
        visualisation_frame = expression.visualisation_frame
        reference_frame = expression.reference_frame
        if visualisation_frame is None or reference_frame is None:
            return expression
        if visualisation_frame is reference_frame:
            return expression
        return self.world.transform(expression, visualisation_frame)

    def stop(self) -> None:
        """
        Delete published markers, stop publishing, and deregister from the world's state
        callbacks.
        """
        if self._publisher is not None:
            self._publisher.clear()
            self._publisher.stop()


@dataclass
class DebugExpressionPublishing(TickingExecutorExtension):
    """
    Visualizes the debug expressions of every compiled statechart as RViz markers.

    .. warning::
        You should only use this while debugging and preferably only in simulation,
        because it slows down the control loop.
    """

    ros_node: Node
    """
    The ROS2 node used to create the marker publisher.
    """

    publisher: DebugExpressionPublisher | None = field(init=False, default=None)
    """
    The publisher of the most recently compiled statechart, None before the first
    compile.
    """

    def after_compile(self, executor: StatechartExecutor) -> None:
        if self.publisher is not None:
            self.publisher.stop()
        self.publisher = DebugExpressionPublisher(
            world=executor.context.world, node=self.ros_node
        )
        self.publisher.attach(executor.statechart)
