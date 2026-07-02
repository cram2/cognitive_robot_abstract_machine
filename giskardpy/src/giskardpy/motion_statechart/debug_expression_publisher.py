from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rclpy.node import Node

from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeVisualization,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    SpatialTypePublisher,
)
from semantic_digital_twin.spatial_types.spatial_types import SpatialType, Vector3
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import DebugExpression
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


@dataclass
class DebugExpressionPublisher:
    """
    Visualizes the debug expressions registered by motion statechart nodes.

    Harvests the spatial debug expressions of every compiled node and keeps them
    updated as the robot moves by delegating to a :class:`SpatialTypePublisher`.
    """

    world: World
    """The world whose state the debug expressions are evaluated against."""

    node: Node
    """The ROS2 node used to create the marker publisher."""

    _publisher: SpatialTypePublisher | None = field(init=False, default=None)
    """The underlying publisher that renders and republishes the debug expressions."""

    def attach(self, motion_statechart: MotionStatechart) -> None:
        """Register the spatial debug expressions of every node for live visualization."""
        requests = [
            self._to_request(debug_expression)
            for debug_expression in motion_statechart.collect_debug_expressions()
            if isinstance(debug_expression.expression, SpatialType)
        ]
        if self._publisher is None:
            self._publisher = SpatialTypePublisher(node=self.node, _world=self.world)
        self._publisher.set_requests(requests)

    def _to_request(
        self, debug_expression: DebugExpression
    ) -> SpatialTypeVisualization:
        """Build a visualization request from a debug expression, anchoring vectors correctly."""
        expression = self._anchored_expression(debug_expression.expression)
        return SpatialTypeVisualization(
            spatial_type=expression,
            color=debug_expression.color,
            namespace=debug_expression.name,
            label=debug_expression.name,
        )

    def _anchored_expression(self, expression: SpatialType) -> SpatialType:
        """Express a vector in its visualisation frame so the rendered arrow points correctly."""
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
        """Stop publishing and deregister from the world's state callbacks."""
        if self._publisher is not None:
            self._publisher.stop()
