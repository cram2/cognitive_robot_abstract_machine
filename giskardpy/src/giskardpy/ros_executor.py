from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from semantic_digital_twin.utils import MockedNodeClass

try:
    from rclpy.node import Node
except ImportError:
    Node = MockedNodeClass

from giskardpy.executor import Executor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros_context import RosContextExtension

if TYPE_CHECKING:
    from giskardpy.motion_statechart.debug_expression_publisher import (
        DebugExpressionPublisher,
    )


@dataclass
class Ros2Executor(Executor):
    """
    A normal Executor which augments the BuildContext with a ros2 node.
    Required if you want to use MotionStatechartNodes that have ros2 dependencies.
    """

    ros_node: Node = field(kw_only=True)

    publish_debug_expressions: bool = field(kw_only=True, default=False)
    """
    Whether the debug expressions of the compiled nodes are visualized as RViz markers.

    .. warning::
        You should only use these tools actively while debugging and preferably only in simulation, because it slows down the control loop.
    """

    _debug_expression_publisher: DebugExpressionPublisher | None = field(
        init=False, default=None
    )
    """The publisher visualizing the debug expressions, created on compile when enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.context.add_extension(RosContextExtension(self.ros_node))

    def compile(self, motion_statechart: MotionStatechart):
        super().compile(motion_statechart)
        if self._debug_expression_publisher is not None:
            self._debug_expression_publisher.stop()
            self._debug_expression_publisher = None
        if not self.publish_debug_expressions:
            return
        # Deferred: DebugExpressionPublisher pulls in rclpy unconditionally, which would
        # otherwise make giskardpy.ros_executor unimportable without ROS2 installed.
        from giskardpy.motion_statechart.debug_expression_publisher import (
            DebugExpressionPublisher,
        )

        self._debug_expression_publisher = DebugExpressionPublisher(
            world=self.context.world, node=self.ros_node
        )
        self._debug_expression_publisher.attach(motion_statechart)
