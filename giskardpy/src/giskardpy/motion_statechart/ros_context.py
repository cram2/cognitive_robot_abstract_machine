from dataclasses import dataclass

from semantic_digital_twin.utils import MockedNodeClass

try:
    from rclpy.node import Node
except ImportError:
    Node = MockedNodeClass

from giskardpy.motion_statechart.context import ContextExtension


@dataclass
class RosContextExtension(ContextExtension):
    ros_node: Node
