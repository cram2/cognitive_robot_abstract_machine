from dataclasses import dataclass, field

from rclpy.node import Node

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.ros_context import RosContextExtension


@dataclass
class Ros2Executor(Executor):
    """
    A normal Executor which augments the BuildContext with a ros2 node.
    Required if you want to use MotionStatechartNodes that have ros2 dependencies.
    """

    ros_node: Node = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        self.context.add_extension(RosContextExtension(self.ros_node))
