from dataclasses import dataclass, field

from rclpy.node import Node

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.ros_context import RosContextExtension


@dataclass
class Ros2Executor(Executor):
    """
    A normal Executor which augments the BuildContext with a ros2 node.
    Required if you want to use MotionStatechartNodes that have ros2 dependencies.
    """

    ros_node: Node = field(kw_only=True)

    @property
    def build_context(self) -> BuildContext:
        build_context = BuildContext(
            world=self.world,
            auxiliary_variable_manager=self.auxiliary_variable_manager,
            collision_scene=self.collision_scene,
            qp_controller_config=self.controller_config,
            control_cycle_variable=self._control_cycles_variable,
        )
        build_context.add_extension(RosContextExtension(self.ros_node))
        return build_context
