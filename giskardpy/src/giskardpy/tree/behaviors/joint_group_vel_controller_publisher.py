from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import numpy as np
from py_trees.common import Status
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import MultiDOFCommand

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    PrismaticConnection,
)


class VelocityCommand(ABC):
    """Builds the command message a joint-group velocity controller expects from per-joint velocities."""

    message_type: ClassVar[type]
    """The ROS message type published on the command topic."""

    @abstractmethod
    def create_message(self, velocities: List[float]) -> Any:
        """Pack the per-joint velocities into a fresh message of :attr:`message_type`."""


@dataclass
class Float64MultiArrayVelocityCommand(VelocityCommand):
    """Commands a controller that consumes :class:`std_msgs.msg.Float64MultiArray`."""

    message_type: ClassVar[type] = Float64MultiArray

    def create_message(self, velocities: List[float]) -> Float64MultiArray:
        message = Float64MultiArray()
        message.data = velocities
        return message


@dataclass
class MultiDOFVelocityCommand(VelocityCommand):
    """Commands a controller that consumes :class:`control_msgs.msg.MultiDOFCommand`."""

    message_type: ClassVar[type] = MultiDOFCommand

    def create_message(self, velocities: List[float]) -> MultiDOFCommand:
        message = MultiDOFCommand()
        message.values = velocities
        return message


class JointGroupVelController(GiskardBehavior):
    connections: List[ActiveConnection1DOF]

    minimum_valid_velocity: float
    """Minimum magnitude that small non-prismatic, non-finger joint velocities are raised
    to so the hardware actually moves. A value of ``0.0`` disables clamping."""

    velocity_command: VelocityCommand
    """Strategy that builds the command message published to the controller."""

    def __init__(
        self,
        cmd_topic: str,
        connections: List[ActiveConnection1DOF],
        minimum_valid_velocity: float,
        velocity_command: Optional[VelocityCommand] = None,
    ):
        super().__init__()
        self.cmd_topic = cmd_topic
        self.velocity_command = velocity_command or Float64MultiArrayVelocityCommand()
        self.cmd_pub = rospy.node.create_publisher(
            self.velocity_command.message_type, self.cmd_topic, 10
        )

        self.connections = connections
        self.minimum_valid_velocity = minimum_valid_velocity
        for connection in self.connections:
            connection.has_hardware_interface = True
        self.msg = None
        rospy.node.get_logger().info(
            f"Created publisher for {self.cmd_topic} for {[c.name.name for c in self.connections]}"
        )

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        velocities = [
            self._commanded_velocity(connection) for connection in self.connections
        ]
        self.cmd_pub.publish(self.velocity_command.create_message(velocities))
        return Status.RUNNING

    def _commanded_velocity(self, connection: ActiveConnection1DOF) -> float:
        """Raise a small non-prismatic, non-finger velocity to the minimum valid magnitude."""
        velocity = connection.velocity
        if (
            isinstance(connection, PrismaticConnection)
            or "finger" in connection.name.name
        ):
            return velocity
        absolute_velocity = abs(velocity)
        if absolute_velocity == 0.0 or absolute_velocity >= self.minimum_valid_velocity:
            return velocity
        return self.minimum_valid_velocity * np.sign(velocity)

    def terminate(self, new_status):
        stop_velocities = [0.0 for _ in self.connections]
        self.cmd_pub.publish(self.velocity_command.create_message(stop_velocities))
        super().terminate(new_status)
