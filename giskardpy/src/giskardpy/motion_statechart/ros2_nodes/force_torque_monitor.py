from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicSubscriberNode
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from semantic_digital_twin.spatial_types import Vector3


@dataclass(eq=False, repr=False)
class ForceTorqueNode(TopicSubscriberNode[WrenchStamped]):
    """
    Superclass for all nodes that subscribe to a ROS topic that contains force and
    torque data.
    """

    msg_type: WrenchStamped = field(init=False, default=WrenchStamped)

    def force_as_np(self) -> np.ndarray:
        return np.array(
            [
                self.current_msg.wrench.force.x,
                self.current_msg.wrench.force.y,
                self.current_msg.wrench.force.z,
            ]
        )

    def force_magnitude(self) -> float:
        return float(np.linalg.norm(self.force_as_np()))

    def torque_as_np(self) -> np.ndarray:
        return np.array(
            [
                self.current_msg.wrench.torque.x,
                self.current_msg.wrench.torque.y,
                self.current_msg.wrench.torque.z,
            ]
        )

    def torque_magnitude(self) -> float:
        return float(np.linalg.norm(self.torque_as_np()))


@dataclass(eq=False, repr=False)
class ForceImpactMonitor(ForceTorqueNode):
    """
    This node checks if the force magnitude is above a threshold.
    """

    threshold: float = field(kw_only=True)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        super().on_tick(context)
        if not self.has_msg():
            return ObservationStateValues.UNKNOWN
        if self.force_magnitude() > self.threshold:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False)
class ForceDirectionMonitor(ForceTorqueNode):
    """Returns TRUE when the force projected onto direction exceeds a threshold."""

    direction: Vector3 = field(kw_only=True)
    """Direction in the sensor frame along which the force is measured. Normalized internally"""

    threshold: float = field(kw_only=True)
    """Trigger threshold in N for the force projected onto direction"""

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        super().on_tick(context)
        if not self.has_msg():
            return ObservationStateValues.UNKNOWN
        force = Vector3.from_iterable(self.force_as_np())
        unit_direction = self.direction / self.direction.norm()
        force_along_direction = float(force.dot(unit_direction))
        if force_along_direction > self.threshold:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
