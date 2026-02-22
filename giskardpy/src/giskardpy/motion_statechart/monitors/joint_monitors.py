from dataclasses import field, dataclass

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)
from ..context import MotionStatechartContext
from ..graph_node import MotionStatechartNode, NodeArtifacts


@dataclass(eq=False, repr=False)
class JointPositionReached(MotionStatechartNode):
    """
    Monitors if a joint position is reached within a certain threshold.
    """

    connection: ActiveConnection1DOF = field(kw_only=True)
    """
    Monitored joint connection.
    """
    position: float = field(kw_only=True)
    """
    Target position to monitor.
    """
    threshold: float = field(default=0.01, kw_only=True)
    """
    Threshold for position error.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        current = self.connection.dof.variables.position
        if (
            isinstance(self.connection, RevoluteConnection)
            and not self.connection.dof.has_position_limits()
        ):
            error = sm.shortest_angular_distance(current, self.position)
        else:
            error = self.position - current
        return NodeArtifacts(observation=sm.abs(error) < self.threshold)
