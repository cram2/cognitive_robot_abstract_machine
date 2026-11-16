from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)
from cramph.node import EndedByOwner
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.node import NodeArtifacts


@dataclass(eq=False, repr=False, kw_only=True)
class JointPositionReached(EndedByOwner, MotionStatechartNode):
    """
    Monitors if a joint position is reached within a certain threshold.
    """

    connection: ActiveConnection1DOF
    """
    Monitored joint connection.
    """

    position: float
    """
    Target position to monitor.
    """

    threshold: float = 0.01
    """
    Threshold for position error.
    """

    def build_artifacts(self) -> NodeArtifacts:
        current = self.connection.dof.variables.position
        if (
            isinstance(self.connection, RevoluteConnection)
            and not self.connection.dof.has_position_limits()
        ):
            error = sm.shortest_angular_distance(current, self.position)
        else:
            error = self.position - current
        return NodeArtifacts(observation=sm.abs(error) < self.threshold)
