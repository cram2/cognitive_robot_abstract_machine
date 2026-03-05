from typing import Optional
from dataclasses import dataclass, field

from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.context import MotionStatechartContext
from krrood.symbolic_math.symbolic_math import trinary_logic_and

from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass(eq=False, repr=False)
class UnlatchDoor(Goal):
    tip_link: KinematicStructureEntity = field(kw_only=True)
    handle_name: KinematicStructureEntity = field(kw_only=True)
    handle_limit: Optional[float] = field(default=None, kw_only=True)

    def expand(self, context: MotionStatechartContext) -> None:
        handle_connection = self.handle_name.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        max_limit_handle = handle_connection.dof.limits.upper.position

        if self.handle_limit is None:
            limit_handle = max_limit_handle
        else:
            limit_handle = min(max_limit_handle, self.handle_limit)

        self.add_nodes(
            [
                JointPositionReached(
                    connection=handle_connection,
                    position=limit_handle,
                    threshold=0.005,
                    name=f"{self.name}_handle_joint_monitor",
                ),
                Open(
                    tip_link=self.tip_link,
                    environment_link=self.handle_name,
                    goal_joint_state=limit_handle,
                    name="OpenHandle",
                ),
            ]
        )

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=trinary_logic_and(
                *[node.observation_variable for node in self.nodes]
            )
        )
