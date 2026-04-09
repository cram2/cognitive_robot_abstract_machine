from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    ActiveConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class MinimalRobot(AbstractRobot):
    """
    Creates the bare minimum semantic annotation.
    Used when you only care that there is a robot.
    """

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Self:
        return world.root

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_other_hardware_interfaces(self):
        for connection in self.connections:
            if isinstance(connection, ActiveConnection):
                connection.has_hardware_interface = True

    def _setup_joint_states(self):
        pass
