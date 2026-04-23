from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
)


@dataclass(eq=False)
class MinimalRobot(AbstractRobot):
    """
    Creates the bare minimum semantic annotation.
    Used when you only care that there is a robot.
    """

    def _setup_robot_parts(self):
        for connection in self.connections:
            if isinstance(connection, ActiveConnection):
                connection.has_hardware_interface = True
        super()._setup_robot_parts()

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Self:
        return world.root

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_joint_states(self):
        pass
