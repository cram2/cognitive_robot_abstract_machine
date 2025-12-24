from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    AbstractRobot,
)
from ..world import World
from ..world_description.world_entity import Body


@dataclass
class MinimalRobot(AbstractRobot):
    """
    Creates the bare minimum semantic annotation.
    Used when you only care that there is a robot.
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def setup_collision_config(self):
        pass

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a minimal semantic robot annotation from the given world, starting at root_body
        """

        with world.modify_world():
            robot = cls(
                name=PrefixedName(name="generic_robot", prefix=world.name),
                root=world.root,
                _world=world,
            )

            world.add_semantic_annotation(robot)

            vel_limits = defaultdict(lambda: 1.0)
            robot.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

        return robot
