from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self

from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    AbstractRobot,
)
from ..world import World
from ..world_description.connections import OmniDrive, ActiveConnection
from ..world_description.world_entity import KinematicStructureEntity


@dataclass
class MinimalRobot(AbstractRobot):
    """
    Creates the bare minimum semantic annotation.
    Used when you only care that there is a robot.
    """

    bodies_of_branch: list[KinematicStructureEntity] = field(default_factory=list)

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def _setup_semantic_annotations(self): ...

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="generic_robot", prefix=world.name),
            root=world.root,
            _world=world,
            bodies_of_branch=world.get_kinematic_structure_entities_of_branch(world.root)
        )

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        for connection in self.connections:
            if isinstance(connection, ActiveConnection):
                connection.has_hardware_interface = True

    def _setup_joint_states(self):
        pass
