from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self

from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
)
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class MinimalRobot(AbstractRobot):
    """
    Creates the bare minimum semantic annotation.
    Used when you only care that there is a robot.
    """

    bodies_of_branch: list[KinematicStructureEntity] = field(default_factory=list)

    @classmethod
    def _get_root_body_name(cls) -> str:
        # Minimal robot uses the world root as the robot root
        return ""

    @classmethod
    def from_branch_in_world(cls, branch_root: KinematicStructureEntity) -> Self:
        """
        Creates a robot from a branch in a world.
        This is useful when you have multiple of the same robots in the same world, which would normally cause naming conflicts.
        """
        world = branch_root._world
        with world.modify_world():
            self = cls(
                root=branch_root,
            )
            world.add_semantic_annotation(self)
            self.setup_robot_part_semantic_annotations()
            for robot_part in self._robot_parts:
                robot_part.setup_hardware_interfaces()
                robot_part.setup_joint_states()
            return self

    def setup_robot_part_semantic_annotations(self):
        self.bodies_of_branch = self._world.get_kinematic_structure_entities_of_branch(
            self.root
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
