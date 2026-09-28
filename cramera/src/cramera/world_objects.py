"""
Native world bodies rendered independently from robot and environment models.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Protocol, runtime_checkable
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from cramera.mesh_format import MeshFormat

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World


# %% native object discovery


@runtime_checkable
class RobotBodyCollection(Protocol):
    """
    The bodies identified by a native robot annotation.
    """

    bodies: list[Body]
    """Bodies belonging to the robot."""


@dataclass
class WorldObjects:
    """
    Select independent objects using native geometry and connections.
    """

    world: World
    """The world whose current bodies are inspected."""

    robot: AbstractRobot | None = None
    """
    The robot represented by its own model, when present.
    """

    def robot_body_names(self) -> set[str]:
        """
        Return bodies belonging to the robot's native annotation.
        """
        if self.robot is None:
            return set()
        bodies = (
            self.robot.bodies
            if isinstance(self.robot, RobotBodyCollection)
            else self.world.get_kinematic_structure_entities_of_branch(self.robot.root)
        )
        return {str(body.name) for body in bodies}

    def free_floating(self) -> list[Body]:
        """
        Select shaped free bodies, excluding robot parts and empty frames.
        """
        robot_names = self.robot_body_names()
        branches = [
            body
            for body in self.world.bodies
            if isinstance(body.parent_connection, Connection6DoF)
            and str(body.name) not in robot_names
        ]
        return self._shaped_branches(branches, robot_names)

    def _shaped_branches(
        self, roots: Iterable[Body], robot_names: set[str]
    ) -> list[Body]:
        """
        Keep each shaped descendant moving with its independent object.

        :param roots: Bodies whose subtrees form independent objects.
        :param robot_names: Native robot bodies represented by their own model.
        :return: Unique shaped bodies outside the robot model.
        """
        descendants = {
            entity
            for root in roots
            for entity in self.world.get_kinematic_structure_entities_of_branch(root)
            if isinstance(entity, Body)
            and str(entity.name) not in robot_names
            and (entity.visual.shapes or entity.collision.shapes)
        }
        return [body for body in self.world.bodies if body in descendants]

    def overlay_bodies(self, previously_published: Iterable[Body] = ()) -> list[Body]:
        """
        Retain tracked object identities through grasp and support attachments.

        :param previously_published: Bodies tracked before the model changed.
        :return: Current independent objects still present in the world.
        """
        tracked = {body.id for body in previously_published}
        tracked.update(body.id for body in self.free_floating())
        root = self.robot.root if self.robot is not None else None
        robot_names = self.robot_body_names()
        roots = [
            body
            for body in self.world.bodies
            if body is not root
            and (
                body.id in tracked
                or (
                    str(body.name) not in robot_names
                    and MeshFormat.of_path(str(body.name).split("/")[-1]) is not None
                )
            )
        ]
        published = set(roots) | set(self._shaped_branches(roots, robot_names))
        return [body for body in self.world.bodies if body in published]
