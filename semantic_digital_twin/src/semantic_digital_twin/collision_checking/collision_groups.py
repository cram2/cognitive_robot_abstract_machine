from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from rustworkx import rustworkx
from typing_extensions import TYPE_CHECKING

from krrood.utils import clear_memoization_cache, memoize
from semantic_digital_twin.collision_checking.collision_manager import (
    CollisionManager,
    CollisionConsumer,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(repr=False, eq=False)
class CollisionGroup:
    """
    Bodies in this group are viewed as a single entity for collision checking purposes.

    A group is a collection of bodies that are connected by fixed or passive
    connections.
    """

    root: Body
    """
    The root kinematic structure entity of the group.

    It does not necessarily have to have collision shapes. Can be viewed as the
    reference frame for this group.
    """

    bodies: set[Body] = field(default_factory=set)
    """
    All bodies belonging to the group.

    .. note: `root` is only in bodies, if it is itself a body.
    """

    def __post_init__(self):
        self.add_body(self.root)

    def __repr__(self) -> str:
        return f"CollisionGroup(root={self.root.name}, bodies={[b.name for b in self.bodies]})"

    def __str__(self) -> str:
        return str(self.root.name)

    def __eq__(self, other) -> bool:
        return self.root == other.root

    def __contains__(self, item):
        return item == self.root or item in self.bodies

    def __hash__(self):
        return hash(self.root)

    def add_body(self, body: Body):
        if body.has_collision():
            self.bodies.add(body)

    def get_max_avoided_bodies(self, collision_manager: CollisionManager) -> int:
        """
        Returns the maximum number of collisions `self` should avoid.

        It is the maximum of all bodies in the group.
        :param collision_manager: collision manager to ask for the max avoided bodies.
        :return: Maximum number of collisions that are allowed for this group.
        """
        max_avoided_bodies = []
        if isinstance(self.root, Body):
            max_avoided_bodies.append(
                collision_manager.get_max_avoided_bodies(self.root)
            )
        max_avoided_bodies.extend(
            collision_manager.get_max_avoided_bodies(b) for b in self.bodies
        )
        return max(max_avoided_bodies, default=1)


@dataclass(eq=False)
class CollisionGroupConsumer(CollisionConsumer, ABC):
    """
    A collision consumer that keeps track of collision groups instead of individual
    bodies.

    All groups of bodies that are connected by fixed or passive connections are
    considered as a single collision group.
    """

    collision_groups: list[CollisionGroup] = field(default_factory=list, init=False)
    """
    Collision groups defined by this post processor.
    """

    def on_world_model_update(self, world: World):
        self.update_collision_groups(world)

    def update_collision_groups(self, world: World):
        """
        Updates the collision groups based on the kinematic structure of the world.

        :param world: Reference to the updated world.
        """
        body_to_robot = world.robot_body_to_robot_mapping

        root_group = CollisionGroup(world.root)
        group_of_entity: dict[KinematicStructureEntity, CollisionGroup] = {
            world.root: root_group
        }
        self.collision_groups = [root_group]
        for parent, children in rustworkx.bfs_successors(
            world.kinematic_structure, world.root.index
        ):
            for child in children:
                parent_C_child = world.get_connection(parent, child)
                if parent_C_child.is_controlled or body_to_robot.get(
                    parent
                ) != body_to_robot.get(child):
                    child_group = CollisionGroup(child)
                    self.collision_groups.append(child_group)
                else:
                    child_group = group_of_entity[parent]
                    child_group.bodies.add(child)
                group_of_entity[child] = child_group

        for group in self.collision_groups:
            group.bodies = set(
                body for body in group.bodies if body in world.bodies_with_collision
            )

        self.collision_groups = [
            group for group in self.collision_groups if len(group.bodies) > 0
        ]

        # The memoized body-to-group mapping refers to the groups of the previous model;
        # without clearing it, lookups keep returning (and mutating) those dead groups.
        clear_memoization_cache(self)

    @memoize
    def get_collision_group(self, body: KinematicStructureEntity) -> CollisionGroup:
        """
        Ever body belongs to at most one collision group.

        :param body: the body for which to get the collision group.
        :return: the collision group for the given body.
        """
        for group in self.collision_groups:
            if body in group:
                return group
        raise Exception(f"No collision group found for {body}")
