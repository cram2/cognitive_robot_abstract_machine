from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from rustworkx import rustworkx
from typing_extensions import TYPE_CHECKING

from .collision_manager import CollisionManager, CollisionConsumer
from ..world_description.world_entity import Body, KinematicStructureEntity

if TYPE_CHECKING:
    from ..world import World


@dataclass(repr=False, eq=False)
class CollisionGroup:
    """
    Bodies in this group are viewed as a single entity for collision checking purposes.
    A group is a collection of bodies that are connected by fixed or passive connections.
    """

    root: KinematicStructureEntity
    """
    The root kinematic structure entity of the group. 
    It does not necessarily have to have collision shapes.
    Can be viewed as the reference frame for this group.
    """
    bodies: set[Body] = field(default_factory=set)
    """
    All bodies belonging to the group.
    .. note: `root` is only in bodies, if it is itself a body.
    """

    def __repr__(self) -> str:
        return f"CollisionGroup(root={self.root.name}, bodies={[b.name for b in self.bodies]})"

    def __str__(self) -> str:
        return str(self.root.name)

    def __eq__(self, other) -> bool:
        return self.root == other.root

    def __contains__(self, item):
        return item == self.root or item in self.bodies

    def __hash__(self):
        return hash((self.root, tuple(sorted(self.bodies, key=lambda b: b.id))))

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


@dataclass
class CollisionGroupConsumer(CollisionConsumer, ABC):
    """
    A collision consumer that keeps track of collision groups instead of individual bodies.
    All groups of bodies that are connected by fixed or passive connections are considered as a single collision group.
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
        self.collision_groups = [CollisionGroup(world.root)]
        for parent, children in rustworkx.bfs_successors(
            world.kinematic_structure, world.root.index
        ):
            for child in children:
                parent_C_child = world.get_connection(parent, child)
                if parent_C_child.is_controlled:
                    self.collision_groups.append(CollisionGroup(child))
                else:
                    collision_group = self.get_collision_group(parent)
                    collision_group.bodies.add(child)

        for group in self.collision_groups:
            group.bodies = set(
                b for b in group.bodies if b in world.bodies_with_collision
            )

        self.collision_groups = [
            group
            for group in self.collision_groups
            if len(group.bodies) > 0 or group.root in world.bodies_with_collision
        ]

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

    def is_any_collision_checked(
        self, group_a: CollisionGroup, group_b: CollisionGroup
    ) -> bool:
        for body_a in group_a.bodies:
            for body_b in group_b.bodies:
                if self.collision_manager.is_collision_checked(body_a, body_b):
                    return True
        return False
