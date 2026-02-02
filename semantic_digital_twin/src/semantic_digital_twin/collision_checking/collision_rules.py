from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from lxml import etree
from typing_extensions import List, Protocol, TYPE_CHECKING, runtime_checkable, Self

from .collision_matrix import (
    CollisionRule,
    CollisionMatrix,
    CollisionCheck,
)
from ..robots.abstract_robot import AbstractRobot
from ..world import World
from ..world_description.world_entity import Body, CollisionCheckingConfig

if TYPE_CHECKING:
    pass


class HasBodies(Protocol):
    bodies: List[Body]


@dataclass
class AvoidCollisionRule(CollisionRule):
    buffer_zone_distance: float = field(default=0.05)
    """
    Distance defining a buffer zone around the entity. The buffer zone represents a soft boundary where
    proximity should be monitored but minor violations are acceptable.
    """

    violated_distance: float = field(default=0.0)
    """
    Critical distance threshold that must not be violated. Any proximity below this threshold represents
    a severe collision risk requiring immediate attention.
    """


@dataclass
class AvoidCollisionBetweenGroups(AvoidCollisionRule):
    body_group1: List[Body] = field(default_factory=list)
    body_group2: List[Body] = field(default_factory=list)

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_checks = set()
        for body_a in self.body_group1:
            for body_b in self.body_group2:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
                )
                collision_checks.add(collision_check)
        collision_matrix.add_collision_checks(collision_checks)


@dataclass
class AvoidAllCollisions(AvoidCollisionRule):
    bodies: List[Body] = field(default_factory=list)

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_checks = set()
        for body_a, body_b in combinations(self.bodies, 2):
            collision_check = CollisionCheck.create_and_validate(
                body_a=body_a, body_b=body_b, distance=self.buffer_zone_distance
            )
            collision_checks.add(collision_check)
        collision_matrix.add_collision_checks(collision_checks)


@dataclass
class AllowCollisionBetweenGroups(CollisionRule):
    body_group1: List[Body] = field(default_factory=list)
    body_group2: List[Body] = field(default_factory=list)

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_checks = set()
        for body_a in self.body_group1:
            for body_b in self.body_group2:
                if body_a == body_b:
                    continue
                collision_check = CollisionCheck.create_and_validate(
                    body_a=body_a, body_b=body_b, distance=0
                )
                collision_checks.add(collision_check)
        collision_matrix.remove_collision_checks(collision_checks)


@runtime_checkable
class Updatable(Protocol):
    def update(self, world: World):
        pass


@dataclass
class HighPriorityAllowCollisionRule(CollisionRule):
    allowed_collision_pairs: set[CollisionCheck] = field(default_factory=set)
    allowed_collision_bodies: set[Body] = field(default_factory=set)

    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        collision_matrix.remove_collision_checks(self.allowed_collision_pairs)
        collision_matrix.collision_checks = {
            collision_check
            for collision_check in collision_matrix.collision_checks
            if collision_check not in self.allowed_collision_bodies
        }


@dataclass
class AllowNonRobotCollisions(HighPriorityAllowCollisionRule):

    def update(self, world: World):
        """
        Disable collision checks between bodies that do not belong to any robot.
        """
        # Bodies that are part of any robot and participate in collisions
        robot_bodies: set[Body] = {
            body
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies_with_collisions
        }

        # Bodies with collisions that are NOT part of a robot
        non_robot_bodies: set[Body] = (
            set(world.bodies_with_enabled_collision) - robot_bodies
        )
        if not non_robot_bodies:
            return

        # Disable every unordered pair (including self-collisions) exactly once
        for a, b in combinations(non_robot_bodies, 2):
            self.allowed_collision_pairs.add(
                CollisionCheck.create_and_validate(body_a=a, body_b=b, distance=0)
            )


@dataclass
class AllowCollisionForAdjacentPairs(HighPriorityAllowCollisionRule):

    def update(self, world: World):
        for body_a, body_b in combinations(world.bodies_with_collision, 2):
            if not world.is_controlled_connection_in_chain(body_a, body_b):
                self.allowed_collision_pairs.add(
                    CollisionCheck.create_and_validate(body_a, body_b)
                )


@dataclass
class SelfCollisionMatrixRule(HighPriorityAllowCollisionRule):

    def compute_collision_matrix(self, world: World) -> set[CollisionCheck]:
        """
        Parses the collision requrests and (temporary) collision configs in the world
        to create a set of collision checks.
        """
        collision_matrix: set[CollisionCheck] = set()
        for collision_request in self.collision_requests:
            if collision_request.all_bodies_for_group1():
                view_1_bodies = world.bodies_with_enabled_collision
            else:
                view_1_bodies = collision_request.body_group1
            if collision_request.all_bodies_for_group2():
                view2_bodies = world.bodies_with_enabled_collision
            else:
                view2_bodies = collision_request.body_group2
            disabled_pairs = world._collision_pair_manager.disabled_collision_pairs
            for body1 in view_1_bodies:
                for body2 in view2_bodies:
                    collision_check = CollisionCheck(
                        body_a=body1, body_b=body2, distance=0
                    )
                    (robot_body, env_body) = collision_check.bodies()
                    if (robot_body, env_body) in disabled_pairs:
                        continue
                    if collision_request.distance is None:
                        distance = max(
                            robot_body.get_collision_config().buffer_zone_distance
                            or 0.0,
                            env_body.get_collision_config().buffer_zone_distance or 0.0,
                        )
                    else:
                        distance = collision_request.distance
                    if not collision_request.is_allow_collision():
                        collision_check.distance = distance
                        collision_check._validate()
                    if collision_request.is_allow_collision():
                        if collision_check in collision_matrix:
                            collision_matrix.remove(collision_check)
                    if collision_request.is_avoid_collision():
                        if collision_request.is_distance_set():
                            collision_matrix.add(collision_check)
                        else:
                            collision_matrix.add(collision_check)
        return collision_matrix

    @classmethod
    def from_collision_srdf(cls, file_path: str, world: World) -> Self:
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        self = cls()
        SRDF_DISABLE_ALL_COLLISIONS: str = "disable_all_collisions"
        SRDF_DISABLE_SELF_COLLISION: str = "disable_self_collision"
        SRDF_MOVEIT_DISABLE_COLLISIONS: str = "disable_collisions"

        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()

        children_with_tag = [child for child in srdf_root if hasattr(child, "tag")]

        child_disable_collisions = [
            c for c in children_with_tag if c.tag == SRDF_DISABLE_ALL_COLLISIONS
        ]

        for c in child_disable_collisions:
            body = world.get_body_by_name(c.attrib["link"])
            self.allowed_collision_bodies.add(body)

        child_disable_moveit_and_self_collision = [
            c
            for c in children_with_tag
            if c.tag in {SRDF_MOVEIT_DISABLE_COLLISIONS, SRDF_DISABLE_SELF_COLLISION}
        ]

        disabled_collision_pairs = [
            (body_a, body_b)
            for child in child_disable_moveit_and_self_collision
            if (body_a := world.get_body_by_name(child.attrib["link1"])).has_collision()
            and (
                body_b := world.get_body_by_name(child.attrib["link2"])
            ).has_collision()
        ]

        for body_a, body_b in disabled_collision_pairs:
            self.allowed_collision_pairs.add(CollisionCheck(body_a, body_b))
        return self
