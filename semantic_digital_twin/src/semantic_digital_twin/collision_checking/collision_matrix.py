from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations

from typing_extensions import Tuple, TYPE_CHECKING, Self

from giskardpy.motion_statechart.data_types import FloatEnum
from ..world_description.world_entity import Body

if TYPE_CHECKING:
    from ..world import World


@dataclass(repr=False)
class CollisionCheck:
    body_a: Body
    """
    First body in the collision check.
    """
    body_b: Body
    """
    Second body in the collision check.
    """
    distance: float | None = None
    """
    Minimum distance to check for collisions.
    """

    @classmethod
    def create_and_validate(
        cls, body_a: Body, body_b: Body, distance: float | None = None
    ) -> Self:
        self = cls(body_a=body_a, body_b=body_b, distance=distance)
        if self.distance is not None and self.distance < 0:
            raise ValueError(f"Distance must be positive, got {self.distance}")

        if self.body_a == self.body_b:
            raise ValueError(
                f'Cannot create collision check between the same body "{self.body_a.name}"'
            )

        if not self.body_a.has_collision():
            raise ValueError(f"Body {self.body_a.name} has no collision geometry")

        if not self.body_b.has_collision():
            raise ValueError(f"Body {self.body_b.name} has no collision geometry")
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.body_a.name}, {self.body_b.name}, {self.distance})"

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b

    def bodies(self) -> Tuple[Body, Body]:
        return self.body_a, self.body_b

    def sort_bodies(self):
        if self.body_a.id > self.body_b.id:
            self.body_a, self.body_b = self.body_b, self.body_a


@dataclass
class CollisionMatrix:
    """
    Describes a matrix in sparse format by storing only unique pairs of bodies with collision checks.
    This is the input for collision checking algorithms.
    .. note:: CollisionRule objects are the intended way to modify collision matrices.
    """

    collision_checks: set[CollisionCheck] = field(default_factory=set)

    def __post_init__(self):
        self.sort_bodies()

    def sort_bodies(self):
        for collision in self.collision_checks:
            collision.sort_bodies()

    def __hash__(self):
        return hash(id(self))

    @classmethod
    def create_all_checks(cls, distance: float, world: World) -> Self:
        return CollisionMatrix(
            collision_checks={
                CollisionCheck(body_a=body_a, body_b=body_b, distance=distance)
                for body_a, body_b in combinations(world.bodies_with_collision, 2)
            }
        )

    def add_collision_checks(self, collision_checks: set[CollisionCheck]):
        self.collision_checks |= collision_checks

    def remove_collision_checks(self, collision_checks: set[CollisionCheck]):
        self.collision_checks -= collision_checks


@dataclass
class CollisionRule(ABC):
    """
    Base class for collision rules.
    They modify collision matrices by adding or removing collision checks.
    """

    @abstractmethod
    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        pass


@dataclass
class MaxAvoidedCollisionsRule(ABC):
    @abstractmethod
    def get_max_avoided_collisions(self, body: Body) -> int | None: ...


@dataclass
class DefaultMaxAvoidedCollisions(MaxAvoidedCollisionsRule):
    def get_max_avoided_collisions(self, body: Body) -> int | None:
        return 1


@dataclass
class MaxAvoidedCollisionsOverride(MaxAvoidedCollisionsRule):
    value: int
    bodies: set[Body]

    def get_max_avoided_collisions(self, body: Body) -> int | None:
        if body in self.bodies:
            return self.value
        return None
