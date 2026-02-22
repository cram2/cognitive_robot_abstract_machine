from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Any

from typing_extensions import Tuple, TYPE_CHECKING, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from ..adapters.world_entity_kwargs_tracker import WorldEntityWithIDKwargsTracker
from ..exceptions import (
    NegativeCollisionCheckingDistanceError,
    InvalidBodiesInCollisionCheckError,
    BodyHasNoGeometryError,
)
from ..world_description.world_entity import Body

if TYPE_CHECKING:
    from ..world import World


@dataclass
class CollisionCheck(SubclassJSONSerializer):
    """
    Represents a collision check between two bodies.
    """

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
        """
        Creates a CollisionCheck instance and validates its properties.
        Makes sure body_a and body_b are sorted properly.
        :param body_a: First body in the collision check.
        :param body_b: Second body in the collision check.
        :param distance: Minimum distance to check for collisions.
        :return: Validated CollisionCheck instance.
        """
        self = cls(body_a=body_a, body_b=body_b, distance=distance)
        if self.distance is not None and self.distance < 0:
            raise NegativeCollisionCheckingDistanceError(self)

        if self.body_a == self.body_b:
            raise InvalidBodiesInCollisionCheckError(self)

        if not self.body_a.has_collision():
            raise BodyHasNoGeometryError(self)

        if not self.body_b.has_collision():
            raise BodyHasNoGeometryError(self)
        self.sort_bodies()
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

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "body_a": to_json(self.body_a.id),
            "body_b": to_json(self.body_b.id),
            "distance": to_json(self.distance),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        return cls(
            body_a=tracker.get_world_entity_with_id(id=from_json(data["body_a"])),
            body_b=tracker.get_world_entity_with_id(id=from_json(data["body_b"])),
            distance=data["distance"],
        )


@dataclass
class CollisionMatrix:
    """
    Describes a matrix in sparse format by storing only unique pairs of bodies with collision checks.
    This is the input for collision checking algorithms.
    .. note:: CollisionRule objects are the intended way to modify collision matrices.
    """

    collision_checks: set[CollisionCheck] = field(default_factory=set)
    """
    Set of collision checks that should be performed.
    """

    def __post_init__(self):
        self.sort_bodies()

    def sort_bodies(self):
        for collision in self.collision_checks:
            collision.sort_bodies()

    def __hash__(self):
        return hash(id(self))

    def apply_buffer(self, buffer: float):
        for collision in self.collision_checks:
            collision.distance = (
                collision.distance + buffer if collision.distance else None
            )

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

    _last_world_model_version: int = field(init=False, default=-1)
    """
    Used to prevent updating the collision matrix when the world model has not changed.
    """

    @abstractmethod
    def apply_to_collision_matrix(self, collision_matrix: CollisionMatrix):
        """
        Modifies the collision matrix by adding or removing collision checks.
        """

    def update(self, world: World):
        """
        Updates the collision rule based on the current state of the world, if the world model has changed.
        :param world: The world used for updating
        """
        if world._model_manager.version == self._last_world_model_version:
            return
        self._update(world)
        self._last_world_model_version = world._model_manager.version

    @abstractmethod
    def _update(self, world: World):
        """
        Specific update logic for the collision rule.
        :param world: The world used for updating.
        """


@dataclass
class MaxAvoidedCollisionsRule(ABC):
    """
    Base class for collision rules that define the maximum number of collisions that can be avoided for a given body.
    """

    @abstractmethod
    def get_max_avoided_collisions(self, body: Body) -> int | None:
        """
        :param body: The body for which to get the maximum number of collisions that can be avoided.
        :return: The maximum number of collisions that can be avoided for the given body, or None if no rule is defined for it.
        """


@dataclass
class DefaultMaxAvoidedCollisions(MaxAvoidedCollisionsRule):
    """
    Default implementation of MaxAvoidedCollisionsRule that sets the maximum number of avoided collisions to 1 for all bodies.
    """

    def get_max_avoided_collisions(self, body: Body) -> int | None:
        return 1


@dataclass
class MaxAvoidedCollisionsOverride(MaxAvoidedCollisionsRule):
    """
    Implementation of MaxAvoidedCollisionsRule that overrides the maximum number of avoided collisions for specific bodies.
    """

    value: int
    """
    Maximum number of avoided collisions for the given bodies.
    """
    bodies: set[Body]
    """
    Bodies for which the maximum number of avoided collisions is overridden.
    """

    def get_max_avoided_collisions(self, body: Body) -> int | None:
        if body not in self.bodies:
            return None
        return self.value
