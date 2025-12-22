from __future__ import annotations

import abc
from dataclasses import dataclass, field
from itertools import chain
from typing_extensions import Tuple, Set, List, Optional, Iterable, TYPE_CHECKING

import numpy as np

from ..world_description.connections import ActiveConnection
from ..world_description.world_entity import Body

if TYPE_CHECKING:
    from ..world import World


@dataclass
class CollisionCheck:
    body_a: Body
    """
    First body in the collision check.
    """
    body_b: Body
    """
    Second body in the collision check.
    """
    distance: float
    """
    Minimum distance to check for collisions.
    """
    _world: World
    """
    The world context for validation and sorting.
    """

    def __post_init__(self):
        self.body_a, self.body_b = self.sort_bodies(self.body_a, self.body_b)

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b

    def bodies(self) -> Tuple[Body, Body]:
        return self.body_a, self.body_b

    def _validate(self) -> None:
        """Validates the collision check parameters."""
        if self.distance <= 0:
            raise ValueError(f"Distance must be positive, got {self.distance}")

        if self.body_a == self.body_b:
            raise ValueError(
                f'Cannot create collision check between the same body "{self.body_a.name}"'
            )

        if not self.body_a.has_collision():
            raise ValueError(f"Body {self.body_a.name} has no collision geometry")

        if not self.body_b.has_collision():
            raise ValueError(f"Body {self.body_b.name} has no collision geometry")

        if self.body_a not in self._world.bodies_with_enabled_collision:
            raise ValueError(
                f"Body {self.body_a.name} is not in list of bodies with collisions"
            )

        if self.body_b not in self._world.bodies_with_enabled_collision:
            raise ValueError(
                f"Body {self.body_b.name} is not in list of bodies with collisions"
            )

        root_chain, tip_chain = self._world.compute_split_chain_of_connections(
            self.body_a, self.body_b
        )
        if all(
            not isinstance(c, ActiveConnection) for c in chain(root_chain, tip_chain)
        ):
            raise ValueError(
                f"Relative pose between {self.body_a.name} and {self.body_b.name} is fixed"
            )

    @classmethod
    def create_and_validate(
        cls, body_a: Body, body_b: Body, distance: float, world: World
    ) -> CollisionCheck:
        """
        Creates a collision check with additional world-context validation.
        Returns None if the check should be skipped (e.g., bodies are linked).
        """
        collision_check = cls(
            body_a=body_a, body_b=body_b, distance=distance, _world=world
        )
        collision_check._validate()
        return collision_check

    def sort_bodies(self, body_a: Body, body_b: Body) -> Tuple[Body, Body]:
        """
        Sort both bodies in a consistent manner, needed to avoid checking B with A, when A with B is already checked.
        """
        if body_a.id > body_b.id:
            body_a, body_b = body_b, body_a
        is_body_a_controlled = self._world.is_body_controlled(body_a)
        is_body_b_controlled = self._world.is_body_controlled(body_b)
        if not is_body_a_controlled and is_body_b_controlled:
            body_a, body_b = body_b, body_a
        return body_a, body_b


@dataclass
class Collision:
    contact_distance: float
    """
    The distance at which the collision was detected.
    """
    body_a: Body = field(default=None)
    """
    First body in the collision.
    """
    body_b: Body = field(default=None)
    """
    Second body in the collision.
    """

    map_P_pa: np.ndarray = field(default=None)
    """
    Contact point on body A with respect to the world frame.
    """
    map_P_pb: np.ndarray = field(default=None)
    """
    Contact point on body B with respect to the world frame.
    """
    map_V_n_input: np.ndarray = field(default=None)
    """
    Contact normal with respect to the world frame.
    """
    a_P_pa: np.ndarray = field(default=None)
    """
    Contact point on body A with respect to the link a frame. 
    """
    b_P_pb: np.ndarray = field(default=None)
    """
    Contact point on body B with respect to the link b frame.
    """

    def __str__(self):
        return f"{self.body_a}|<->|{self.body_b}: {self.contact_distance}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b


@dataclass
class CollisionDetector(abc.ABC):
    """
    Abstract class for collision detectors.
    """

    _world: World

    @abc.abstractmethod
    def sync_world_model(self) -> None:
        """
        Synchronize the collision checker with the current world model
        """

    @abc.abstractmethod
    def sync_world_state(self) -> None:
        """
        Synchronize the collision checker with the current world state
        """

    @abc.abstractmethod
    def check_collisions(
        self, collision_matrix: Optional[Iterable[CollisionCheck]] = None
    ) -> List[Collision]:
        """
        Computes the collisions for all checks in the collision matrix.
        If collision_matrix is None, checks all collisions.
        :param collision_matrix:
        :return: A list of detected collisions.
        """

    @abc.abstractmethod
    def reset_cache(self):
        """
        Reset any caches the collision checker may have.
        """
