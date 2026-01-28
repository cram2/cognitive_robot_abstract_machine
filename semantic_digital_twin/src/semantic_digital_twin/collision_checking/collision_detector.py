from __future__ import annotations

import abc
from dataclasses import dataclass, field
from itertools import chain, combinations
from uuid import UUID

import numpy as np
from typing_extensions import Tuple, List, Optional, Iterable, TYPE_CHECKING, Self

from krrood.symbolic_math.symbolic_math import (
    Matrix,
    VariableParameters,
    CompiledFunction,
)
from ..callbacks.callback import ModelChangeCallback, StateChangeCallback
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

    @classmethod
    def create_and_validate(cls, body_a: Body, body_b: Body, distance: float) -> Self:
        self = cls(body_a=body_a, body_b=body_b, distance=distance)
        if self.distance < 0:
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
    collision_checks: set[CollisionCheck] = field(default_factory=set)

    def __post_init__(self):
        self.sort_bodies()

    def sort_bodies(self):
        for collision in self.collision_checks:
            collision.sort_bodies()

    @classmethod
    def create_all_checks(cls, distance: float, world: World) -> Self:
        return CollisionMatrix(
            collision_checks={
                CollisionCheck(body_a=body_a, body_b=body_b, distance=distance)
                for body_a, body_b in combinations(
                    world.bodies_with_enabled_collision, 2
                )
            }
        )


@dataclass
class CollisionCheckingResult:
    contacts: list[Collision] = field(default_factory=list)

    def any(self) -> bool:
        return len(self.contacts) > 0


@dataclass
class Collision:
    contact_distance: float
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
class CollisionDetectorModelUpdater(ModelChangeCallback):
    collision_detector: CollisionDetector
    world: World = field(init=False)
    compiled_collision_fks: CompiledFunction = field(init=False)

    def __post_init__(self):
        self.world = self.collision_detector.world
        super().__post_init__()

    def _notify(self):
        self.collision_detector.sync_world_model()
        self.compile_collision_fks()

    def compile_collision_fks(self):
        collision_fks = []
        world_root = self.world.root
        for body in self.world.bodies_with_enabled_collision:
            if body == world_root:
                continue
            collision_fks.append(
                self.world.compose_forward_kinematics_expression(world_root, body)
            )
        collision_fks = Matrix.vstack(collision_fks)

        self.compiled_collision_fks = collision_fks.compile(
            parameters=VariableParameters.from_lists(
                self.world.state.position_float_variables
            )
        )
        self.compiled_collision_fks.bind_args_to_memory_view(
            0, self.world.state.positions
        )

    def compute(self) -> np.ndarray:
        return self.compiled_collision_fks.evaluate()


@dataclass
class CollisionDetectorStateUpdater(StateChangeCallback):
    collision_detector: CollisionDetector
    world: World = field(init=False)

    def __post_init__(self):
        self.world = self.collision_detector.world
        super().__post_init__()

    def _notify(self):
        self.collision_detector.world_model_updater.compiled_collision_fks.evaluate()
        self.collision_detector.sync_world_state()


@dataclass
class CollisionDetector(abc.ABC):
    """
    Abstract class for collision detectors.
    """

    world: World
    world_model_updater: CollisionDetectorModelUpdater = field(init=False)
    world_state_updater: CollisionDetectorStateUpdater = field(init=False)

    def __post_init__(self):
        self.world_model_updater = CollisionDetectorModelUpdater(
            collision_detector=self
        )
        self.world_state_updater = CollisionDetectorStateUpdater(
            collision_detector=self
        )
        self.world_model_updater.notify()
        self.world_state_updater.notify()

    def get_all_collision_fks(self) -> np.ndarray:
        return self.world_model_updater.compiled_collision_fks._out

    def get_collision_fk(self, body_id: UUID):
        pass

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
        self, collision_matrix: CollisionMatrix
    ) -> CollisionCheckingResult:
        """
        Computes the collisions for all checks in the collision matrix.
        If collision_matrix is None, checks all collisions.
        :param collision_matrix:
        :return: A list of detected collisions.
        """

    def check_collision_between_bodies(
        self, body_a: Body, body_b: Body, distance: float = 0.0
    ) -> Collision | None:
        collision = self.check_collisions(
            CollisionMatrix(
                {CollisionCheck.create_and_validate(body_a, body_b, distance)}
            )
        )
        return collision.contacts[0] if collision.any() else None

    @abc.abstractmethod
    def reset_cache(self):
        """
        Reset any caches the collision checker may have.
        """
