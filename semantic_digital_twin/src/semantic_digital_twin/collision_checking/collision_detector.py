from __future__ import annotations

import abc
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from typing_extensions import TYPE_CHECKING, Self

from krrood.symbolic_math.symbolic_math import (
    Matrix,
    VariableParameters,
    CompiledFunction,
)
from .collision_matrix import CollisionMatrix, CollisionCheck
from ..callbacks.callback import ModelChangeCallback, StateChangeCallback
from ..spatial_types import HomogeneousTransformationMatrix
from ..world_description.world_entity import Body

if TYPE_CHECKING:
    from ..world import World


@dataclass
class CollisionCheckingResult:
    contacts: list[Collision] = field(default_factory=list)

    def any(self) -> bool:
        return len(self.contacts) > 0


@dataclass
class Collision:
    body_a: Body
    """
    First body in the collision.
    """
    body_b: Body
    """
    Second body in the collision.
    """

    contact_distance: float
    root_P_pa: np.ndarray
    root_P_pb: np.ndarray
    root_V_n: np.ndarray

    def __str__(self):
        return (
            f"{self.original_body_a}|-|{self.original_body_b}: {self.contact_distance}"
        )

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b

    def reverse(self):
        return Collision(
            body_a=self.body_b,
            body_b=self.body_a,
            root_P_pa=self.root_P_pb,
            root_P_pb=self.root_P_pa,
            root_V_n=-self.root_V_n,
            contact_distance=self.contact_distance,
        )


@dataclass
class CollisionDetectorModelUpdater(ModelChangeCallback):
    collision_detector: CollisionDetector
    world: World = field(init=False)
    compiled_collision_fks: CompiledFunction = field(init=False)

    def __post_init__(self):
        self.world = self.collision_detector.world
        super().__post_init__()

    def _notify(self):
        if self.world.is_empty():
            return
        self.collision_detector.sync_world_model()
        self.compile_collision_fks()

    def compile_collision_fks(self):
        collision_fks = []
        world_root = self.world.root
        for body in self.world.bodies_with_collision:
            if body == world_root:
                if body.has_collision():
                    collision_fks.append(HomogeneousTransformationMatrix())
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
        if self.world.is_empty():
            return
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


@dataclass
class NullCollisionDetector(CollisionDetector):
    def sync_world_model(self) -> None:
        pass

    def sync_world_state(self) -> None:
        pass

    def check_collisions(
        self, collision_matrix: CollisionMatrix
    ) -> CollisionCheckingResult:
        return CollisionCheckingResult()

    def reset_cache(self):
        pass
