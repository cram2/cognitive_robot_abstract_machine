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
from ..world_description.world_entity import (
    Body,
    WorldEntityWithID,
    WorldEntityWithClassBasedID,
)


@dataclass
class CollisionCheckingResult:
    """
    Result of a collision checking operation.
    """

    contacts: list[ClosestPoints] = field(default_factory=list)
    """
    List of contacts detected during the collision checking operation.
    """

    def any(self) -> bool:
        """
        Check if there are any contacts in the result.
        :return: True if there are contacts, False otherwise.
        """
        return len(self.contacts) > 0


@dataclass
class ClosestPoints:
    """
    Encapsulates the closest points data between two bodies returned by the collision detector.
    """

    body_a: Body
    """
    First body in the collision.
    """
    body_b: Body
    """
    Second body in the collision.
    """

    distance: float
    """
    Closest distance between the two bodies.
    """
    root_P_point_on_body_a: np.ndarray
    """
    Closest point on body_a with respect to the worlds root.
    """
    root_P_point_on_body_b: np.ndarray
    """
    Closest point on body_b with respect to the worlds root.
    """
    root_V_contact_normal_from_b_to_a: np.ndarray
    """
    Normal vector of the contact plane from body_b to body_a with respect to the worlds root.
    The contact normal points from body_a to body_b.
    """

    def __str__(self):
        return f"{self.body_a}|-|{self.body_b}: {self.distance}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b

    def reverse(self):
        """
        Returns a new ClosestPoints object with the same data but with body_a and body_b swapped.
        """
        return ClosestPoints(
            body_a=self.body_b,
            body_b=self.body_a,
            root_P_point_on_body_a=self.root_P_point_on_body_b,
            root_P_point_on_body_b=self.root_P_point_on_body_a,
            root_V_contact_normal_from_b_to_a=-self.root_V_contact_normal_from_b_to_a,
            distance=self.distance,
        )


@dataclass(eq=False)
class CollisionDetectorModelUpdater(ModelChangeCallback):
    """
    Updates and compiles the collision detector's collision forward kinematics expressions when the world model changes.
    """

    collision_detector: CollisionDetector = field(kw_only=True)
    """
    Reference to the collision detector.
    """
    compiled_collision_fks: CompiledFunction = field(init=False)
    """
    Compiled collision FK function.
    """

    def __post_init__(self):
        self._world = self.collision_detector._world
        super().__post_init__()

    def _notify(self, **kwargs):
        if self._world.is_empty():
            return
        self.collision_detector.sync_world_model()
        self.compile_collision_fks()

    def compile_collision_fks(self):
        """
        Compile the collision FK functions for all bodies with collision.
        """
        collision_fks = []
        world_root = self._world.root
        for body in self._world.bodies_with_collision:
            if body == world_root:
                if body.has_collision():
                    collision_fks.append(HomogeneousTransformationMatrix())
                continue
            collision_fks.append(
                self._world.compose_forward_kinematics_expression(world_root, body)
            )
        collision_fks = Matrix.vstack(collision_fks)

        self.compiled_collision_fks = collision_fks.compile(
            parameters=VariableParameters.from_lists(
                self._world.state.position_float_variables
            )
        )
        if not collision_fks.is_constant():
            self.compiled_collision_fks.bind_args_to_memory_view(
                0, self._world.state.positions
            )

    def compute(self) -> np.ndarray:
        return self.compiled_collision_fks.evaluate()


@dataclass
class CollisionDetectorStateUpdater(StateChangeCallback):
    """
    Updates the collision detector's collision FK cache when the world state changes.
    """

    collision_detector: CollisionDetector = field(kw_only=True)
    """
    Reference to the collision detector that this updater belongs to.
    """

    def __post_init__(self):
        self._world = self.collision_detector._world
        super().__post_init__()

    def _notify(self, **kwargs):
        if self._world.is_empty():
            return
        self.collision_detector.world_model_updater.compiled_collision_fks.evaluate()
        self.collision_detector.sync_world_state()


@dataclass(eq=False)
class CollisionDetector(WorldEntityWithClassBasedID, abc.ABC):
    """
    Abstract class for collision detectors.
    """

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
    ) -> ClosestPoints | None:
        """
        Checks for collisions between two bodies.
        :param body_a: The first body to check for collisions.
        :param body_b: The second body to check for collisions.
        :param distance: The distance threshold for collision detection.
        :return: The closest points of contact if a collision is detected, otherwise None.
        """
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
