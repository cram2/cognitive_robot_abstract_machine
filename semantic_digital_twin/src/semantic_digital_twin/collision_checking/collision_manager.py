from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rustworkx import rustworkx
from typing_extensions import List, TYPE_CHECKING

from .collision_detector import (
    CollisionMatrix,
    CollisionCheckingResult,
    CollisionDetector,
)
from .collision_matrix import (
    CollisionRule,
    MaxAvoidedCollisionsRule,
    DefaultMaxAvoidedCollisions,
)
from .collision_rules import (
    Updatable,
    AllowCollisionForAdjacentPairs,
    AllowNonRobotCollisions,
    AvoidCollisionRule,
)
from ..callbacks.callback import ModelChangeCallback
from ..world_description.world_entity import Body, KinematicStructureEntity

if TYPE_CHECKING:
    from ..world import World


@dataclass(repr=False, eq=False)
class CollisionGroup:
    """
    Bodies in this group are viewed as a single body.
    """

    root: KinematicStructureEntity
    bodies: set[Body] = field(default_factory=set)

    def __repr__(self) -> str:
        return f"CollisionGroup(root={self.root.name}, bodies={[b.name for b in self.bodies]})"

    def __eq__(self, other) -> bool:
        return self.root == other.root

    def __contains__(self, item):
        return item == self.root or item in self.bodies

    def __hash__(self):
        return hash((self.root, tuple(sorted(self.bodies))))

    def get_max_avoided_bodies(self, collision_manager: CollisionManager):
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
class CollisionConsumer(ABC):
    collision_manager: CollisionManager = field(init=False)

    @abstractmethod
    def on_reset(self):
        """
        Called when the collision matrix changes.
        """

    @abstractmethod
    def on_compute_collisions(self, collision_results: CollisionCheckingResult):
        """
        Called when collision checking is finished.
        :param collision_results:
        """

    @abstractmethod
    def on_world_model_update(self, world: World): ...

    @abstractmethod
    def on_collision_matrix_update(self): ...


@dataclass
class CollisionGroupConsumer(CollisionConsumer, ABC):
    collision_groups: list[CollisionGroup] = field(default_factory=list, init=False)

    def on_world_model_update(self, world: World):
        self.update_collision_groups(world)

    def update_collision_groups(self, world: World):
        self.collision_groups = [CollisionGroup(world.root)]
        for parent, childs in rustworkx.bfs_successors(
            world.kinematic_structure, world.root.index
        ):
            collision_group = self.get_collision_group(parent)
            for child in childs:
                parent_C_child = world.get_connection(parent, child)
                if parent_C_child.is_controlled:
                    self.collision_groups.append(CollisionGroup(child))
                else:
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
        for group in self.collision_groups:
            if body in group.bodies or body == group.root:
                return group
        raise Exception(f"No collision group found for {body}")


@dataclass
class CollisionManager(ModelChangeCallback):
    """
    Manages collision rules and turn them into collision matrices.
    1. apply default rules
    2. apply temporary rules
    3. apply ignore-collision rules
        this is usually allow collisions, like the self collision matrix
    """

    collision_checker: CollisionDetector
    collision_matrix: CollisionMatrix = field(init=False)

    default_rules: List[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix before temporary rules.
    Any other rules will overwrite these.
    """
    temporary_rules: List[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix after default rules.
    These are intended for task specific rules.
    """
    ignore_collision_rules: List[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix to ignore collisions.
    The permanently allow collisions and cannot be overwritten by other rules.
    """

    max_avoided_bodies_rules: List[MaxAvoidedCollisionsRule] = field(
        default_factory=lambda: [DefaultMaxAvoidedCollisions()]
    )

    collision_consumers: list[CollisionConsumer] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.ignore_collision_rules.extend(
            [AllowNonRobotCollisions(), AllowCollisionForAdjacentPairs()]
        )
        self._notify()

    def _notify(self):
        if self.world.is_empty():
            return
        for rule in self.rules:
            if isinstance(rule, Updatable):
                rule.update(self.world)
        for consumer in self.collision_consumers:
            consumer.on_world_model_update(self.world)

    def reset_temporary_rules(self):
        self.temporary_rules = []

    def add_collision_consumer(self, consumer: CollisionConsumer):
        self.collision_consumers.append(consumer)
        consumer.collision_manager = self
        consumer.on_world_model_update(self.world)

    def reset_consumers(self):
        for consumer in self.collision_consumers:
            consumer.on_reset()

    def update_collision_matrix(self, buffer: float = 0.05):
        self.collision_matrix = CollisionMatrix()
        for rule in self.default_rules:
            rule.apply_to_collision_matrix(self.collision_matrix)
        for rule in self.temporary_rules:
            rule.apply_to_collision_matrix(self.collision_matrix)
        for rule in self.ignore_collision_rules:
            rule.apply_to_collision_matrix(self.collision_matrix)
        for consumer in self.collision_consumers:
            consumer.on_collision_matrix_update()
        if buffer is not None:
            self.collision_matrix.apply_buffer(buffer)

    def compute_collisions(self, buffer: float = 0.05) -> CollisionCheckingResult:
        self.update_collision_matrix(buffer)
        collision_results = self.collision_checker.check_collisions(
            self.collision_matrix
        )
        for consumer in self.collision_consumers:
            consumer.on_compute_collisions(collision_results)
        return collision_results

    def get_max_avoided_bodies(self, body: Body) -> int:
        for rule in reversed(self.max_avoided_bodies_rules):
            max_avoided_bodies = rule.get_max_avoided_collisions(body)
            if max_avoided_bodies is not None:
                return max_avoided_bodies
        raise Exception(f"No rule found for {body}")

    def get_buffer_zone_distance(self, body_a: Body, body_b: Body) -> float:
        """
        Returns the buffer-zone distance for the body by scanning rules from highest to lowest priority.
        """
        for rule in reversed(self.rules):
            if isinstance(rule, AvoidCollisionRule):
                value = rule.buffer_zone_distance_for(body_a, body_b)
                if value is not None:
                    return value
        raise ValueError(f"No buffer-zone rule found for {body_a, body_b}")

    def get_violated_distance(self, body_a: Body, body_b: Body) -> float:
        """
        Returns the violated distance for the body by scanning rules from highest to lowest priority.
        """
        for rule in reversed(self.rules):
            if isinstance(rule, AvoidCollisionRule):
                value = rule.violated_distance_for(body_a, body_b)
                if value is not None:
                    return value
        raise ValueError(f"No violated-distance rule found for {body_a, body_b}")

    @property
    def rules(self) -> List[CollisionRule]:
        return self.default_rules + self.temporary_rules + self.ignore_collision_rules
