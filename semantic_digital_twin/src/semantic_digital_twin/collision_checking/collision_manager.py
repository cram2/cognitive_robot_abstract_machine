from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Any, Self

from typing_extensions import List, TYPE_CHECKING

from krrood.adapters.json_serializer import to_json, from_json
from .collision_detector import (
    CollisionMatrix,
    CollisionCheckingResult,
    CollisionDetector,
)
from .collision_matrix import (
    CollisionRule,
    MaxAvoidedCollisionsRule,
    DefaultMaxAvoidedCollisions,
    CollisionCheck,
)
from .collision_rules import (
    AllowCollisionForAdjacentPairs,
    AllowNonRobotCollisions,
    AvoidCollisionRule,
    AllowCollisionRule,
)
from .pybullet_collision_detector import BulletCollisionDetector
from ..callbacks.callback import ModelChangeCallback
from ..world_description.world_entity import Body
from ..world_description.world_modification import synchronized_attribute_modification

if TYPE_CHECKING:
    from ..world import World


@dataclass
class CollisionConsumer(ABC):
    """
    Interface for classes that want to be notified about changes in the collision matrix or when collision checking is performed.
    These classes are used for postprocessing collision checking results for specific purposes, like external/self collision avoidance tasks in giskard.
    """

    collision_manager: CollisionManager = field(init=False)
    """
    Backreference to the collision manager that owns this consumer.
    """

    @abstractmethod
    def on_compute_collisions(self, collision_results: CollisionCheckingResult):
        """
        Called when collision checking is finished.
        :param collision_results:
        """

    @abstractmethod
    def on_world_model_update(self, world: World):
        """
        Called when the world model changes.
        :param world: Reference to the updated world.
        """

    @abstractmethod
    def on_collision_matrix_update(self):
        """
        Called when the collision matrix is updated.
        """


@dataclass(eq=False)
class CollisionManager(ModelChangeCallback):
    """
    This class is intended as the primary interface for collision checking.
    It manages collision rules, owns the collision checker, and manages collision consumers using an observer pattern.
    This class is a world model callback and will update the collision detector's scene and collision matrix on world model changes.

    Collision matrices are updated using rules in the following order:
    1. apply default rules
    2. apply temporary rules
    3. apply ignore-collision rules
        this is usually allow collisions, like the self collision matrix
    Within these lists, rules that are later in the list overwrite rules that are earlier in the list.
    """

    collision_detector: CollisionDetector = field(kw_only=True)
    """
    The collision detector implementation used for computing closest points between bodies.
    """

    collision_matrix: CollisionMatrix = field(init=False, repr=False)
    """
    The collision matrix describing for which body pairs the collision detector should check for closest points.
    """

    default_rules: List[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix before temporary rules.
    They are intended for the most general rules, like default distance thresholds.
    Any other rules will overwrite these.
    """
    temporary_rules: List[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix after default rules.
    These are intended for task specific rules.
    """
    ignore_collision_rules: List[AllowCollisionRule] = field(
        default_factory=lambda: [
            AllowCollisionForAdjacentPairs(),
            AllowNonRobotCollisions(),
        ]
    )
    """
    Rules that are applied to the collision matrix to ignore collisions.
    The permanently allow collisions and cannot be overwritten by other rules.
    
    By default we allow collisions between non-robot bodies and between adjacent bodies.
    """

    max_avoided_bodies_rules: List[MaxAvoidedCollisionsRule] = field(
        default_factory=lambda: [DefaultMaxAvoidedCollisions()]
    )
    """
    Rules that determine the maximum number of collisions considered for avoidance tasks between two bodies.
    """

    collision_consumers: list[CollisionConsumer] = field(default_factory=list)
    """
    Objects that are notified about changes in the collision matrix.
    """

    def __hash__(self):
        return hash(id(self))

    def __post_init__(self):
        super().__post_init__()
        self._notify()

    def _notify(self, **kwargs):
        if self._world.is_empty():
            return
        for consumer in self.collision_consumers:
            consumer.on_world_model_update(self._world)

    def has_consumers(self) -> bool:
        return len(self.collision_consumers) > 0

    @synchronized_attribute_modification
    def add_default_rule(self, rule: CollisionRule):
        self.default_rules.append(rule)

    @synchronized_attribute_modification
    def extend_default_rules(self, rules: List[CollisionRule]):
        self.default_rules.extend(rules)

    @synchronized_attribute_modification
    def add_ignore_collision_rule(self, rule: AllowCollisionRule):
        self.ignore_collision_rules.append(rule)

    @synchronized_attribute_modification
    def add_temporary_rule(self, rule: CollisionRule):
        """
        Adds a rule to the temporary collision rules.
        """
        self.temporary_rules.append(rule)

    @synchronized_attribute_modification
    def clear_temporary_rules(self):
        """
        Call this before starting a new task.
        """
        self.temporary_rules.clear()

    @synchronized_attribute_modification
    def extend_max_avoided_bodies_rules(self, rules: List[MaxAvoidedCollisionsRule]):
        self.max_avoided_bodies_rules.extend(rules)

    def add_collision_consumer(self, consumer: CollisionConsumer):
        """
        Adds a collision consumer to the list of consumers.
        It will be notified when:
        - when the collision matrix is updated
        - with the world, when its model updates
        - with the results of `compute_collisions` when it is called.
        """
        self.collision_consumers.append(consumer)
        consumer.collision_manager = self
        consumer.on_world_model_update(self._world)

    def update_collision_matrix(self, buffer: float = 0.05):
        """
        Creates a new collision matrix based on the current rules and applies it to the collision detector.
        .. note:: This method is not called in `compute_collisions` because it is potentially expensive
            and you quite often want to compute collisions without updating the collision matrix.
        :param buffer: A buffer is added to the collision matrix distance thresholds.
            This is useful when you want to react to collisions before they go below the threshold.
        """
        for rule in self.rules:
            rule.update(self._world)
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
        self.get_buffer_zone_distance.cache_clear()
        self.get_violated_distance.cache_clear()

    def is_collision_checked(self, body_a: Body, body_b: Body) -> bool:
        """
        Checks if the collision matrix contains a collision check for the given bodies.
        """
        return (
            CollisionCheck(body_a, body_b) in self.collision_matrix.collision_checks
            or CollisionCheck(body_b, body_a) in self.collision_matrix.collision_checks
        )

    def compute_collisions(self) -> CollisionCheckingResult:
        """
        Computes collisions based on the current collision matrix.
        .. note:: You may want to call `update_collision_matrix` before calling this method if rules or the world model have changed.
        :return: Result of the collision checking.
        """
        collision_results = self.collision_detector.check_collisions(
            self.collision_matrix
        )
        for consumer in self.collision_consumers:
            consumer.on_compute_collisions(collision_results)
        return collision_results

    def get_max_avoided_bodies(self, body: Body) -> int:
        """
        Returns the maximum number of collisions `body` should avoid.
        :param body: The body to check.
        :return: Maximum number of collisions that are allowed between two bodies.
        """
        for rule in reversed(self.max_avoided_bodies_rules):
            max_avoided_bodies = rule.get_max_avoided_collisions(body)
            if max_avoided_bodies is not None:
                return max_avoided_bodies
        raise Exception(f"No rule found for {body}")

    @lru_cache
    def get_buffer_zone_distance(self, body_a: Body, body_b: Body) -> float:
        """
        Returns the buffer-zone distance for the body pair by scanning rules from highest to lowest priority.
        """
        for rule in reversed(self.rules):
            if isinstance(rule, AvoidCollisionRule):
                value = rule.buffer_zone_distance_for(body_a, body_b)
                if value is not None:
                    return value
        raise ValueError(f"No buffer-zone rule found for {body_a, body_b}")

    @lru_cache
    def get_violated_distance(self, body_a: Body, body_b: Body) -> float:
        """
        Returns the violated distance for the body pair by scanning rules from highest to lowest priority.
        """
        for rule in reversed(self.rules):
            if isinstance(rule, AvoidCollisionRule):
                value = rule.violated_distance_for(body_a, body_b)
                if value is not None:
                    return value
        raise ValueError(f"No violated-distance rule found for {body_a, body_b}")

    @property
    def rules(self) -> List[CollisionRule]:
        """
        :return: all rules in the order they are applied.
        """
        return self.default_rules + self.temporary_rules + self.ignore_collision_rules

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": to_json(self.id),
            "default_rules": to_json(self.default_rules),
            "temporary_rules": to_json(self.temporary_rules),
            "ignore_collision_rules": to_json(self.ignore_collision_rules),
            "max_avoided_bodies_rules": to_json(self.max_avoided_bodies_rules),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            collision_detector=BulletCollisionDetector(),
            default_rules=from_json(data["default_rules"], **kwargs),
            temporary_rules=from_json(data["temporary_rules"], **kwargs),
            ignore_collision_rules=from_json(data["ignore_collision_rules"], **kwargs),
            max_avoided_bodies_rules=from_json(
                data["max_avoided_bodies_rules"], **kwargs
            ),
        )
