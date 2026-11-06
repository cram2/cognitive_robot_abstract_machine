from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any, Self


from krrood.adapters.json_serializer import to_json, from_json
from krrood.patterns.caching import memoize, clear_memoization_cache
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from krrood.patterns.field_metadata import JSONMetadata
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionMatrix,
    CollisionCheckingResult,
    CollisionDetector,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionRule,
    MaxAvoidedCollisionsRule,
    DefaultMaxAvoidedCollisions,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenEndEffectorsAndHeldBodies,
    AllowCollisionForAdjacentPairs,
    AllowNonRobotCollisions,
    AvoidCollisionRule,
    AllowCollisionRule,
)
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    BulletCollisionDetector,
)
from semantic_digital_twin.exceptions import TemporaryCollisionRulesDependOnOrder
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from ..world import World


@dataclass
class CollisionConsumer(ABC):
    """
    Interface for classes that want to be notified about changes in the collision matrix
    or when collision checking is performed.

    These classes are used for postprocessing collision checking results for specific
    purposes, like external/self collision avoidance tasks in giskard.
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

    It manages collision rules, owns the collision checker, and manages collision consumers using
    an observer pattern. This class is a world model callback and will update the
    collision detector's scene and collision matrix on world model changes.

    Collision matrices are updated using rules in the following order:
    1. apply default rules
    2. apply temporary rules
    3. apply ignore-collision rules
        this is usually allow collisions, like the self collision matrix
    Within these lists, rules that are later in the list overwrite rules that are earlier in the list.
    """

    collision_detector: CollisionDetector = field(kw_only=True)
    """
    The collision detector implementation used for computing closest points between
    bodies.
    """

    collision_matrix: CollisionMatrix = field(init=False, repr=False)
    """
    The collision matrix describing for which body pairs the collision detector should
    check for closest points.
    """

    default_rules: list[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix before temporary rules.

    They are intended for the most general rules, like default distance thresholds. Any
    other rules will overwrite these. .. note: These rules ARE synced with other worlds.
    """

    temporary_rules: list[CollisionRule] = field(default_factory=list)
    """
    Rules that are applied to the collision matrix after default rules.

    These are intended for task specific rules. .. note: These rules are NOT synced with
    other worlds.
    """

    ignore_collision_rules: list[AllowCollisionRule] = field(
        default_factory=lambda: [
            AllowCollisionForAdjacentPairs(),
            AllowNonRobotCollisions(),
            AllowCollisionBetweenEndEffectorsAndHeldBodies(),
        ]
    )
    """
    Rules that are applied to the collision matrix to ignore collisions.

    The permanently allow collisions and cannot be overwritten by other rules.

    By default we allow collisions between non-robot bodies, between adjacent bodies,
    and between end effectors and the bodies they hold.

    .. note: This is only meant for collision that should NEVER be checked.
        Allow collision rules can also be added to default or temporary rules if needed.
    .. note: These rules ARE synced with other worlds.
    """

    max_avoided_bodies_rules: list[MaxAvoidedCollisionsRule] = field(
        default_factory=lambda: [DefaultMaxAvoidedCollisions()]
    )
    """
    Rules that determine the maximum number of collisions considered for avoidance tasks
    between two bodies.
    """

    collision_consumers: list[CollisionConsumer] = field(
        default_factory=list, metadata=JSONMetadata(serialize=False).as_dict()
    )
    """
    Objects that are notified about changes in the collision matrix.
    """

    def __post_init__(self):
        super().__post_init__()
        self.on_model_change()

    def on_model_change(self, **kwargs):
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
    def extend_default_rules(self, rules: list[CollisionRule]):
        self.default_rules.extend(rules)

    @synchronized_attribute_modification
    def add_ignore_collision_rule(self, rule: AllowCollisionRule):
        self.ignore_collision_rules.append(rule)

    def add_temporary_rule(self, rule: CollisionRule):
        """
        Adds a rule to the temporary collision rules.
        """
        self.temporary_rules.append(rule)

    def extend_temporary_rule(self, rules: list[CollisionRule]):
        """
        Adds a list rule to the temporary collision rules.
        """
        self.temporary_rules.extend(rules)

    def clear_temporary_rules(self):
        """
        Call this before starting a new task.
        """
        self.temporary_rules.clear()

    def put_temporary_rules_in_force(self, rules: list[CollisionRule]) -> None:
        """
        Add `rules` to the temporary rules and update the collision matrix.

        :param rules: The rules to apply after the temporary rules already in force.
        :raises TemporaryCollisionRulesDependOnOrder: If applying `rules` before the
            temporary rules already in force would give another collision matrix.
        """
        if self.temporary_rules and (
            self._checked_distances([*self.temporary_rules, *rules])
            != self._checked_distances([*rules, *self.temporary_rules])
        ):
            raise TemporaryCollisionRulesDependOnOrder(
                rules_in_force=list(self.temporary_rules), added_rules=rules
            )
        self.temporary_rules.extend(rules)
        self.update_collision_matrix()

    def take_temporary_rules_out_of_force(self, rules: list[CollisionRule]) -> None:
        """
        Remove exactly the rule objects in `rules` from the temporary rules and update
        the collision matrix.

        :param rules: The rules to remove; rules that are not in force are skipped.
        """
        self.temporary_rules[:] = [
            rule_in_force
            for rule_in_force in self.temporary_rules
            if not any(rule_in_force is rule for rule in rules)
        ]
        self.update_collision_matrix()

    def _checked_distances(
        self, temporary_rules: list[CollisionRule]
    ) -> dict[tuple[Body, Body], float | None]:
        """
        :param temporary_rules: The temporary rules to apply, in this order.
        :return: The distance every body pair would be checked at with
            `temporary_rules` in force.
        """
        return {
            check.bodies(): check.distance
            for check in self._collision_matrix_of(temporary_rules).collision_checks
        }

    def _collision_matrix_of(
        self, temporary_rules: list[CollisionRule]
    ) -> CollisionMatrix:
        """
        :param temporary_rules: The temporary rules to apply, in this order.
        :return: The collision matrix of the default rules, `temporary_rules` and the
            ignore-collision rules, without a buffer.
        """
        collision_matrix = CollisionMatrix()
        for rule in [
            *self.default_rules,
            *temporary_rules,
            *self.ignore_collision_rules,
        ]:
            rule.update(self._world)
            rule.apply_to_collision_matrix(collision_matrix)
        return collision_matrix

    @synchronized_attribute_modification
    def extend_max_avoided_bodies_rules(self, rules: list[MaxAvoidedCollisionsRule]):
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

    def remove_collision_consumer(self, consumer: CollisionConsumer):
        """
        Removes a collision consumer from the list of consumers.
        """
        self.collision_consumers.remove(consumer)

    def update_collision_matrix(self, buffer: float = 0.05):
        """
        Creates a new collision matrix based on the current rules and applies it to the
        collision detector.

        .. note:: This method is not called in `compute_collisions` because it is potentially expensive
            and you quite often want to compute collisions without updating the collision matrix.
        :param buffer: A buffer is added to the collision matrix distance thresholds.
            This is useful when you want to react to collisions before they go below the threshold.
        """
        self.collision_matrix = self._collision_matrix_of(self.temporary_rules)
        for consumer in self.collision_consumers:
            consumer.on_collision_matrix_update()
        if buffer is not None:
            self.collision_matrix.apply_buffer(buffer)
        clear_memoization_cache(self)

    def set_collision_matrix(self, collision_matrix: CollisionMatrix):
        """
        Sets the collision matrix directly and clears caches.

        .. warning: if the collision matrix was computed with a different world model version, you may get unexpected results.
        :param collision_matrix: New collision matrix.
        """
        self.collision_matrix = collision_matrix
        clear_memoization_cache(self)

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

    @memoize
    def get_buffer_zone_distance(self, body_a: Body, body_b: Body) -> float:
        """
        Returns the buffer-zone distance for the body pair by scanning rules from
        highest to lowest priority.
        """
        for rule in reversed(self.rules):
            if not isinstance(rule, AvoidCollisionRule):
                continue
            value = rule.buffer_zone_distance_for(body_a, body_b)
            if value is not None:
                return value
        raise ValueError(f"No buffer-zone rule found for {body_a, body_b}")

    @memoize
    def get_violated_distance(self, body_a: Body, body_b: Body) -> float:
        """
        Returns the violated distance for the body pair by scanning rules from highest
        to lowest priority.
        """
        for rule in reversed(self.rules):
            if not isinstance(rule, AvoidCollisionRule):
                continue
            value = rule.violated_distance_for(body_a, body_b)
            if value is not None:
                return value
        raise ValueError(f"No violated-distance rule found for {body_a, body_b}")

    @property
    def rules(self) -> list[CollisionRule]:
        """
        :return: all rules in the order they are applied.
        """
        return self.default_rules + self.temporary_rules + self.ignore_collision_rules

    def to_json(self, **kwargs) -> dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            "id": to_json(self.id, **kwargs),
            "default_rules": to_json(self.default_rules, **kwargs),
            "temporary_rules": to_json(self.temporary_rules, **kwargs),
            "ignore_collision_rules": to_json(self.ignore_collision_rules, **kwargs),
            "max_avoided_bodies_rules": to_json(
                self.max_avoided_bodies_rules, **kwargs
            ),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            collision_detector=BulletCollisionDetector(),
            default_rules=from_json(data["default_rules"], **kwargs),
            temporary_rules=from_json(data["temporary_rules"], **kwargs),
            ignore_collision_rules=from_json(data["ignore_collision_rules"], **kwargs),
            max_avoided_bodies_rules=from_json(
                data["max_avoided_bodies_rules"], **kwargs
            ),
        )
