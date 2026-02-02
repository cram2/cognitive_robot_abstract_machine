from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from .collision_detector import CollisionMatrix
from .collision_matrix import (
    CollisionRule,
    MaxAvoidedCollisionsRule,
    DefaultMaxAvoidedCollisions,
)
from .collision_rules import (
    Updatable,
    AllowCollisionBetweenGroups,
    AllowCollisionForAdjacentPairs,
    AllowNonRobotCollisions,
)
from ..callbacks.callback import ModelChangeCallback
from ..world_description.world_entity import Body


@dataclass
class CollisionManager(ModelChangeCallback):
    """
    Manages collision rules and turn them into collision matrices.
    1. apply default rules
    2. apply temporary rules
    3. apply final rules
        this is usually allow collisions, like the self collision matrix
    """

    low_priority_rules: List[CollisionRule] = field(default_factory=list)
    normal_priority_rules: List[CollisionRule] = field(default_factory=list)
    high_priority_rules: List[CollisionRule] = field(default_factory=list)

    max_avoided_bodies_rules: List[MaxAvoidedCollisionsRule] = field(
        default_factory=lambda: [DefaultMaxAvoidedCollisions()]
    )

    def __post_init__(self):
        super().__post_init__()
        self.high_priority_rules.extend(
            [AllowNonRobotCollisions(), AllowCollisionForAdjacentPairs()]
        )
        self._notify()

    def _notify(self):
        for rule in self.rules:
            if isinstance(rule, Updatable):
                rule.update(self.world)

    def get_max_avoided_bodies(self, body: Body) -> int:
        for rule in reversed(self.max_avoided_bodies_rules):
            max_avoided_bodies = rule.get_max_avoided_collisions(body)
            if max_avoided_bodies is not None:
                return max_avoided_bodies
        raise Exception(f"No rule found for {body}")

    @property
    def rules(self) -> List[CollisionRule]:
        return (
            self.low_priority_rules
            + self.normal_priority_rules
            + self.high_priority_rules
        )

    def create_collision_matrix(self) -> CollisionMatrix:
        collision_matrix = CollisionMatrix()
        for rule in self.low_priority_rules:
            rule.apply_to_collision_matrix(collision_matrix)
        for rule in self.normal_priority_rules:
            rule.apply_to_collision_matrix(collision_matrix)
        for rule in self.high_priority_rules:
            rule.apply_to_collision_matrix(collision_matrix)
        return collision_matrix
