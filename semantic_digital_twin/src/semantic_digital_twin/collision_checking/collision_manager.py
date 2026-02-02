from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from .collision_detector import CollisionMatrix
from .collision_matrix import CollisionRule
from .collision_rules import (
    Updatable,
    AllowCollisionBetweenGroups,
    AllowCollisionForAdjacentPairs,
)
from ..callbacks.callback import ModelChangeCallback


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

    def __post_init__(self):
        super().__post_init__()
        self.high_priority_rules.extend(
            [AllowCollisionBetweenGroups(), AllowCollisionForAdjacentPairs()]
        )
        self._notify()

    def _notify(self):
        for rule in self.rules:
            if isinstance(rule, Updatable):
                rule.update(self.world)

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
