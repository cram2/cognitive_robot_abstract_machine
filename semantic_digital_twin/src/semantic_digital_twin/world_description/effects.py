"""
Core effect and task-request types for the Body Motion Problem framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass(eq=False, kw_only=True)
class Effect:
    """
    Represents a desired or achieved change to a property of a world object.

    Provides both the goal value and a way to read the current value, enabling
    BMP predicates to check whether a motion achieved its intent.
    """

    target_object: SemanticAnnotation
    """The object being affected."""

    property_getter: Callable[[SemanticAnnotation], float]
    """A callable that reads the relevant property from the target object."""

    goal_value: float
    """Target value for the property."""

    tolerance: float = 0.05
    """Acceptable deviation from goal value."""

    name: str = field(default="")
    """Display name for this effect."""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.__class__.__name__}({self.target_object.name})"

    def is_achieved(self) -> bool:
        """Check if the effect is achieved given the current property value."""
        return abs(self.current_value - self.goal_value) <= self.tolerance

    @property
    def current_value(self) -> float:
        return self.property_getter(self.target_object)


@dataclass(eq=False, kw_only=True)
class MonotoneIncreasingEffect(Effect):
    """
    Effect achieved when the property value reaches or exceeds the goal.
    """

    def is_achieved(self) -> bool:
        return self.current_value >= self.goal_value - self.tolerance


@dataclass(eq=False, kw_only=True)
class MonotoneDecreasingEffect(Effect):
    """
    Effect achieved when the property value falls at or below the goal.
    """

    def is_achieved(self) -> bool:
        return self.current_value <= self.goal_value + self.tolerance


@dataclass(eq=True)
class TaskRequest:
    """
    Represents a manipulation task specification with a semantic goal condition.
    """

    task_type: str
    """Task type identifier (e.g., 'open', 'close', 'pour', 'grasp')."""

    name: str
    """Name identifying the task or target object."""

    goal: Callable[[Effect], bool] = field(compare=False)
    """Predicate that checks whether an Effect satisfies this task."""
