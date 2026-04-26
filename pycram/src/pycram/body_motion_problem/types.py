"""
Core types for the Body Motion Problem (BMP) framework.

The BMP framework decomposes the success of a robot manipulation action into
three independently checkable properties: semantic correctness (does the outcome
match the task?), causal sufficiency (does the motion physically produce that
outcome?), and embodiment feasibility (can the robot execute the motion?).

This module defines the data structures shared across all BMP domains:

- ``Effect``       — a desired world-state change to verify against
- ``TaskRequest``  — a manipulation task with a semantic goal condition
- ``PhysicsModel`` — interface for simulating a motion against the world
- ``Motion``       — a candidate motion as a sequence of actuator positions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Connection,
)


@dataclass(eq=False, kw_only=True)
class Effect:
    """
    Represents a desired or achieved effect in the environment.

    An effect describes a measurable change to a property of a target object,
    such as the joint angle of a drawer (opening) or the fill level of a cup
    (pouring). It provides both the goal value and a way to read the current
    value, enabling the BMP predicates to check whether a motion achieved its
    intent.
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

    Use this for any domain where success means the value climbs to a threshold
    (e.g., fill level for pouring, joint angle for opening).
    """

    def is_achieved(self) -> bool:
        return self.current_value >= self.goal_value - self.tolerance


@dataclass(eq=False, kw_only=True)
class MonotoneDecreasingEffect(Effect):
    """
    Effect achieved when the property value falls at or below the goal.

    Use this for any domain where success means the value drops to a threshold
    (e.g., joint angle for closing, force releasing).
    """

    def is_achieved(self) -> bool:
        return self.current_value <= self.goal_value + self.tolerance


@dataclass(eq=True)
class TaskRequest:
    """
    Represents a manipulation task specification.

    Holds the task type, a name for the target, and a goal condition that
    determines which effects count as success. The goal condition is checked
    by the semantic correctness predicate to verify that a proposed outcome
    matches the task intent.
    """

    task_type: str
    """Task type identifier (e.g., 'open', 'close', 'pour', 'grasp')."""

    name: str
    """Name identifying the task or target object."""

    goal: Callable[[Effect], bool] = field(compare=False)
    """Predicate that checks whether an Effect satisfies this task."""


class PhysicsModel(ABC):
    """
    Abstract physics model used to simulate the causal effect of a motion.

    Implementations define how a motion trajectory changes the world state
    within a specific physical regime (e.g., rigid-body kinematics, fluid-flow
    dynamics). The causal sufficiency predicate uses this model to verify or
    generate motions.
    """

    @abstractmethod
    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate the motion and return the recorded actuator trajectory and whether
        the effect was achieved.

        :param effect: The desired effect to check against.
        :param world: The world to simulate in (state is reset after simulation).
        :return: (trajectory, achieved) where trajectory is the recorded actuator
                 positions and achieved indicates whether the effect was satisfied.
        """

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> List[Tuple[Connection, List[float]]]:
        """
        Return secondary actuator trajectories recorded by the most recent run().

        :param effect: The effect passed to the most recent run() call.
        :return: List of (connection, positions) pairs parallel to the primary trajectory.
        """
        return []


@dataclass(eq=True)
class Motion:
    """
    Represents a candidate motion as a sequence of actuator positions.

    Trajectories are expressed in actuator (joint) space. When a physics model
    is provided and the trajectory is empty, the model is used to generate the
    trajectory on demand during the causal sufficiency check.
    """

    trajectory: List[float]
    """Trajectory points in actuator space."""

    actuator: Connection
    """The connection (joint) that is manipulated by this motion."""

    motion_model: Optional[PhysicsModel] = field(default=None)
    """Optional physics model used to generate the trajectory when it is empty."""

    secondary_trajectories: List[Tuple[Connection, List[float]]] = field(
        default_factory=list
    )
    """
    Additional coupled actuator trajectories replayed alongside the primary.

    Each entry is ``(connection, positions)`` where ``positions`` is parallel
    to ``trajectory``.
    """

    dt: Optional[float] = field(default=None)
    """
    Time step between trajectory samples in seconds.

    When set, :meth:`~pycram.body_motion_problem.predicates.Causes` steps all
    ``HasUpdateState`` connections after each position update, allowing coupled
    physics (e.g. fill-level ODEs) to be inferred from the primary trajectory
    without explicit secondary trajectories.
    """
