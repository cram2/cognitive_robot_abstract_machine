"""
Core types for the Body Motion Problem (BMP) framework.

Defines the data structures corresponding to the formal BMP entities:
  - PhysicsModel  — the scoped physics Φ
  - Effect        — a predicate over G_final (final SDT state)
  - TaskRequest   — a manipulation task specification Π
  - Motion        — a candidate trajectory τ
  - TEEClass      — a Task-Environment-Embodiment scope D = ⟨T, E, R, Φ, I_Φ⟩
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Connection,
)


@dataclass(eq=False, kw_only=True)
class Effect:
    """
    Represents a desired or achieved effect in the environment.

    Corresponds to a predicate over the final SDT state G_final.
    An effect describes a change to a property of a target object,
    such as opening a door (changing joint angle) or pouring liquid
    (changing liquid level).
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
    Represents a manipulation task specification Π.

    Corresponds to Π_goal in the BMP formalism — a set of semantically valid
    trajectories expressed as a high-level task type and a target name.
    """

    task_type: str
    """Task type identifier (e.g., 'open', 'close', 'pour', 'grasp')."""

    name: str
    """Name identifying the task or target object."""


class PhysicsModel(ABC):
    """
    Abstract physics model Φ used to simulate the causal effect of a motion.

    Corresponds to Φ_sim in Causes(τ, G_final, Φ, I_Φ): the model that
    determines whether executing trajectory τ produces final state G_final.
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

        Override in physics models that drive more than one DOF alongside the
        primary actuator (e.g. fill-level joint in pouring). The base returns
        an empty list, meaning only the primary trajectory is replayed.

        :param effect: The effect passed to the most recent run() call.
        :return: List of (connection, positions) pairs parallel to the primary trajectory.
        """
        return []


class OutOfScopeError(Exception):
    """Raised by Causes when world parameters fall outside I_Φ."""


@dataclass(eq=True)
class Motion:
    """
    Represents a candidate motion trajectory τ over an actuator.

    Trajectories are expressed in actuator (joint) space and optionally
    backed by a physics model that can generate them from scratch.
    """

    trajectory: List[float]
    """Trajectory points in actuator space."""

    actuator: Connection
    """The connection (joint) that is manipulated by this motion."""

    motion_model: Optional[PhysicsModel] = field(default=None)
    """
    Optional physics model Φ used to generate the trajectory if it is empty.
    """

    secondary_trajectories: List[Tuple[Connection, List[float]]] = field(
        default_factory=list
    )
    """
    Additional coupled actuator trajectories replayed alongside the primary.

    Each entry is ``(connection, positions)`` where ``positions`` is parallel
    to ``trajectory``. Use this for domains where the effect depends on more
    than one DOF (e.g., tilt joint + fill-level joint in pouring).
    """


@dataclass
class TEEClass:
    """
    Task-Environment-Embodiment scope. D = ⟨T, E, R, Φ, I_Φ⟩.

    Scopes the BMP by specifying which task types, environments, embodiments,
    physics model, and validity intervals are considered together.
    """

    task_types: frozenset
    """T: the set of task type identifiers handled by this TEE class."""

    physics_model: Optional[PhysicsModel] = field(default=None)
    """Φ: the governing physics model for causal sufficiency checks."""

    validity_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    """I_Φ: validity intervals for physics parameters (param_name → (lower, upper))."""
