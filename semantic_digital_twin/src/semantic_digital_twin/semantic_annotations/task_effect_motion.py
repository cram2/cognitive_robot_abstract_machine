from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any

from giskardpy.executor import Executor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from ..spatial_types import Point3
from ..world import World
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    Connection,
)


@dataclass(eq=False)
class Effect(SemanticAnnotation):
    """
    Represents a desired or achieved effect in the environment.

    An effect describes a change to a property of a target object,
    such as opening a door (changing joint angle) or pouring liquid
    (changing liquid level).
    """

    target_object: SemanticAnnotation
    """The object being affected."""

    property_getter: Callable[[SemanticAnnotation], float]
    """The property being changed (e.g., 'joint_angle', 'liquid_level')."""

    goal_value: float
    """Target value for the property."""

    execution_model: EffectExecutionModel
    """Model defining an execution function for the effect."""

    tolerance: float = 0.05
    """Acceptable deviation from goal value."""

    def is_achieved(self) -> bool:
        """Check if the effect is achieved given the current property value."""
        return abs(self.current_value - self.goal_value) <= self.tolerance

    @property
    def current_value(self) -> float:
        return self.property_getter(self.target_object)


class OpenedEffect(Effect):
    def is_achieved(self) -> bool:
        return self.current_value >= self.goal_value - self.tolerance


class ClosedEffect(Effect):
    def is_achieved(self) -> bool:
        return self.current_value <= self.goal_value + self.tolerance


@dataclass(eq=False)
class TaskRequest(SemanticAnnotation):
    """
    Represents a high-level task request.
    Good definition might something like 'Fill me a glas with water' which needs to be mapped to the specific effect
    Effect(target_object=water_glass, property_getter=water_glass.fill_level, goal_value=80%, tolerance=5%)
    TODO: Needs to be aligned with pycram Task representation?
    """

    task_type: str
    """Task type identifier (e.g., 'open', 'pour', 'grasp')."""


@dataclass(eq=False)
class Motion(SemanticAnnotation):
    """
    Represents a planned motion trajectory.

    A motion describes the trajectory of an actuator/ActiveConnection
    TODO: Alternative Motion definitions
        1. Generalize to a WorldState trajectory
        2. Use Trajectories on Bodies, like a handle, instead of the movable joint for that handle
        3. Use a MotionStatechart representation
        4. Are options 1 and 2 directly interchangeable? also with option 3?
    """

    trajectory: List[float]
    """Planned trajectory points in actuator space."""

    actuator: Connection
    """The connection that must be manipulated."""

    motion_model: RunMSCModel = field(default=None)


class MissingMotionStatechartError(Exception):
    """
    Raised when a model is asked to run without a configured MotionStatechart.
    """


class EffectExecutionModel(ABC):
    def run(self, effect: Effect, world: World):
        pass


@dataclass
class RunMSCModel(EffectExecutionModel):
    """
    Execute an already-constructed (but uncompiled) MotionStatechart against a World.

    The MotionStatechart must be fully parameterized (nodes and edges added) before being passed
    into this model. This model only binds it to the provided World via Executor.compile(...),
    rolls it out introspectively (recording the Effect's observed value after each tick), and resets
    the World's state before returning.
    """

    msc: MotionStatechart
    actuator: Connection
    timeout: int = 500

    def run(self, effect: Effect, world: World) -> Tuple[Optional[Motion], bool]:
        if self.msc is None:
            raise MissingMotionStatechartError(
                "RunMSCModel requires a MotionStatechart instance to run."
            )

        executor = Executor(world=world)
        executor.compile(motion_statechart=self.msc)

        # Introspective rollout: mutate world during rollout but always reset before returning
        initial_state_data = world.state.data.copy()
        try:
            trajectory: List[float] = []
            for _ in range(self.timeout):
                executor.tick()
                trajectory.append(effect.current_value)
                if self.msc.is_end_motion():
                    break

            motion = Motion(trajectory=trajectory, actuator=self.actuator)
            return motion, effect.is_achieved()
        finally:
            world.state.data = initial_state_data
            world.notify_state_change()
