from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable

from ..spatial_types import Point3
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
    TODO: can target_object and property_name automatically query for the current value?
    TODO: child classes, specialized effects, need to overwrite is_achieved(). Smth else that needs to be overwritten?
    """

    target_object: SemanticAnnotation
    """The object being affected."""

    # TODO: make property a link to some property of the target object
    property_getter: Callable[[SemanticAnnotation], float]
    """The property being changed (e.g., 'joint_angle', 'liquid_level')."""

    goal_value: float
    """Target value for the property."""

    tolerance: float = 0.05
    """Acceptable deviation from goal value."""

    def is_achieved(self) -> bool:
        """Check if the effect is achieved given the current property value."""
        return abs(self.current_value - self.goal_value) <= self.tolerance

    @property
    def current_value(self) -> float:
        return self.property_getter(self.target_object)


class OpenEffect(Effect):
    def is_achieved(self) -> bool:
        return self.current_value >= self.goal_value - self.tolerance


class ClosedEffect(Effect):
    def is_achieved(self) -> bool:
        return self.current_value <= self.goal_value + self.tolerance


@dataclass(eq=False)
class TaskRequest(SemanticAnnotation):
    """
    Represents a high-level task request.

    A task specifies what should be achieved, referencing the desired
    effect without specifying how to achieve it. If desired_effect is None,
    the task is generic (e.g., "open_anything") and matches any compatible effect.
    TODO: the None behaviour is not good.
    TODO: Needs to be aligned with pycram Task representation?
    TODO: the direct desired effect mapping makes queries from tasks to effects kinda useless.
            Can be nice for queries from effect to tasks.
    """

    task_type: str
    """Task type identifier (e.g., 'open', 'pour', 'grasp')."""

    desired_effect: Optional[Effect] = None
    """The effect this task aims to achieve (None for generic tasks)."""

    priority: float = 1.0
    """Task priority for scheduling (higher = more important)."""


@dataclass(eq=False)
class Motion(SemanticAnnotation):
    """
    Represents a planned motion trajectory.

    A motion describes a concrete plan for how to achieve an effect,
    including the trajectory, actuator, and expected outcome.
    TODO: is there a Motion representation in PyCram already?
            Giskard would refer to motion statecharts(MSC) instead of motions.
            But a MSC or controller could be viewed as an specific isntantiation of a motion
            Trajectory might be good because its generic to other methods than giskard.
            A canAchieve predicate can test if a specific MSC satisfies a given motion.
    TODO: actuator should maybe mean the body thats moved. Could be a handle but also a end effector
    TODO: expected effect mapping. Again makes some queries kinda useless.
            A causes(Motion, effect) predicate should check whether a given motion satisfies a given effect.
            Case1: causes(motion?, effect1) -> calculate an motion satisfying the desired effect.
            Case2: causes(motion1, effect?) -> execute motion and check if any known effect is satisfied that was not satisfied before.
            Case3: causes(motion?, effect?) -> Union of Case2 for all known motions and Case1 for all known effects.
    """

    trajectory: List[float]
    """Planned trajectory points in actuator space."""

    actuator: Connection
    """The connection that must be manipulated."""


"""
satisfiesTaskRequest(Task, Effect): Strongly depends on the Task Model used, i.e. provided by PyCram    
canAchieve(Motion, Robot): needs to transform a generic motion in the environment into an specific MSC for a specific Robot
                            only rarely a Robot will be given and a motion will be searched. Would need to return any motion (infinite) the robot an execute
                            would need additional filter like causes(Effect, Motion?), canAchieve(Motion?, Robot) to 
                            search a Motion of a specific Robot satisfying a specific effect.
                            Would that motion directly be a MSC? Can be in this framework, but a generic trajectory could also be nice for scientific reasons.
"""
