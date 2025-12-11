"""
Classes for Task-Achieving Body Motion Predicates.

This module defines the three predicates from the
Body Motion Problem. Different problem domains might need to overwrite an implementation:
"""

from typing import Optional
from dataclasses import dataclass, field

from giskardpy.executor import Executor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.predicate import Predicate
from ..robots.abstract_robot import AbstractRobot
from ..semantic_annotations.task_effect_motion import (
    TaskRequest,
    Effect,
    Motion,
    OpenedEffect,
    ClosedEffect,
)
from ..world import World


@dataclass
class Causes(Predicate):
    """
    A causes(Motion, Effect) predicate should check whether a given motion satisfies a given effect.
            Case1: causes(motion?, effect1) -> calculate a motion satisfying the desired effect.
            Case2: causes(motion1, effect?) -> execute motion and check if any known effect is satisfied that was not satisfied before.
            Case3: causes(motion?, effect?) -> Union of Case2 for all known motions and Case1 for all known effects.
    """

    effect: Effect

    environment: World

    motion: Optional[Motion]

    def __call__(self, *args, **kwargs):
        if self.effect.is_achieved():
            return False

        if self.motion and self.motion.motion_model:
            motion, success = self.motion.motion_model.run(
                self.effect, self.environment
            )
            if motion is not None:
                self.motion = motion
            return success
        return self._map_motion_to_effect()

    def _map_motion_to_effect(self):
        initial_state_data = self.environment.state.data.copy()
        trajectory = self.motion.trajectory
        actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        for position in trajectory:
            self.environment.set_positions_1DOF_connection({actuator: float(position)})

        is_achieved_post = self.effect.is_achieved()

        self.environment.state.data = initial_state_data
        self.environment.notify_state_change()

        return not is_achieved_pre and is_achieved_post


@dataclass
class CausesMotion(Causes):
    """
    Overwrites the Causes Predicate for case 1 where a motion needs to be calculated that satisfies a given effect.
    The calculated motion is written to the motion field.
    This is a special implementation for calculating motions from motion statecharts.
    """

    motion: Optional[Motion] = field(default=None, init=False)

    def __call__(self, *args, **kwargs):
        if self.effect.is_achieved():
            return False

        # If an execution model is provided on the effect, delegate to it.
        if self.effect.execution_model:
            motion, success = self.effect.execution_model.run(
                self.effect, self.environment
            )
            if motion is not None:
                self.motion = motion
            return success

        return False


@dataclass
class SatisfiesRequest(Predicate):
    """
    Check whether an effect satisfies a task request.
    For the sake of demonstration a placeholder mapping is used
    """

    task: TaskRequest
    effect: Effect

    def __call__(self, *args, **kwargs) -> bool:
        return self._effect_satisfies_task(self.task, self.effect)

    @staticmethod
    def _effect_satisfies_task(task: TaskRequest, effect: Effect) -> bool:
        # Placeholder logic
        task_type = (task.task_type or "").lower()
        if task_type == "open":
            return isinstance(effect, OpenedEffect)
        if task_type == "close":
            return isinstance(effect, ClosedEffect)
        return False


@dataclass
class CanExecute(Predicate):
    """
    Check whether a Robot can execute a motion.
    TODO: Implementation
    For the sake of demonstration the simplest way would be to place hte gripper of the robot at the handle
    and execute the Open motion statechart again. Repeat for all grippers.
    """

    motion: Motion
    robot: AbstractRobot

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
