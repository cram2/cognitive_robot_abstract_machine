"""
Abstract Base Class for Task-Achieving Body Motion Predicates.

This module defines the interface for implementing the three predicates from the
Law of Task-Achieving Body Motion paper. Different problem domains can provide
their own implementations:

- Motion planning: Execute motion statecharts to observe effects
- Fluid simulation: Use probabilistic models for pouring tasks
- Language/AI: Use LLM or learned models to reason about task-effect relationships
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field

from giskardpy.executor import Executor
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.predicate import Predicate
from ..robots.abstract_robot import AbstractRobot
from ..semantic_annotations.semantic_annotations import Drawer, Fridge, Door
from ..semantic_annotations.task_effect_motion import (
    TaskRequest,
    Effect,
    Motion,
    OpenEffect,
    ClosedEffect,
)
from ..world import World
from ..world_description.connections import PrismaticConnection, RevoluteConnection
from ..world_description.world_entity import SemanticAnnotation


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
class CausesOpening(Causes):

    motion: Optional[Motion] = field(default=None, init=False)

    def __call__(self, *args, **kwargs):
        if self.effect.is_achieved():
            return False

        initial_state_data = self.environment.state.data.copy()
        executor = Executor(world=self.environment)

        handle, joint = self._extract_container_info(self.effect.target_object)

        open_goal = Open(
            tip_link=handle.body,
            environment_link=handle.body,
            goal_joint_state=self.effect.goal_value,
        )

        msc = MotionStatechart()
        msc.add_node(open_goal)
        msc.add_node(EndMotion.when_true(open_goal))

        executor.compile(motion_statechart=msc)

        trajectory = self._execute_and_record_trajectory(executor, msc)

        is_achieved = self.effect.is_achieved()

        # Reset state
        self.environment.state.data = initial_state_data
        self.environment.notify_state_change()

        self.motion = Motion(trajectory=trajectory, actuator=joint)

        return is_achieved

    def _extract_container_info(self, annotation: SemanticAnnotation):
        """
        Extracts body, handle, and joint info from a semantic annotation.
        """
        if isinstance(annotation, Drawer):
            body = annotation.container.body
            handle = annotation.handle
        elif isinstance(annotation, Fridge):
            body = annotation.door.body
            handle = annotation.door.handle
        elif isinstance(annotation, Door):
            body = annotation.body
            handle = annotation.handle
        else:
            return None

        joint = None
        if body.parent_connection:
            connection = body.parent_connection
            if isinstance(connection, (PrismaticConnection, RevoluteConnection)):
                joint = connection

        return handle, joint

    def _execute_and_record_trajectory(self, executor: Executor, msc: MotionStatechart):
        timeout = 500
        trajectory = []
        for _ in range(timeout):
            executor.tick()
            trajectory_value = self.effect.current_value
            trajectory.append(trajectory_value)
            if msc.is_end_motion():
                break
        else:
            print("Timeout reached.")
        return trajectory


@dataclass
class SatisfiesRequest(Predicate):
    """
    Check whether an effect satisfies a task request.

    The predicate is a pure compatibility check. The query engine provides
    the task and effect combinations and this predicate returns True only
    if the given effect satisfies the given task.
    """

    task: TaskRequest
    effect: Effect

    def __call__(self, *args, **kwargs) -> bool:
        return self._effect_satisfies_task(self.task, self.effect)

    # --- helpers ---
    def _effect_satisfies_task(self, task: TaskRequest, effect: Effect) -> bool:
        # If the task specifies a concrete desired effect, enforce compatibility
        if task.desired_effect is not None:
            return self._effects_compatible(task.desired_effect, effect)

        # Fallback: map by task type to effect kind
        task_type = (task.task_type or "").lower()
        if task_type == "open":
            return isinstance(effect, OpenEffect)
        if task_type == "close":
            return isinstance(effect, ClosedEffect)
        return False

    @staticmethod
    def _effects_compatible(desired: Effect, actual: Effect) -> bool:
        # same intent via class match
        if type(desired) is not type(actual):
            return False

        # same target if desired specifies one
        desired_target = getattr(desired, "target_object", None)
        if desired_target is not None and desired_target is not getattr(
            actual, "target_object", None
        ):
            return False

        # directional goal check for known effects
        if isinstance(desired, OpenEffect):
            current = actual.property_getter(actual.target_object)
            return current >= desired.goal_value - max(
                getattr(desired, "tolerance", 0.0), getattr(actual, "tolerance", 0.0)
            )
        if isinstance(desired, ClosedEffect):
            current = actual.property_getter(actual.target_object)
            return current <= desired.goal_value + max(
                getattr(desired, "tolerance", 0.0), getattr(actual, "tolerance", 0.0)
            )

        # generic tolerance check
        current = actual.property_getter(actual.target_object)
        tol = max(getattr(desired, "tolerance", 0.0), getattr(actual, "tolerance", 0.0))
        return abs(current - desired.goal_value) <= tol
