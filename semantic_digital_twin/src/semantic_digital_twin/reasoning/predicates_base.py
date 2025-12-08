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
from ..semantic_annotations.task_effect_motion import Task, Effect, Motion
from ..world import World
from ..world_description.connections import PrismaticConnection, RevoluteConnection
from ..world_description.world_entity import SemanticAnnotation


@dataclass(frozen=True)
class PredicateMatch:
    """
    Represents a match from a predicate query.

    Different predicates populate different fields:
    - satisfies_request: task and effect
    - causes: motion and effect
    - can_perform: robot_capability and motion
    - find_valid_motions: all fields
    """

    task: Optional[Task] = None
    effect: Optional[Effect] = None
    motion: Optional[Motion] = None


class TaskAchievingBodyMotionPredicates(ABC):
    """
    Abstract base class for the three predicates from the Law of Task-Achieving Body Motion.

    These predicates form the foundation of the 3-step problem-solving framework.
    Different domains should subclass this and provide domain-specific implementations.

    Some predicates support bidirectional queries where unbound variables
    return all matching solutions.
    """

    @abstractmethod
    def satisfies_request(
        self, world: World, task: Optional[Task] = None, effect: Optional[Effect] = None
    ) -> List[PredicateMatch]:
        """
        Determine which effects satisfy which tasks.

        :param world: Semantic world containing annotations
        :param task: Specific task to check (None = query all tasks)
        :param effect: Specific effect to check (None = query all effects)
        :return: List of matches with task and effect populated
        """
        pass

    @abstractmethod
    def causes(
        self,
        world: World,
        motion: Optional[Motion] = None,
        effect: Optional[Effect] = None,
    ) -> List[PredicateMatch]:
        """
        Determine which motions cause which effects.

        :param world: Semantic world representing environment state
        :param motion: Specific motion to check (None = query all motions)
        :param effect: Specific effect to check (None = query all effects)
        :return: List of matches with motion and effect populated
        """
        pass

    @abstractmethod
    def can_perform(
        self,
        world: World,
        motion: Optional[Motion] = None,
        robot: Optional[AbstractRobot] = None,
    ) -> List[PredicateMatch]:
        """
        Determine which robots can execute which motions.

        :param world: Semantic world
        :param robot: Specific robot to check (None = query all robots)
        :param motion: Specific motion to check (None = query all motions)
        :return: List of matches with robot_capability and motion populated
        """
        pass


@dataclass
class CausesOpening(Predicate):
    """
    A causes(Motion, Effect) predicate should check whether a given motion satisfies a given effect.
            Case1: causes(motion?, effect1) -> calculate a motion satisfying the desired effect.
            Case2: causes(motion1, effect?) -> execute motion and check if any known effect is satisfied that was not satisfied before.
            Case3: causes(motion?, effect?) -> Union of Case2 for all known motions and Case1 for all known effects.
    """

    effect: Effect

    environment: World

    motion: Optional[Motion] = None

    def __call__(self, *args, **kwargs):
        if self.effect.is_achieved():
            return False

        initial_state_data = self.environment.state.data.copy()
        executor = Executor(world=self.environment)

        handle = self._extract_container_info(self.effect.target_object)

        open_goal = Open(
            tip_link=handle.body,
            environment_link=handle.body,
            goal_joint_state=self.effect.goal_value,
        )

        msc = MotionStatechart()
        msc.add_node(open_goal)
        msc.add_node(EndMotion.when_true(open_goal))

        executor.compile(motion_statechart=msc)

        executor.tick_until_end(timeout=500)

        is_achieved = self.effect.is_achieved()
        # current_value=self.environment.get_connection_by_name(joint_name).position

        # Reset state
        # self.environment.state.data = initial_state_data
        # self.environment.notify_state_change()

        self.motion = Motion(
            trajectory=[], actuator=handle.body, expected_effect=self.effect, duration=2
        )

        return True

    def _extract_container_info(self, annotation: SemanticAnnotation):
        """
        Extracts body, handle, and joint info from a semantic annotation.
        """
        if isinstance(annotation, Drawer) or isinstance(annotation, Door):
            handle = annotation.handle
        elif isinstance(annotation, Fridge):
            handle = annotation.door.handle
        else:
            return None

        return handle
