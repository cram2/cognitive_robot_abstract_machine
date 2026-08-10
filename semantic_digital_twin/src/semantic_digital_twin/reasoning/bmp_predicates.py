"""
BMP predicate definitions for the Law of Task-Achieving Body Motion.

The law states that a robot can successfully execute a manipulation task if and
only if three independent conditions hold simultaneously:

- **Semantic correctness** (SatisfiesRequest): the intended world-state change
  matches the task goal.
- **Causal sufficiency** (Causes): the motion physically produces that world-state
  change under the scoped physics model.
- **Embodiment feasibility** (CanPerform): the robot can actually execute the
  motion — defined in the layer that owns the execution machinery.
"""

# %% imports

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    clause,
    Noun,
    Verb,
)

from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.exceptions import MissingMotionTrajectoryError
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.effects import Effect, TaskRequest
from semantic_digital_twin.world_description.motion import Motion

if TYPE_CHECKING:
    from krrood.entity_query_language.predicate import RenderedFields
    from krrood.entity_query_language.verbalization.fragments.base import (
        VerbalizationFragment,
    )


# %% causal sufficiency


@dataclass
class Causes(Predicate):
    """
    Causal sufficiency predicate.

    Checks whether a motion trajectory physically produces the desired world-state
    change under the scoped physics model. If no trajectory is available but a physics
    model is attached to the motion, the model is first used to generate a trajectory
    from the current world state.

    Returns ``False`` if the effect is already achieved before the motion.
    """

    effect: Effect
    """The world-state change the motion is checked to produce."""

    environment: World
    """
    The world in which the motion's consequences are simulated.
    """

    motion: Optional[Motion]
    """The candidate motion, or ``None`` when no motion is proposed."""

    def __call__(self) -> bool:
        """
        Check whether the motion causes the effect.

        When the motion carries a physics model but no trajectory yet, the model is run
        and the generated trajectory is stored on the motion.

        :return:``True`` if replaying the trajectory achieves the not-yet-achieved
            effect.
        """
        if self.effect.is_achieved():
            return False

        if (
            self.motion
            and self.motion.motion_model
            and self.motion.motion_trajectory is None
        ):
            motion_trajectory = self.motion.motion_model.run(
                self.effect, self.environment
            )
            if not motion_trajectory.is_empty():
                self.motion.motion_trajectory = motion_trajectory

        return self._map_motion_to_effect()

    def replay(self, step_delay: float = 0.05) -> None:
        """
        Re-apply the computed trajectory to the world with a per-step delay.

        :param step_delay: Seconds to sleep between steps (default 50 ms ≈ 20 fps).
        :raises MissingMotionTrajectoryError: If there is no motion, it carries no
            trajectory, or the trajectory does not track the motion's connection.
        """
        if self.motion is None or self.motion.motion_trajectory is None:
            raise MissingMotionTrajectoryError(motion=self.motion)
        actuator_positions = self.motion.motion_trajectory.positions_for(
            self.motion.connection
        )
        if not actuator_positions:
            raise MissingMotionTrajectoryError(motion=self.motion)
        for step in range(len(actuator_positions)):
            JointState.from_mapping(
                self.motion.motion_trajectory.position_updates_at(step)
            ).apply_to(self.environment)
            time.sleep(step_delay)

    def _map_motion_to_effect(self) -> bool:
        """
        Replay the trajectory in a sandboxed world and check whether the effect is
        achieved.
        """
        if self.motion is None or self.motion.motion_trajectory is None:
            return False

        actuator_positions = self.motion.motion_trajectory.positions_for(
            self.motion.connection
        )
        is_achieved_pre = self.effect.is_achieved()

        with self.environment.reset_state_context():
            for step in range(len(actuator_positions)):
                JointState.from_mapping(
                    self.motion.motion_trajectory.position_updates_at(step)
                ).apply_to(self.environment)
                if self.motion.time_step is not None:
                    self.environment.step_physics(self.motion.time_step)

            is_achieved_post = self.effect.is_achieved()

        return (not is_achieved_pre) and is_achieved_post

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        return clause(Noun(fields["motion"]), Verb("cause"), Noun(fields["effect"]))


# %% semantic correctness


@dataclass
class SatisfiesRequest(Predicate):
    """
    Semantic correctness predicate.

    Checks that the intended effect matches the goal condition embedded in the task
    specification, independently of whether any motion can physically produce it.
    """

    task: TaskRequest
    """
    The task specification whose goal condition is checked.
    """

    effect: Effect
    """
    The intended effect checked against the task's goal condition.
    """

    def __call__(self) -> bool:
        return self.task.goal(self.effect)

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        return clause(Noun(fields["effect"]), Verb("satisfy"), Noun(fields["task"]))


# %% embodiment feasibility


@dataclass
class CanPerform(Predicate):
    """
    Embodiment feasibility predicate.

    Checks whether a robot can physically execute the motion trajectory, independently
    of task success. Concrete implementations live in the layer that owns the execution
    machinery (e.g., coraplex with giskardpy).
    """

    motion: Motion
    """
    The motion trajectory the robot must execute.
    """

    robot: AbstractRobot
    """
    The robot whose embodiment is checked against the motion.
    """

    @abstractmethod
    def __call__(self) -> bool: ...

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        return clause(Noun(fields["robot"]), Verb("perform"), Noun(fields["motion"]))
