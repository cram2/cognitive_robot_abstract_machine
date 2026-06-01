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

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from krrood.entity_query_language.predicate import Predicate

from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.effects import Effect, TaskRequest
from semantic_digital_twin.world_description.motion import Motion


@dataclass
class Causes(Predicate):
    """
    Causal sufficiency predicate.

    Checks whether a motion trajectory physically produces the desired world-state
    change under the scoped physics model. If no trajectory is available but a
    physics model is attached to the motion, the model is first used to generate
    a trajectory from the current world state.

    Returns ``False`` if the effect is already achieved before the motion.
    """

    effect: Effect
    environment: World
    motion: Optional[Motion]

    def __call__(self) -> bool:
        if self.effect.is_achieved():
            return False

        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory and len(trajectory) > 0:
                self.motion.trajectory = trajectory
                self.motion.secondary_trajectories = (
                    self.motion.motion_model.build_secondary_trajectories(self.effect)
                )

        return self._map_motion_to_effect()

    def replay(self, step_delay: float = 0.05) -> None:
        """
        Re-apply the computed trajectory to the world with a per-step delay.

        :param step_delay: Seconds to sleep between steps (default 50 ms ≈ 20 fps).
        """
        for i, position in enumerate(self.motion.trajectory):
            updates = {self.motion.actuator: float(position)}
            for conn, traj in self.motion.secondary_trajectories:
                updates[conn] = float(traj[i])
            self.environment.set_positions_1DOF_connection(updates)
            time.sleep(step_delay)

    def _map_motion_to_effect(self) -> bool:
        """
        Replay the trajectory in a sandboxed world and check whether the effect is achieved.
        """
        trajectory = self.motion.trajectory
        actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        with self.environment.reset_state_context():
            for i, position in enumerate(trajectory):
                updates = {actuator: float(position)}
                for conn, traj in self.motion.secondary_trajectories:
                    updates[conn] = float(traj[i])
                self.environment.set_positions_1DOF_connection(updates)
                if self.motion.time_step is not None:
                    self.environment.step_physics(self.motion.time_step)

            is_achieved_post = self.effect.is_achieved()

        return (not is_achieved_pre) and is_achieved_post


@dataclass
class SatisfiesRequest(Predicate):
    """
    Semantic correctness predicate.

    Checks that the intended effect matches the goal condition embedded in the
    task specification, independently of whether any motion can physically produce it.
    """

    task: TaskRequest
    effect: Effect

    def __call__(self) -> bool:
        return self.task.goal(self.effect)


@dataclass
class CanPerform(Predicate):
    """
    Embodiment feasibility predicate.

    Checks whether a robot can physically execute the motion trajectory,
    independently of task success. Concrete implementations live in the layer
    that owns the execution machinery (e.g., pycram with giskardpy).
    """

    motion: Motion
    robot: AbstractRobot

    @abstractmethod
    def __call__(self) -> bool: ...
