"""
Abstract BMP predicate definitions.

The three predicates of the Law of Task-Achieving Body Motion:

  SatisfiesRequest(Π, G_final) — semantic correctness
  Causes(τ, G_final, Φ, I_Φ)  — causal sufficiency
  CanPerform(R, τ)              — embodiment feasibility

Together they form the axiom:
  ∀R, E, Π, G, τ:
    SatisfiesRequest(Π, G_final) ∧
    Causes(τ, G_final, Φ, I_Φ) ∧
    CanPerform(R, τ)
    ⟹ CanAchieve(R, E, Π, τ)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.entity_query_language.predicate import Predicate

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    Effect,
    Motion,
    OutOfScopeError,
    TaskRequest,
    TEEClass,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World


@dataclass
class Causes(Predicate):
    """
    Causal sufficiency predicate: Causes(τ, G_final, Φ, I_Φ).

    Checks whether trajectory τ is a physically valid explanation for
    transitioning from the current SDT state to G_final under the scoped
    physics model Φ.

    Supports three usage modes:
      Case 1 — motion unknown: generate τ from motion_model, verify against effect.
      Case 2 — effect unknown: execute τ and check which effects become true.
      Case 3 — both unknown: union of Case 1 and Case 2.

    Raises OutOfScopeError if tee_class is set and the current world parameters
    fall outside the validity intervals I_Φ.
    """

    effect: Effect

    environment: World

    motion: Optional[Motion]

    tee_class: Optional[TEEClass] = field(default=None)

    def __call__(self) -> bool:
        self._check_validity_intervals()

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

        Leaves the world in the final state so the result is visible in RViz.
        Call this after a successful __call__.

        :param step_delay: Seconds to sleep between steps (default 50 ms ≈ 20 fps).
        """
        for i, position in enumerate(self.motion.trajectory):
            updates = {self.motion.actuator: float(position)}
            for conn, traj in self.motion.secondary_trajectories:
                updates[conn] = float(traj[i])
            self.environment.set_positions_1DOF_connection(updates)
            time.sleep(step_delay)

    def _check_validity_intervals(self) -> None:
        """
        Raise OutOfScopeError if any physics parameter is outside its I_Φ interval.

        :raises OutOfScopeError: when the current value of the effect property
            falls outside the bounds declared in tee_class.validity_intervals.
        """
        if not self.tee_class or not self.tee_class.validity_intervals:
            return
        current = self.effect.current_value
        for param_name, (lower, upper) in self.tee_class.validity_intervals.items():
            if not (lower <= current <= upper):
                raise OutOfScopeError(
                    f"Parameter '{param_name}' value {current} is outside [{lower}, {upper}]"
                )

    def _map_motion_to_effect(self) -> bool:
        trajectory = self.motion.trajectory
        actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        with self.environment.reset_state_context():
            for i, position in enumerate(trajectory):
                updates = {actuator: float(position)}
                for conn, traj in self.motion.secondary_trajectories:
                    updates[conn] = float(traj[i])
                self.environment.set_positions_1DOF_connection(updates)

            is_achieved_post = self.effect.is_achieved()

        return (not is_achieved_pre) and is_achieved_post


@dataclass
class SatisfiesRequest(Predicate):
    """
    Semantic correctness predicate: SatisfiesRequest(Π, G_final).

    Checks that a final SDT state (represented by an Effect) matches the
    intent of a task specification (TaskRequest).

    When task_type and effect_type are provided, the base __call__ handles the
    check directly. Subclass only when additional domain-specific validation is
    needed (e.g. name-matching).

    :param task_type: Expected task_type string (e.g. ``"pour"``).
    :param effect_type: Expected Effect subclass (e.g. ``PouringEffect``).
    """

    task: TaskRequest
    effect: Effect
    task_type: Optional[str] = field(default=None)
    effect_type: Optional[Type[Effect]] = field(default=None)

    def __call__(self) -> bool:
        if self.task_type is not None and self.task.task_type != self.task_type:
            return False
        if self.effect_type is not None and not isinstance(
            self.effect, self.effect_type
        ):
            return False
        if self.task_type is None and self.effect_type is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set task_type/effect_type or implement __call__."
            )
        return True


@dataclass
class CanPerform(Predicate):
    """
    Embodiment feasibility predicate: CanPerform(R, τ).

    Checks whether trajectory τ is executable by robot R with respect to
    kinematic and dynamic limits and absence of self-collision:
      CanPerform(R, τ) ⟺ ∀t∈τ: (q_t, q̇_t, q̈_t) ∈ K_R ∧ ¬SelfCollision(q_t)

    Subclass this predicate to implement embodiment feasibility checks for
    a specific robot morphology or TEE class.
    """

    motion: Motion
    robot: AbstractRobot

    def __call__(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__.")

    @staticmethod
    def _build_cartesian_waypoint_sequence(
        poses: list,
        root_link,
        tip_link,
        name_prefix: str = "waypoint",
        sequence_name: str = "full_trajectory_sequence",
        threshold: float = 0.05,
    ) -> Sequence:
        """
        Build a Sequence of CartesianPose waypoints from a list of target poses.

        :param poses: Ordered list of goal poses for the tip link.
        :param root_link: Root link for the Cartesian goal.
        :param tip_link: Tip link (e.g. gripper tool frame) to move.
        :param name_prefix: Prefix for individual waypoint node names.
        :param sequence_name: Name of the resulting Sequence node.
        :param threshold: Position tolerance for each waypoint.
        :return: A Sequence node ready to be added to a MotionStatechart.
        """
        waypoints = [
            CartesianPose(
                root_link=root_link,
                tip_link=tip_link,
                goal_pose=pose,
                name=f"{name_prefix}_{i}",
                threshold=threshold,
            )
            for i, pose in enumerate(poses)
        ]
        return Sequence(nodes=waypoints, name=sequence_name)

    @staticmethod
    def _add_trajectory_following_nodes(
        msc: MotionStatechart,
        sequence: Sequence,
        robot: AbstractRobot,
    ) -> None:
        """
        Add EndMotion and ExternalCollisionAvoidance nodes to msc for a trajectory sequence.

        :param msc: MotionStatechart to modify in place.
        :param sequence: The Sequence node that triggers EndMotion on completion.
        :param robot: Robot used for collision avoidance.
        """
        msc.add_node(EndMotion.when_true(sequence))
        msc.add_node(
            ExternalCollisionAvoidance(
                name="external_collision_avoidance",
                robot=robot,
            )
        )
