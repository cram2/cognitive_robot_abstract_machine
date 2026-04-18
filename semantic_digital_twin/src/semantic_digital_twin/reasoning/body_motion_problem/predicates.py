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

from dataclasses import dataclass, field
from typing import List, Optional

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

    def __call__(self, *args, **kwargs):
        self._check_validity_intervals()

        if self.effect.is_achieved():
            return False

        # Case 1: no trajectory yet — generate one via the physics model
        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory and len(trajectory) > 0:
                self.motion.trajectory = trajectory

        return self._map_motion_to_effect()

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

    Subclass this predicate to implement domain-specific semantic correctness
    checks for a given TEE class.
    """

    task: TaskRequest
    effect: Effect

    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__.")


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

    def __call__(self, *args, **kwargs) -> bool:
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
