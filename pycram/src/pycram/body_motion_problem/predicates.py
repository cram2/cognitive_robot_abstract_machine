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
from giskardpy.motion_statechart.graph_node import EndMotion, CancelMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.entity_query_language.predicate import Predicate

from pycram.body_motion_problem.types import (
    Effect,
    Motion,
    TaskRequest,
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
        trajectory = self.motion.trajectory
        actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        with self.environment.reset_state_context():
            for i, position in enumerate(trajectory):
                updates = {actuator: float(position)}
                for conn, traj in self.motion.secondary_trajectories:
                    updates[conn] = float(traj[i])
                self.environment.set_positions_1DOF_connection(updates)
                if self.motion.dt is not None:
                    self.environment.step_physics(self.motion.dt)

            is_achieved_post = self.effect.is_achieved()

        return (not is_achieved_pre) and is_achieved_post


@dataclass
class SatisfiesRequest(Predicate):
    """
    Semantic correctness predicate: SatisfiesRequest(Π, G_final).

    Checks that G_final satisfies the goal predicate Π_goal embedded in the
    task specification, i.e. G_final ⊨ Π_goal.
    """

    task: TaskRequest
    effect: Effect

    def __call__(self) -> bool:
        return self.task.goal(self.effect)


@dataclass
class CanPerform(Predicate):
    """
    Embodiment feasibility predicate: CanPerform(R, τ).

    Checks whether trajectory τ is executable by robot R with respect to
    kinematic and dynamic limits and absence of self-collision.

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
        msc.add_node(
            local_min_reached := LocalMinimumReached(name="local_minimum_reached")
        )
        msc.add_node(EndMotion.when_true(sequence))
        msc.add_node(CancelMotion.when_true(local_min_reached))
        msc.add_node(
            ExternalCollisionAvoidance(
                name="external_collision_avoidance",
                robot=robot,
            )
        )
