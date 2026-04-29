"""
BMP predicate definitions for the Law of Task-Achieving Body Motion.

The law states that a robot can successfully execute a manipulation task if and
only if three independent conditions hold simultaneously:

- **Semantic correctness** (SatisfiesRequest): the intended world-state change
  matches the task goal, i.e. the right effect is the one that was requested.
- **Causal sufficiency** (Causes): the motion physically produces that world-state
  change under the scoped physics model within its declared validity range.
- **Embodiment feasibility** (CanPerform): the robot can actually execute the
  motion within its kinematic and dynamic limits.

Each condition is an independently callable predicate. The same predicate
structure applies uniformly across all manipulation domains. By holding some
arguments fixed and solving for others, the predicates support motion generation,
verification, failure diagnosis, and counterfactual reasoning.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
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
    Causal sufficiency predicate.

    Checks whether a motion trajectory physically produces the desired world-state
    change under the scoped physics model. The check replays the trajectory in a
    sandboxed copy of the world and tests whether the effect is achieved at the end.

    If no trajectory is available but a physics model is attached to the motion,
    the model is first used to generate a trajectory from the current world state.

    Returns ``False`` if the effect is already achieved before the motion, treating
    it as a no-op rather than a success.
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
    Semantic correctness predicate.

    Checks that the intended effect matches the goal condition embedded in the
    task specification. This verifies that the proposed world-state change is
    semantically consistent with what was requested, independently of whether
    any motion can physically produce it.
    """

    task: TaskRequest
    effect: Effect

    def __call__(self) -> bool:
        return self.task.goal(self.effect)


@dataclass
class CanPerform(Predicate):
    """
    Embodiment feasibility predicate.

    Checks whether a robot can physically execute a motion trajectory given its
    kinematic and dynamic limits and collision-free constraints. The check runs a
    whole-body motion planner that attempts to track the trajectory with each of
    the robot's manipulators in turn.

    Returns ``True`` as soon as any gripper successfully completes the trajectory.
    Returns ``False`` if no gripper can complete it within the timeout.

    Subclass and implement the abstract methods to define embodiment feasibility
    for a specific manipulation domain.
    """

    motion: Motion
    robot: AbstractRobot

    _timeout: ClassVar[int] = 500

    def __call__(self) -> bool:
        if not self.motion.trajectory:
            return False
        target, trajectory = self._resolve_target_and_trajectory()
        return self._execute_for_any_gripper(target, trajectory)

    def _resolve_target_and_trajectory(self) -> tuple[Any, list]:
        """
        Resolve the target body and compute its world-space trajectory.

        Override to wrap resolution in a world state reset context.
        """
        target = self._resolve_target()
        return target, self._compute_body_trajectory(target)

    @abstractmethod
    def _resolve_target(self) -> Any:
        """Return the target body or annotation that the gripper will track."""

    @abstractmethod
    def _compute_body_trajectory(self, target: Any) -> list:
        """Convert the actuator-space trajectory to a sequence of world-space poses."""

    @abstractmethod
    def _build_msc(
        self, root: Any, gripper: Any, target: Any, trajectory: list
    ) -> MotionStatechart:
        """Build the MotionStatechart for a single gripper attempt."""

    @abstractmethod
    def _build_collision_rules(self, gripper: Any, target: Any) -> list:
        """Return the collision rules to apply for a single gripper attempt."""

    def _build_executor(self, msc: MotionStatechart) -> Executor:
        """Construct the Executor for a single gripper attempt."""
        return Executor(context=MotionStatechartContext(world=self.robot._world))

    def _is_expected_exception(self, exception: Exception) -> bool:
        """Return True if the exception is an expected execution failure."""
        return isinstance(exception, TimeoutError)

    def _execute_for_any_gripper(self, target: Any, trajectory: list) -> bool:
        root = self.robot._world.root
        result = False
        for gripper in self.robot.manipulators:
            msc = self._build_msc(root, gripper, target, trajectory)

            self.robot._world.collision_manager.clear_temporary_rules()
            self.robot._world.collision_manager.extend_temporary_rule(
                self._build_collision_rules(gripper, target)
            )
            self.robot._world.collision_manager.update_collision_matrix()

            executor = self._build_executor(msc)
            executor.compile(motion_statechart=msc)

            with self.robot._world.reset_state_context():
                try:
                    executor.tick_until_end(timeout=self._timeout)
                except Exception as exception:
                    if not self._is_expected_exception(exception):
                        raise
                result = msc.is_end_motion()

            self.robot._world.collision_manager.clear_temporary_rules()
            if result:
                break

        return result

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
    def _add_motion_termination_nodes(
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
        msc.add_node(local_min := LocalMinimumReached(name="local_minimum_reached"))
        msc.add_node(CancelMotion.when_true(local_min))
        msc.add_node(
            ExternalCollisionAvoidance(
                name="external_collision_avoidance",
                robot=robot,
            )
        )
