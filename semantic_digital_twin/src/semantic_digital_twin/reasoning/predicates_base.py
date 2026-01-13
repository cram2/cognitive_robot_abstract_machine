"""
Classes for Task-Achieving Body Motion Predicates.

This module defines the three predicates from the
Body Motion Problem. Different problem domains might need to overwrite an implementation:
"""

from typing import Optional
from dataclasses import dataclass, field

from giskardpy.executor import Executor
from giskardpy.model.collision_matrix_manager import CollisionRequest
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.collision_avoidance import CollisionAvoidance
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.entity_query_language.predicate import Predicate
from pycram.utils import link_pose_for_joint_config
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

        # Generate a trajectory from a motion model and check it fits the effect
        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory and len(trajectory) > 0:
                self.motion.trajectory = trajectory
            # # Verify by simulating the generated trajectory on the environment.
            # return self._map_motion_to_effect()

        # If trajectory exists check if it fits the effect
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

        return (not is_achieved_pre) and is_achieved_post


@dataclass
class SatisfiesRequest(Predicate):
    """
    Check whether an effect satisfies a task request.
    For the sake of demonstration a placeholder mapping is used
    """

    task: TaskRequest
    effect: Effect

    def __call__(self, *args, **kwargs) -> bool:
        # print(f"CALL IN SATISFIESREQUEST {self.task.task_type} {self.task.name} {self.effect.name} {self._effect_satisfies_task(self.task, self.effect)}")
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
    This predicates checks whether a motion can be executed by a robot.
    An input motion can be abstract to an robot in the way that it describes the movement of an environment entity
    and not of an part of the robot.
    For this implementation the robot only uses its grippers to interact with the environment.
    """

    motion: Motion
    robot: AbstractRobot

    def __call__(self, *args, **kwargs) -> bool:
        """
        Check if the motion can be executed by any of the robot's grippers.
        """
        if not self.motion.trajectory:
            return False

        # The child of the connection (actuator) is typically the movable part (e.g., drawer container)

        target_body = self.motion.motion_model.msc.nodes[0].tip_link

        # 1. Transform trajectory to handle coordinates (PoseStamped sequence)
        handle_trajectory = []
        for position in self.motion.trajectory:
            joint_config = {self.motion.actuator.name.name: position}
            # Calculate the global pose of the target body for the given joint position
            pose = link_pose_for_joint_config(
                target_body, joint_config, self.robot._world
            )
            handle_trajectory.append(pose)

        # 2. Test execution for each gripper
        for gripper in self.robot.manipulators:
            msc = MotionStatechart()

            # Create CartesianPose tasks for each waypoint
            waypoints = []
            root = self.robot._world.root
            for i, pose in enumerate(handle_trajectory):
                goal = CartesianPose(
                    root_link=root,
                    tip_link=gripper.tool_frame,
                    goal_pose=pose.to_spatial_type(),
                    name=f"waypoint_{i}",
                )
                waypoints.append(goal)

            # Use Sequence to wire waypoints together
            sequence_goal = Sequence(nodes=waypoints, name="trajectory_sequence")
            msc.add_node(sequence_goal)

            collision_node = CollisionAvoidance(
                collision_entries=[CollisionRequest.avoid_all_collision()],
            )
            msc.add_node(collision_node)

            # The MSC ends when the sequence is done
            msc.add_node(EndMotion.when_true(sequence_goal))
            # Simulate execution in the world
            executor = Executor(
                world=self.robot._world, collision_checker=CollisionCheckerLib.bpb
            )
            executor.compile(msc)

            with self.robot._world.reset_state_context():
                # Tick the executor until the motion ends or times out
                try:
                    executor.tick_until_end(timeout=400)
                except TimeoutError as e:
                    # If timeout is reached, the motion is considered not executable
                    pass

                # If the sequence goal reached the DONE state, this gripper can execute the motion
                # if sequence_goal.life_cycle_state == LifeCycleValues.DONE:
                if msc.is_end_motion():
                    return True

        return False
