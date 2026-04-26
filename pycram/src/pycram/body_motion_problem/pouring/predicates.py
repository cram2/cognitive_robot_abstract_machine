"""
Predicate implementations for the liquid pouring domain.

Provides concrete implementations of the BMP predicates for the domain of
pouring liquids from a source container into a receiver. The physics for this
domain is based on fluid-flow differential equations coupled with QP-based
tilt control.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from pycram.body_motion_problem.predicates import CanPerform
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PouringCanPerform(CanPerform):
    """
    Embodiment feasibility check for liquid pouring.

    Verifies that a robot can execute a cup-tilt trajectory by running a
    whole-body motion planner that tracks the cup body as it tilts through
    the recorded trajectory.
    """

    def _resolve_target(self) -> Body:
        return self.motion.actuator.child

    def _compute_body_trajectory(self, target: Body) -> list:
        """
        Convert the tilt-angle trajectory to a sequence of cup body poses in world space.
        """
        reasoning_world = deepcopy(target._world)
        tilt_dof_id = self.motion.actuator.raw_dof.id
        trajectory = []
        for tilt_angle in self.motion.trajectory:
            reasoning_world.state[tilt_dof_id].position = tilt_angle
            reasoning_world.notify_state_change()
            trajectory.append(reasoning_world.get_body_by_name(target.name).global_pose)
        return trajectory

    def _build_collision_rules(self, gripper: Any, target: Body) -> list:
        cup_collision_bodies = [target] if target.has_collision() else []
        rules = [AvoidExternalCollisions(robot=self.robot)]
        if cup_collision_bodies:
            rules.append(
                AllowCollisionBetweenGroups(
                    body_group_a=[b for b in gripper.bodies if b.has_collision()],
                    body_group_b=cup_collision_bodies,
                )
            )
        return rules

    def _build_msc(
        self, root: Any, gripper: Any, target: Body, trajectory: list
    ) -> MotionStatechart:
        """
        Build the MotionStatechart for following the cup tilt trajectory.
        """
        msc = MotionStatechart()
        full_sequence = self._build_cartesian_waypoint_sequence(
            trajectory, root, gripper.tool_frame
        )
        msc.add_node(full_sequence)
        self._add_motion_termination_nodes(msc, full_sequence, self.robot)
        return msc


@dataclass
class CoupledPouringCanPerform(PouringCanPerform):
    """
    Embodiment feasibility check for the coupled pouring domain.

    Extends :class:`PouringCanPerform` by replaying x/z translation DOFs from
    ``motion.secondary_trajectories`` alongside each tilt step.
    """

    def _compute_body_trajectory(self, target: Body) -> list:
        """
        Compute the cup body trajectory with x/z DOFs set from the recorded QP trajectory.
        """
        reasoning_world = deepcopy(target._world)
        tilt_dof_id = self.motion.actuator.raw_dof.id
        secondary_by_dof = {
            conn.raw_dof.id: traj for conn, traj in self.motion.secondary_trajectories
        }
        trajectory = []
        for i, tilt_angle in enumerate(self.motion.trajectory):
            reasoning_world.state[tilt_dof_id].position = tilt_angle
            for dof_id, traj in secondary_by_dof.items():
                reasoning_world.state[dof_id].position = traj[i]
            reasoning_world.notify_state_change()
            trajectory.append(reasoning_world.get_body_by_name(target.name).global_pose)
        return trajectory
