"""
Concrete BMP predicate implementations for the pouring domain (D_pour).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.reasoning.body_motion_problem.predicates import (
    CanPerform,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PouringCanPerform(CanPerform):
    """
    Embodiment feasibility check for the D_pour TEE class.

    Verifies that a robot can execute the cup-tilt trajectory by simulating
    whole-body motion planning: the robot faces the cup, approaches it, and
    follows the cup body as it tilts through the recorded trajectory.
    """

    def __call__(self) -> bool:
        if not self.motion.trajectory:
            return False

        cup_body = self.motion.actuator.child
        cup_trajectory = self._compute_cup_trajectory(cup_body)

        result = False
        root = self.robot._world.root
        for gripper in self.robot.manipulators:
            msc = self._build_msc(root, gripper, cup_trajectory)

            self.robot._world.collision_manager.clear_temporary_rules()
            cup_collision_bodies = [cup_body] if cup_body.has_collision() else []
            collision_rules = [AvoidExternalCollisions(robot=self.robot)]
            if cup_collision_bodies:
                collision_rules.append(
                    AllowCollisionBetweenGroups(
                        body_group_a=[b for b in gripper.bodies if b.has_collision()],
                        body_group_b=cup_collision_bodies,
                    )
                )
            self.robot._world.collision_manager.extend_temporary_rule(collision_rules)
            self.robot._world.collision_manager.update_collision_matrix()

            executor = Executor(
                context=MotionStatechartContext(world=self.robot._world),
                pacer=SimulationPacer(real_time_factor=1.0),
            )
            executor.compile(motion_statechart=msc)

            with self.robot._world.reset_state_context():
                try:
                    executor.tick_until_end(timeout=900)
                except TimeoutError:
                    pass
                result = msc.is_end_motion()

            self.robot._world.collision_manager.clear_temporary_rules()
            if result:
                break

        return result

    def _compute_cup_trajectory(self, cup_body: Body) -> list:
        """
        Convert the tilt-angle trajectory to a sequence of cup body poses in world space.
        """
        reasoning_world = deepcopy(cup_body._world)
        tilt_dof_id = self.motion.actuator.raw_dof.id
        trajectory = []
        for tilt_angle in self.motion.trajectory:
            reasoning_world.state[tilt_dof_id].position = tilt_angle
            reasoning_world.notify_state_change()
            trajectory.append(
                reasoning_world.get_body_by_name(cup_body.name).global_pose
            )
        return trajectory

    def _build_msc(self, root, gripper, cup_trajectory) -> MotionStatechart:
        """
        Build the MotionStatechart for following the cup tilt trajectory.
        """
        msc = MotionStatechart()
        full_sequence = self._build_cartesian_waypoint_sequence(
            cup_trajectory, root, gripper.tool_frame
        )
        msc.add_node(full_sequence)
        self._add_trajectory_following_nodes(msc, full_sequence, self.robot)
        return msc


@dataclass
class CoupledPouringCanPerform(PouringCanPerform):
    """
    Embodiment feasibility check for the coupled pouring domain.

    Extends :class:`PouringCanPerform` by replaying x/z translation DOFs from
    ``motion.secondary_trajectories`` alongside each tilt step, so the robot
    follows the cup as it both tilts and moves to keep the rim above the receiver.
    """

    def _compute_cup_trajectory(self, cup_body: Body) -> list:
        """
        Compute the cup body trajectory with x/z DOFs set from the recorded QP trajectory.
        """
        reasoning_world = deepcopy(cup_body._world)
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
            trajectory.append(
                reasoning_world.get_body_by_name(cup_body.name).global_pose
            )
        return trajectory
