"""
Concrete BMP predicate implementations for the pouring domain (D_pour).
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.reasoning.body_motion_problem.predicates import (
    CanPerform,
    Causes,
    SatisfiesRequest,
)
from semantic_digital_twin.reasoning.body_motion_problem.pouring.effects import (
    PouringEffect,
    ReceiverFillEffect,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world_description.world_entity import Body
from dataclasses import field as dataclass_field


@dataclass
class PouringSatisfiesRequest(SatisfiesRequest):
    """
    Semantic correctness check for the D_pour TEE class.

    Returns ``True`` when the task type is ``"pour"`` and the effect is a
    :class:`PouringEffect`.
    """

    def __call__(self, *args, **kwargs) -> bool:
        if self.task.task_type != "pour":
            return False
        return isinstance(self.effect, PouringEffect)


@dataclass
class PouringCauses(Causes):
    """
    Causal sufficiency predicate for D_pour.

    Populates ``motion.secondary_trajectories`` with the fill-level joint
    trajectory after simulation so that the base :meth:`_map_motion_to_effect`
    can replay both DOFs (tilt + fill-level) when verifying the effect.
    """

    def __call__(self, *args, **kwargs) -> bool:
        if self.effect.is_achieved():
            return False

        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory:
                self.motion.trajectory = trajectory
                fill_connection = self.effect.target_object.fill_connection
                self.motion.secondary_trajectories = [
                    (fill_connection, self.motion.motion_model.last_fill_trajectory)
                ]

        return self._map_motion_to_effect()

    def replay(self, step_delay: float = 0.05) -> None:
        """
        Re-apply the computed trajectory to the world with a per-step delay.

        Leaves the world in the final state of the trajectory so that the result
        is visible in RViz. Call this after a successful :meth:`__call__`.

        :param step_delay: Seconds to sleep between steps (default 50 ms ≈ 20 fps).
        """
        tilt_actuator = self.motion.actuator
        secondary = self.motion.secondary_trajectories
        for i, tilt_pos in enumerate(self.motion.trajectory):
            updates = {tilt_actuator: float(tilt_pos)}
            for conn, traj in secondary:
                updates[conn] = float(traj[i])
            self.environment.set_positions_1DOF_connection(updates)
            time.sleep(step_delay)


@dataclass
class PouringCanPerform(CanPerform):
    """
    Embodiment feasibility check for the D_pour TEE class.

    Verifies that a robot can execute the cup-tilt trajectory by simulating
    whole-body motion planning: the robot faces the cup, approaches it, and
    follows the cup body as it tilts through the recorded trajectory.
    """

    def __call__(self, *args, **kwargs) -> bool:
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
class CoupledPouringCauses(Causes):
    """
    Causal sufficiency predicate for the coupled pouring chain (source → receiver).

    After running the :class:`CoupledPouringMSCModel`, populates
    ``motion.secondary_trajectories`` with three entries:

    1. Source tilt connection (primary trajectory, already in ``motion.trajectory``)
    2. Source fill connection (fill level decreasing)
    3. Receiver fill connection (fill level increasing)

    The base :meth:`_map_motion_to_effect` then replays all three DOFs
    and checks that the *receiver* fill effect is achieved.

    :param source: Source container, used to access its fill connection.
    """

    source: HasFillLevel = dataclass_field(kw_only=True)

    def __call__(self, *args, **kwargs) -> bool:
        if self.effect.is_achieved():
            return False

        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory:
                self.motion.trajectory = trajectory
                model = self.motion.motion_model
                self.motion.secondary_trajectories = [
                    (
                        self.source.fill_connection,
                        model.source_model.last_fill_trajectory,
                    ),
                    (
                        self.effect.target_object.fill_connection,
                        model.last_receiver_fill_trajectory,
                    ),
                    (model.x_translation_connection, model.last_x_trajectory),
                    (model.z_translation_connection, model.last_z_trajectory),
                ]

        return self._map_motion_to_effect()

    def replay(self, step_delay: float = 0.05) -> None:
        """
        Re-apply the computed trajectory to the world with a per-step delay,
        animating tilt, source fill decrease, and receiver fill increase.

        :param step_delay: Seconds to sleep between steps.
        """
        tilt_actuator = self.motion.actuator
        secondary = self.motion.secondary_trajectories
        for i, tilt_pos in enumerate(self.motion.trajectory):
            updates = {tilt_actuator: float(tilt_pos)}
            for conn, traj in secondary:
                updates[conn] = float(traj[i])
            self.environment.set_positions_1DOF_connection(updates)
            time.sleep(step_delay)


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
