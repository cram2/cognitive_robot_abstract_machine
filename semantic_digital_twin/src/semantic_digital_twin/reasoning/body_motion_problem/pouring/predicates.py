"""
Concrete BMP predicate implementations for the pouring domain (D_pour).
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
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
)
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.world_entity import Body


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

    Extends :class:`Causes` to handle the two coupled DOFs of the pouring domain:
    the tilt joint (``motion.actuator``) and the fill-level joint read from the
    effect's target object. Both trajectories are replayed during
    :meth:`_map_motion_to_effect` so that the fill-level effect can be verified
    against the world state.
    """

    _fill_trajectory: List[float] = field(default_factory=list, init=False)

    @property
    def _fill_connection(self) -> PrismaticConnection:
        """Fill-level connection read from the effect's target object."""
        return self.effect.target_object.fill_connection

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
                self._fill_trajectory = self.motion.motion_model.last_fill_trajectory

        return self._map_motion_to_effect()

    def _map_motion_to_effect(self) -> bool:
        initial_state = self.environment.state._data.copy()
        tilt_actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        for tilt_pos, fill_pos in zip(self.motion.trajectory, self._fill_trajectory):
            self.environment.set_positions_1DOF_connection(
                {
                    tilt_actuator: float(tilt_pos),
                    self._fill_connection: float(fill_pos),
                }
            )

        is_achieved_post = self.effect.is_achieved()

        self.environment.state._data[:] = initial_state
        self.environment.notify_state_change()

        return (not is_achieved_pre) and is_achieved_post

    def replay(self, step_delay: float = 0.05) -> None:
        """
        Re-apply the computed trajectory to the world with a per-step delay.

        Leaves the world in the final state of the trajectory so that the result
        is visible in RViz. Call this after a successful :meth:`__call__`.

        :param step_delay: Seconds to sleep between steps (default 50 ms ≈ 20 fps).
        """
        tilt_actuator = self.motion.actuator
        for tilt_pos, fill_pos in zip(self.motion.trajectory, self._fill_trajectory):
            self.environment.set_positions_1DOF_connection(
                {
                    tilt_actuator: float(tilt_pos),
                    self._fill_connection: float(fill_pos),
                }
            )
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
            initial_state_data = self.robot._world.state._data.copy()
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
            try:
                executor.tick_until_end(timeout=900)
            except TimeoutError:
                pass

            result = msc.is_end_motion()
            self.robot._world.state._data[:] = initial_state_data
            self.robot._world.notify_state_change()
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

        waypoints = [
            CartesianPose(
                root_link=root,
                tip_link=gripper.tool_frame,
                goal_pose=pose,
                name=f"waypoint_{i}",
                threshold=0.05,
            )
            for i, pose in enumerate(cup_trajectory)
        ]

        full_sequence = Sequence(nodes=waypoints, name="full_trajectory_sequence")
        msc.add_node(full_sequence)

        msc.add_node(EndMotion.when_true(full_sequence))
        msc.add_node(
            ExternalCollisionAvoidance(
                name="external_collision_avoidance",
                robot=self.robot,
            )
        )
        return msc
