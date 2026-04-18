"""
Concrete BMP predicate implementations for articulated container manipulation (D_artic).

Implements SatisfiesRequest and CanPerform for the domain of opening and
closing articulated containers (drawers, doors) in kitchen environments.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.pointing import Pointing
from krrood.entity_query_language.factories import an, entity, variable, or_, and_
from krrood.entity_query_language.predicate import HasType

from pycram.utils import link_pose_for_joint_config

from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.reasoning.body_motion_problem.predicates import (
    CanPerform,
    SatisfiesRequest,
)
from semantic_digital_twin.reasoning.body_motion_problem.container_manipulation.effects import (
    ClosedEffect,
    OpenedEffect,
)
from semantic_digital_twin.reasoning.body_motion_problem.types import (
    Effect,
    Motion,
    TaskRequest,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Drawer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


@dataclass
class ContainerSatisfiesRequest(SatisfiesRequest):
    """
    Semantic correctness check for the D_artic TEE class.

    Maps open/close task types to OpenedEffect/ClosedEffect, and validates
    that the effect's target object matches the task's target name.
    """

    def __call__(self, *args, **kwargs) -> bool:
        return self._check(self.task, self.effect)

    @staticmethod
    def _check(task: TaskRequest, effect: Effect) -> bool:
        task_type = (task.task_type or "").lower()
        if "open" not in task.name and "close" not in task.name:
            target_name = effect.target_object.root.name.name
            if target_name != task.name:
                return False
        if task_type == "open":
            return isinstance(effect, OpenedEffect)
        if task_type == "close":
            return isinstance(effect, ClosedEffect)
        return False


@dataclass
class ContainerCanPerform(CanPerform):
    """
    Embodiment feasibility check for the D_artic TEE class.

    Verifies that a robot can execute a container-opening motion by
    simulating whole-body motion planning: the robot faces the handle,
    approaches it, and follows the handle trajectory with one gripper,
    while respecting external collision constraints.
    """

    def __call__(self, *args, **kwargs) -> bool:
        """
        Check if any of the robot's grippers can follow the handle trajectory.
        """
        if not self.motion.trajectory:
            return False

        with self.robot._world.reset_state_context():
            target_body = self._resolve_target_body()
            handle_bodies = (
                [target_body]
                if isinstance(target_body, Body)
                else list(target_body.bodies)
            )
            handle_trajectory = self._compute_handle_trajectory(target_body)

        approach_trajectory = handle_trajectory[: len(handle_trajectory) // 4][::-1]

        result = False
        root = self.robot._world.root
        for gripper in self.robot.manipulators:
            msc = self._build_msc(
                root, gripper, target_body, approach_trajectory, handle_trajectory
            )

            self.robot._world.collision_manager.clear_temporary_rules()
            self.robot._world.collision_manager.extend_temporary_rule(
                [
                    AvoidExternalCollisions(robot=self.robot),
                    AllowCollisionBetweenGroups(
                        body_group_a=[b for b in gripper.bodies if b.has_collision()],
                        body_group_b=[b for b in handle_bodies if b.has_collision()],
                    ),
                ]
            )
            self.robot._world.collision_manager.update_collision_matrix()

            executor = Executor(
                context=MotionStatechartContext(world=self.robot._world),
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

    def _resolve_target_body(self):
        """
        Resolve the handle body from the motion model or via EQL query.
        """
        if self.motion.motion_model:
            return self.motion.motion_model.msc.nodes[0].tip_link
        return list(
            an(
                entity(drawer := variable(SemanticAnnotation, None)).where(
                    or_(
                        and_(
                            HasType(drawer, Drawer),
                            drawer.root.parent_connection == self.motion.actuator,
                        ),
                        and_(
                            HasType(drawer, Door),
                            drawer.root.parent_connection == self.motion.actuator,
                        ),
                    )
                )
            ).evaluate()
        )[0].handle

    def _compute_handle_trajectory(self, target_body) -> list:
        """
        Convert the actuator-space trajectory to a sequence of handle poses in world space.
        """
        handle_trajectory = []
        for position in self.motion.trajectory[:]:
            joint_config = {self.motion.actuator.name.name: position}
            pose = link_pose_for_joint_config(target_body, joint_config)
            handle_trajectory.append(pose)
        return handle_trajectory

    def _build_msc(
        self, root, gripper, target_body, approach_trajectory, handle_trajectory
    ) -> MotionStatechart:
        """
        Build the MotionStatechart for approaching and following the handle trajectory.
        """
        msc = MotionStatechart()

        goal_point = handle_trajectory[0].to_position()
        goal_point.z = self.robot.base.bodies[0].global_pose.z
        point = Pointing(
            root_link=root,
            tip_link=self.robot.root,
            pointing_axis=self.robot.base.main_axis,
            goal_point=goal_point,
            threshold=0.2,
        )
        msc.add_node(point)

        approach_sequence = self._build_cartesian_waypoint_sequence(
            approach_trajectory,
            root,
            gripper.tool_frame,
            name_prefix="approach_waypoint",
            sequence_name="approach_trajectory_sequence",
        )
        msc.add_node(approach_sequence)

        full_sequence = self._build_cartesian_waypoint_sequence(
            handle_trajectory,
            root,
            gripper.tool_frame,
        )
        msc.add_node(full_sequence)

        keep_relation = CartesianPose(
            name="hold handle",
            root_link=target_body,
            tip_link=gripper.tool_frame,
            goal_pose=HomogeneousTransformationMatrix(
                reference_frame=gripper.tool_frame, child_frame=gripper.tool_frame
            ),
        )
        msc.add_node(keep_relation)

        approach_sequence.start_condition = point.observation_variable
        full_sequence.start_condition = approach_sequence.observation_variable
        keep_relation.start_condition = approach_sequence.observation_variable
        approach_sequence.end_condition = approach_sequence.observation_variable
        point.end_condition = point.observation_variable

        self._add_trajectory_following_nodes(msc, full_sequence, self.robot)

        return msc
