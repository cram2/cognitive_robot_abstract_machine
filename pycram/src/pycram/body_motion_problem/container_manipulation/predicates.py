"""
Predicate implementations for articulated container manipulation.

Provides concrete implementations of the BMP predicates for the domain of
opening and closing articulated containers (drawers, cupboard doors, oven doors,
dishwasher doors) in kitchen environments. The physics for this domain is based
on rigid-body kinematics of articulated mechanisms.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.pointing import Pointing
from krrood.entity_query_language.factories import an, entity, variable, or_, and_
from krrood.entity_query_language.predicate import HasType

from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from pycram.body_motion_problem.predicates import CanPerform
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Drawer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


@dataclass
class ContainerCanPerform(CanPerform):
    """
    Embodiment feasibility check for articulated container manipulation.

    Verifies that a robot can execute a container-opening or -closing motion by
    running a whole-body motion planner that:

    1. Orients the robot to face the container handle.
    2. Approaches the handle with a gripper.
    3. Follows the handle along its kinematic trajectory.

    Collision between the gripper and handle is permitted; all other collisions
    are avoided.
    """

    def _resolve_target_and_trajectory(self) -> tuple[Any, list]:
        with self.robot._world.reset_state_context():
            target = self._resolve_target()
            trajectory = self._compute_body_trajectory(target)
        return target, trajectory

    def _resolve_target(self) -> Any:
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

    def _compute_body_trajectory(self, target: Any) -> list:
        """
        Convert the actuator-space trajectory to a sequence of handle poses in world space.
        """
        handle_trajectory = []
        reasoning_world = deepcopy(target._world)
        reasoning_body = reasoning_world.get_body_by_name(target.name)
        actuator_dof_id = reasoning_world.get_degree_of_freedom_by_name(
            self.motion.actuator.name.name
        ).id

        for position in self.motion.trajectory:
            reasoning_world.state[actuator_dof_id].position = position
            reasoning_world.notify_state_change()
            handle_trajectory.append(reasoning_body.global_pose)
        return handle_trajectory

    def _build_collision_rules(self, gripper: Any, target: Any) -> list:
        handle_bodies = [target] if isinstance(target, Body) else list(target.bodies)
        return [
            AvoidExternalCollisions(robot=self.robot),
            AllowCollisionBetweenGroups(
                body_group_a=[b for b in gripper.bodies if b.has_collision()],
                body_group_b=[b for b in handle_bodies if b.has_collision()],
            ),
        ]

    def _build_msc(
        self, root: Any, gripper: Any, target: Any, trajectory: list
    ) -> MotionStatechart:
        """
        Build the MotionStatechart for approaching and following the handle trajectory.
        """
        approach_trajectory = trajectory[: len(trajectory) // 4][::-1]

        msc = MotionStatechart()

        goal_point = trajectory[0].to_position()
        goal_point.z = self.robot.base.bodies[0].global_pose.z
        main_axis = self.robot.base.main_axis
        pointing_axis = Vector3(
            main_axis.x,
            main_axis.y,
            main_axis.z,
            reference_frame=self.robot.root,
        )
        point = Pointing(
            root_link=root,
            tip_link=self.robot.root,
            pointing_axis=pointing_axis,
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
            trajectory,
            root,
            gripper.tool_frame,
        )
        msc.add_node(full_sequence)

        keep_relation = CartesianPose(
            name="hold handle",
            root_link=target,
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

        self._add_motion_termination_nodes(msc, full_sequence, self.robot)

        return msc

    def _is_expected_exception(self, exception: Exception) -> bool:
        return isinstance(exception, TimeoutError) or "local_minimum_reached" in str(
            exception
        )
