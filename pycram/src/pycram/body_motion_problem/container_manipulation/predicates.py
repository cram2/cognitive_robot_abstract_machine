"""
Concrete BMP predicate implementations for articulated container manipulation (D_artic).

Implements SatisfiesRequest and CanPerform for the domain of opening and
closing articulated containers (drawers, doors) in kitchen environments.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
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
from pycram.body_motion_problem.types import (
    Effect,
    Motion,
    TaskRequest,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Drawer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


@dataclass
class ContainerCanPerform(CanPerform):
    """
    Embodiment feasibility check for the D_artic TEE class.

    Verifies that a robot can execute a container-opening motion by
    simulating whole-body motion planning: the robot faces the handle,
    approaches it, and follows the handle trajectory with one gripper,
    while respecting external collision constraints.
    """

    def __call__(self) -> bool:
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
                context=(MotionStatechartContext(world=self.robot._world)),
            )
            executor.compile(motion_statechart=msc)

            with self.robot._world.reset_state_context():
                try:
                    executor.tick_until_end(timeout=500)
                except Exception as e:
                    # the giskardpy local_minimum_reached cancels the motion and raises an exception.
                    # And there is no cancel Motion Exception yet.
                    if isinstance(e, TimeoutError) or "local_minimum_reached" in str(e):
                        pass
                    else:
                        raise e
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
        reasoning_world = deepcopy(target_body._world)
        reasoning_body = reasoning_world.get_body_by_name(target_body.name)
        actuator_dof_id = reasoning_world.get_degree_of_freedom_by_name(
            self.motion.actuator.name.name
        ).id

        for position in self.motion.trajectory:
            reasoning_world.state[actuator_dof_id].position = position
            reasoning_world.notify_state_change()
            handle_trajectory.append(reasoning_body.global_pose)
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
