from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import ClassVar

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    CollisionViolatedError,
    LocalMinimumReachedError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, CancelMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose

from krrood.utils import clear_memoization_cache
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.reasoning.bmp_predicates import CanPerform
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    EndEffector,
    MobileBase,
)
from semantic_digital_twin.semantic_annotations.mixins import HasHandle
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.locations.costmaps import (
    OccupancyCostmap,
    OrientationGenerator,
    RingCostmap,
)


@dataclass
class MotionStatechartCanPerform(CanPerform):
    """
    Actual implementation of the abstract interface defined in CanPerform.
    Whole-body feasibility check using QP-based motion planning with
    costmap-driven base placement.

    Samples candidate base poses around the start of the trajectory to avoid
    local minima in the QP solver. For each candidate, tests whether the gripper
    can follow the full trajectory of world-space poses.
    """

    _timeout: ClassVar[int] = 200
    _costmap_samples: ClassVar[int] = 10
    _arm_reach_distance: ClassVar[float] = 0.7
    _external_collision_buffer_distance: ClassVar[float] = 0.01

    def __call__(self) -> bool:
        gc.collect()
        if not self.motion.trajectory:
            return False
        world = self.robot._world
        with world.reset_state_context():
            target = self._resolve_target()
            trajectory = self._compute_body_trajectory(target)
        with world.reset_state_context():
            return self._execute_for_any_gripper(target, trajectory)

    def _resolve_target(self) -> Body:
        """
        :return: The body the gripper will track. Prefers the physics model's
                 interaction body, then the handle of any semantic annotation
                 whose actuator matches, then the actuator child as last resort.
        """
        if self.motion.motion_model:
            body = self.motion.motion_model.interaction_body()
            if body is not None:
                return body
        for annotation in self.robot._world.get_semantic_annotations_by_type(HasHandle):
            if (
                annotation.root.parent_connection == self.motion.actuator
                and annotation.handle is not None
            ):
                return annotation.handle.root
        return self.motion.actuator.child

    def _compute_body_trajectory(self, target: Body) -> list[Pose]:
        """
        :return: World-space poses of the target body at each trajectory step.

        Caller must hold a ``reset_state_context()`` so that world state is
        restored after this method returns.
        """
        world = target._world
        dof_id = self.motion.actuator.raw_dof.id
        trajectory = []
        for position in self.motion.trajectory:
            world.state[dof_id].position = position
            world.notify_state_change()
            trajectory.append(target.global_pose)
        return trajectory

    def _build_motion_statechart(
        self, root: Body, gripper: Body, trajectory: list[Pose]
    ) -> tuple[MotionStatechart, Sequence]:
        """
        :return: A MotionStatechart that tracks the trajectory as a sequence of
                 CartesianPose goals, and the Sequence node for termination wiring.
        """
        motion_statechart = MotionStatechart()
        sequence = Sequence(
            [
                CartesianPose(
                    root_link=root,
                    tip_link=gripper,
                    goal_pose=pose,
                    name=f"step_{i}",
                )
                for i, pose in enumerate(trajectory)
            ]
        )
        motion_statechart.add_node(sequence)
        return motion_statechart, sequence

    def _build_collision_rules(self, gripper: EndEffector, target: Body) -> list:
        """
        :return: Rules allowing gripper–target collision during trajectory following.
                 Covers the full kinematic chain from the target body up to the world root
                 so the gripper can approach without being blocked by parent links.
        """
        chain = self._kinematic_chain_to_root(target)
        target_collision_bodies = [b for b in chain if b.has_collision()]
        if not target_collision_bodies:
            return []
        return [
            AllowCollisionBetweenGroups(
                body_group_a=[b for b in gripper.bodies if b.has_collision()],
                body_group_b=target_collision_bodies,
            )
        ]

    @staticmethod
    def _kinematic_chain_to_root(body: Body) -> list[Body]:
        """
        :return: All bodies from ``body`` up to (but not including) the world root,
                 walking via parent_connection.
        """
        chain = []
        current = body
        while current.parent_connection is not None:
            chain.append(current)
            current = current.parent_connection.parent
        return chain

    def _build_temporary_collision_rules(
        self, robot: AbstractRobot, gripper: EndEffector, target: Body
    ) -> list:
        """
        :return: Temporary collision rules combining a reduced-distance avoidance rule with
                 explicit allow-rules for the target's kinematic chain.
        """
        return [
            AvoidExternalCollisions(
                robot=robot,
                buffer_zone_distance=self._external_collision_buffer_distance,
            ),
            *self._build_collision_rules(gripper, target),
        ]

    _max_trajectory_waypoints: ClassVar[int] = 20
    """Maximum number of trajectory waypoints passed to the CartesianPose sequence."""

    def _subsample_trajectory(self, trajectory: list[Pose]) -> list[Pose]:
        """
        :return: A subsampled trajectory of at most :attr:`_max_trajectory_waypoints` poses,
                 always including the first and last pose.
        """
        if len(trajectory) <= self._max_trajectory_waypoints:
            return trajectory
        step = len(trajectory) // self._max_trajectory_waypoints
        subsampled = trajectory[::step]
        if subsampled[-1] is not trajectory[-1]:
            subsampled.append(trajectory[-1])
        return subsampled

    def _execute_for_any_gripper(self, target: Body, trajectory: list[Pose]) -> bool:
        trajectory = self._subsample_trajectory(trajectory)
        world = self.robot._world
        original_temporary_rules = list(world.collision_manager.temporary_rules)
        original_collision_matrix = getattr(
            world.collision_manager, "collision_matrix", None
        )
        original_origin = self.robot.root.parent_connection.origin

        try:
            for index in range(len(list(self.robot.get_end_effectors()))):
                gripper = list(self.robot.get_end_effectors())[index]
                costmap = self._setup_costmap(trajectory[0], world)

                for base_candidate in costmap:
                    base_candidate.z = original_origin.z
                    self.robot.root.parent_connection.origin = base_candidate

                    motion_statechart, sequence = self._build_motion_statechart(
                        world.root, gripper.tool_frame, trajectory
                    )
                    motion_statechart.add_node(
                        UpdateTemporaryCollisionRules(
                            temporary_rules=self._build_temporary_collision_rules(
                                self.robot, gripper, target
                            ),
                        )
                    )
                    self._add_motion_termination_nodes(
                        motion_statechart, sequence, self.robot
                    )

                    executor = Executor(context=MotionStatechartContext(world=world))
                    with world.reset_state_context():
                        executor.compile(motion_statechart=motion_statechart)
                        try:
                            executor.tick_until_end(timeout=self._timeout)
                        except (
                            TimeoutError,
                            CollisionViolatedError,
                            LocalMinimumReachedError,
                        ):
                            pass

                    if motion_statechart.is_end_motion():
                        return True
        finally:
            self.robot.root.parent_connection.origin = original_origin
            world.collision_manager.clear_temporary_rules()
            world.collision_manager.extend_temporary_rule(original_temporary_rules)
            if original_collision_matrix is not None:
                world.collision_manager.set_collision_matrix(original_collision_matrix)
            clear_memoization_cache(world.collision_manager.collision_detector)
            world.notify_state_change()

        return False

    def _setup_costmap(self, target_pose: Pose, world: World):
        """
        :return: An occupancy + ring costmap centred on the ground projection of target_pose.
        """
        position = target_pose.to_position().to_np()
        ground_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position[0], y=position[1], z=0.0
        )
        [base] = [p for p in self.robot._robot_parts if isinstance(p, MobileBase)]
        base_bb = base.bounding_box
        occupancy = OccupancyCostmap(
            distance_to_obstacle=(base_bb.depth / 2 + base_bb.width / 2) / 2,
            world=world,
            robot_view=self.robot,
            width=200,
            height=200,
            resolution=0.02,
            origin=ground_pose,
        )
        ring = RingCostmap(
            distance=self._arm_reach_distance,
            std=15,
            world=world,
            resolution=0.02,
            width=200,
            height=200,
            origin=ground_pose,
        )
        costmap = occupancy + ring
        costmap.number_of_samples = self._costmap_samples
        costmap.orientation_generator = (
            OrientationGenerator.orientation_generator_for_axis(
                list(base.forward_axis.to_np())
            )
        )
        return costmap

    @staticmethod
    def _add_motion_termination_nodes(
        motion_statechart: MotionStatechart,
        sequence: Sequence,
        robot: AbstractRobot,
    ) -> None:
        """
        Add EndMotion and collision-avoidance nodes to the MotionStatechart.

        :param motion_statechart: MotionStatechart to modify in place.
        :param sequence: The Sequence node that triggers EndMotion on completion.
        :param robot: Robot used for collision avoidance.
        """
        motion_statechart.add_node(EndMotion.when_true(sequence))
        motion_statechart.add_node(
            local_minimum_reached := LocalMinimumReached(name="local_minimum_reached")
        )
        motion_statechart.add_node(
            CancelMotion.when_true(
                local_minimum_reached, exception=LocalMinimumReachedError()
            )
        )
        motion_statechart.add_node(
            ExternalCollisionAvoidance(
                name="external_collision_avoidance",
                robot=robot,
            )
        )
