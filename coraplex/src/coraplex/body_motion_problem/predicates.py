from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    CollisionViolatedError,
    LocalMinimumReachedError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.qp.exceptions import QPSolverException
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
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    EndEffector,
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
    can follow the full trajectory of world-space poses. Robots without a mobile
    base are tested from their current base placement instead.
    """

    _timeout: ClassVar[int] = 2000
    _costmap_samples: ClassVar[int] = 10
    _arm_reach_distance: ClassVar[float] = 0.7
    _external_collision_buffer_distance: ClassVar[float] = 0.01
    _max_trajectory_waypoints: ClassVar[int] = 20
    """Maximum number of trajectory waypoints passed to the CartesianPose sequence."""

    def __call__(self) -> bool:
        if (
            self.motion.motion_trajectory is None
            or self.motion.motion_trajectory.is_empty()
        ):
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
                annotation.root.parent_connection == self.motion.connection
                and annotation.handle is not None
            ):
                return annotation.handle.root
        return self.motion.connection.child

    def _compute_body_trajectory(self, target: Body) -> list[Pose]:
        """
        :return: World-space poses of the target body at each trajectory step.

        Caller must hold a ``reset_state_context()`` so that world state is
        restored after this method returns.
        """
        world = target._world
        trajectory = []
        for position in self.motion.motion_trajectory.positions_for(
            self.motion.connection
        ):
            world.set_positions_1DOF_connection({self.motion.connection: position})
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
        world = target._world
        chain = world.compute_chain_of_kinematic_structure_entities(world.root, target)
        target_collision_bodies = [b for b in chain if b.has_collision()]
        if not target_collision_bodies:
            return []
        return [
            AllowCollisionBetweenGroups(
                body_group_a=[b for b in gripper.bodies if b.has_collision()],
                body_group_b=target_collision_bodies,
            )
        ]

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

    def _subsample_trajectory(self, trajectory: list[Pose]) -> list[Pose]:
        """
        :return: A subsampled trajectory of at most :attr:`_max_trajectory_waypoints` poses,
                 always including the first and last pose.
        """
        if len(trajectory) <= self._max_trajectory_waypoints:
            return trajectory
        waypoint_indices = np.linspace(
            0, len(trajectory) - 1, self._max_trajectory_waypoints, dtype=int
        )
        return [trajectory[index] for index in waypoint_indices]

    def _execute_for_any_gripper(self, target: Body, trajectory: list[Pose]) -> bool:
        trajectory = self._subsample_trajectory(trajectory)
        world = self.robot._world
        base_is_relocatable = isinstance(self.robot, HasMobileBase)
        original_temporary_rules = list(world.collision_manager.temporary_rules)
        original_collision_matrix = getattr(
            world.collision_manager, "collision_matrix", None
        )
        original_origin = self.robot.root.parent_connection.origin

        try:
            for gripper in self.robot.get_end_effectors():
                if not base_is_relocatable:
                    if self._gripper_can_follow_trajectory(gripper, target, trajectory):
                        return True
                    continue

                for base_candidate in self._setup_costmap(trajectory[0], world):
                    base_candidate.z = original_origin.z
                    self.robot.root.parent_connection.origin = base_candidate
                    if self._gripper_can_follow_trajectory(gripper, target, trajectory):
                        return True
        finally:
            if base_is_relocatable:
                self.robot.root.parent_connection.origin = original_origin
            world.collision_manager.clear_temporary_rules()
            world.collision_manager.extend_temporary_rule(original_temporary_rules)
            if original_collision_matrix is not None:
                world.collision_manager.set_collision_matrix(original_collision_matrix)
            clear_memoization_cache(world.collision_manager.collision_detector)
            world.notify_state_change()

        return False

    def _gripper_can_follow_trajectory(
        self, gripper: EndEffector, target: Body, trajectory: list[Pose]
    ) -> bool:
        """
        :return: Whether the gripper can follow the trajectory from the robot's
                 current base placement.
        """
        world = self.robot._world
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
        self._add_motion_termination_nodes(motion_statechart, sequence, self.robot)

        executor = Executor(context=MotionStatechartContext(world=world))
        with world.reset_state_context():
            executor.compile(motion_statechart=motion_statechart)
            try:
                executor.tick_until_end(timeout=self._timeout)
            except (
                TimeoutError,
                CollisionViolatedError,
                LocalMinimumReachedError,
                QPSolverException,
            ):
                # Each of these means "this candidate placement cannot follow the
                # trajectory", not a programming error: the motion ran out of ticks,
                # violated a collision constraint, converged into a local minimum,
                # or posed an infeasible/unsolvable QP.
                pass

        return motion_statechart.is_end_motion()

    def _setup_costmap(self, target_pose: Pose, world: World):
        """
        :return: An occupancy + ring costmap centred on the ground projection of target_pose.
        """
        position = target_pose.to_position().to_np()
        ground_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position[0], y=position[1], z=0.0
        )
        base = self.robot.mobile_base
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
