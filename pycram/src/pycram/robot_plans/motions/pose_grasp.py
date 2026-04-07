from dataclasses import dataclass
from enum import Enum, auto
from typing import List

from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPosition,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.enums import Arms
from pycram.robot_plans.motions.base import BaseMotion
from pycram.view_manager import ViewManager


class RetractDirection(Enum):
    """Direction to retract the gripper after grasping."""

    WORLD_Z = auto()
    """Lift along the world root's positive Z axis."""
    GRIPPER_Z = auto()
    """Retract along the gripper's negative Z axis (pull back)."""


def make_rule_for_allowing_collision_between_two_groups(
    bodies1: List[Body],
    bodies2: List[Body],
    robot: AbstractRobot,
) -> UpdateTemporaryCollisionRules:
    """Create a temporary collision rule that allows collisions between two groups of bodies."""
    return UpdateTemporaryCollisionRules(
        temporary_rules=[
            AllowCollisionBetweenGroups(
                body_group_a=[
                    b for b in bodies1 if b is not None and b.has_collision()
                ],
                body_group_b=[
                    b for b in bodies2 if b is not None and b.has_collision()
                ],
            ),
            AvoidExternalCollisions(robot=robot),
        ]
    )


@dataclass(kw_only=True)
class CollisionAwareArmMotion(BaseMotion):
    """
    Base class for arm motions that need collision avoidance.
    Resolves the tool frame and hand bodies from the arm via ViewManager,
    and provides optional collision avoidance with a configurable allowed-collision set.
    """

    arm: Arms
    """The arm performing the motion."""

    allowed_collision_bodies: List[Body]
    """Bodies to allow collision with (typically the object being manipulated)."""

    use_collision_avoidance: bool = True
    """Whether to enable collision avoidance during the motion."""

    def perform(self):
        pass

    def _with_collision_avoidance(
        self, tasks: List[Task], minimum_success: int = 1
    ) -> Task:
        if not self.use_collision_avoidance:
            if len(tasks) == 1:
                return tasks[0]
            return Parallel(tasks, minimum_success=minimum_success)
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)
        allow_rule = make_rule_for_allowing_collision_between_two_groups(
            list(hand.bodies), self.allowed_collision_bodies, robot=hand._robot
        )
        motion = Parallel(
            [
                *tasks,
                ExternalCollisionAvoidance(robot=hand._robot),
                SelfCollisionAvoidance(robot=hand._robot),
            ],
            minimum_success=minimum_success,
        )
        return Parallel([allow_rule, motion])


@dataclass(kw_only=True)
class PoseGraspMotion(CollisionAwareArmMotion):
    """
    Motion that moves to a grasp pose with collision avoidance.
    First moves to a pre-grasp pose (offset along the negative z-axis of the
    grasp pose), then moves to the actual grasp pose.
    """

    grasp_pose: Pose
    pre_grasp_distance: float = 0.1
    """Distance to offset the pre-grasp pose along the negative z-axis of the grasp pose (in meters)."""
    pre_grasp_threshold: float = 0.25

    @property
    def _motion_chart(self) -> Task:
        world = self.world
        tool_frame = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame

        grasp_rotation = self.grasp_pose.to_rotation_matrix()
        grasp_z_axis = grasp_rotation.z_vector()
        pre_grasp_position = (
            self.grasp_pose.to_position() - grasp_z_axis * self.pre_grasp_distance
        )
        pre_grasp_pose = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=pre_grasp_position,
            rotation_matrix=grasp_rotation,
        )

        move_to_pre_grasp = CartesianPose(
            goal_pose=pre_grasp_pose,
            root_link=world.root,
            tip_link=tool_frame,
            reference_angular_velocity=0.5,
            reference_linear_velocity=0.5,
            threshold=self.pre_grasp_threshold,
        )
        move_to_grasp = CartesianPose(
            goal_pose=self.grasp_pose,
            root_link=world.root,
            tip_link=tool_frame,
        )
        grasp_sequence = Sequence([move_to_pre_grasp, move_to_grasp])
        return self._with_collision_avoidance([grasp_sequence])


@dataclass(kw_only=True)
class RetractMotion(CollisionAwareArmMotion):
    """
    Motion that retracts the gripper with collision avoidance.
    Keeps the orientation of the gripper fixed while moving.
    """

    distance: float
    """Distance to retract in meters."""

    direction: RetractDirection = RetractDirection.WORLD_Z
    """Direction to retract along."""

    reference_velocity: float = 0.2
    """Maximum velocity during retract."""

    def _compute_goal(self) -> CartesianPosition:
        tool_frame = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        if self.direction == RetractDirection.WORLD_Z:
            goal_point = tool_frame.global_pose.to_position() + Vector3(
                0, 0, self.distance, reference_frame=self.world.root
            )
        else:
            gripper_z = tool_frame.global_pose.to_rotation_matrix().z_vector()
            goal_point = (
                tool_frame.global_pose.to_position() - gripper_z * self.distance
            )

        return CartesianPosition(
            goal_point=goal_point,
            root_link=self.world.root,
            tip_link=tool_frame,
            reference_velocity=self.reference_velocity,
        )

    @property
    def _motion_chart(self) -> Task:
        tool_frame = ViewManager.get_end_effector_view(self.arm, self.robot).tool_frame
        retract_position = self._compute_goal()
        keep_orientation = CartesianOrientation(
            root_link=self.world.root,
            tip_link=tool_frame,
            goal_orientation=tool_frame.global_pose.to_rotation_matrix(),
        )
        return self._with_collision_avoidance(
            [retract_position, keep_orientation], minimum_success=2
        )
