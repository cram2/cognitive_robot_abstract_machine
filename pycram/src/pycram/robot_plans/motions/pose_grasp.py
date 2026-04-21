from dataclasses import dataclass
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
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.enums import Arms, RetractDirection
from pycram.robot_plans.motions.base import BaseMotion
from pycram.view_manager import ViewManager


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
    Executes the provided pose_sequence (pre_grasp → grasp) as a Sequence.
    """

    pose_sequence: List[Pose]
    pre_grasp_threshold: float = 0.01
    grasp_approach_velocity: float = 0.05
    """Linear velocity when moving from pre-grasp to grasp pose."""

    @property
    def _motion_chart(self) -> Task:
        world = self.world
        hand = ViewManager.get_end_effector_view(self.arm, self.robot)
        tool_frame = hand.tool_frame

        pre_grasp_pose, grasp_pose = self.pose_sequence

        move_to_pre_grasp = CartesianPose(
            goal_pose=pre_grasp_pose,
            root_link=world.root,
            tip_link=tool_frame,
            threshold=self.pre_grasp_threshold,
        )
        move_to_grasp = CartesianPose(
            goal_pose=grasp_pose,
            root_link=world.root,
            tip_link=tool_frame,
            reference_linear_velocity=self.grasp_approach_velocity,
        )

        if not self.use_collision_avoidance:
            return Sequence([move_to_pre_grasp, move_to_grasp])

        pre_grasp_step = Parallel(
            [
                move_to_pre_grasp,
                ExternalCollisionAvoidance(robot=hand._robot),
                SelfCollisionAvoidance(robot=hand._robot),
            ],
            minimum_success=1,
        )
        grasp_step = self._with_collision_avoidance([move_to_grasp])

        return Sequence([pre_grasp_step, grasp_step])


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
            gripper_approach_axis = (
                tool_frame.global_pose.to_rotation_matrix().z_vector()
            )
            goal_point = (
                tool_frame.global_pose.to_position()
                - gripper_approach_axis * self.distance
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
