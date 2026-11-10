from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from coraplex.robot_plans.actions.base import ParallelAction, SequentialAction
from cramph.composites import Parallel
from cramph.node import StatechartNode
from giskardpy.motion_statechart.mixins import HasFingerStallToleranceParameters
from coraplex.robot_plans.mixins import (
    HasArmParameter,
    HasGraspApproachParameters,
    HasGraspCandidateParameter,
    HasGraspDetectionThresholdParameter,
    HasPickUpTuningParameters,
    HasReachTuningParameters,
    MovesToolCenterPoint,
    OperatesGripper,
    ReattachesBranches,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.weights import DefaultWeights
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from krrood_spatial_math.spatial_types import RotationMatrix
from krrood_spatial_math.spatial_types import Pose

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False, kw_only=True)
class ReachAction(
    MovesToolCenterPoint,
    OperatesGripper,
    SequentialAction,
    HasGraspCandidateParameter,
    HasArmParameter,
    HasGraspApproachParameters,
    HasReachTuningParameters,
    HasGraspDetectionThresholdParameter,
):
    """
    Let the robot reach a specific pose.
    """

    reverse_reach_order: bool = False
    """
    Whether to come down onto the grasp from the retreat pose above it, as a release
    does, instead of from the pre-grasp pose.
    """

    open_gripper_at_pre_pose: bool = False
    """
    Whether to open the gripper once the pre-pose is reached, used by
    :class:`PickUpAction` to open before its slower final approach.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper to touch the surroundings of the grasp it
            comes down onto.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        poses = self.grasp_pose_sequence(
            self.grasp.grasp_pose, self.arm.end_effector, self.grasp
        )
        pre_pose = poses.retreat if self.reverse_reach_order else poses.pre_grasp
        children = [
            self.tool_center_point_goal(
                pre_pose,
                self.arm,
                max_linear_velocity=self.pre_approach_linear_velocity,
            ),
        ]
        if self.open_gripper_at_pre_pose:
            children.append(
                self.gripper_goal_type.moving_gripper_to(
                    self.arm.end_effector.default_configuration(GripperState.OPEN)
                )
            )
        children.append(
            self.tool_center_point_goal(
                poses.grasp,
                self.arm,
                max_linear_velocity=self.final_approach_linear_velocity,
            )
        )
        return children


@dataclass(eq=False, repr=False, kw_only=True)
class PickUpAction(
    MovesToolCenterPoint,
    ReattachesBranches,
    SequentialAction,
    HasGraspCandidateParameter,
    HasArmParameter,
    HasGraspApproachParameters,
    HasReachTuningParameters,
    HasPickUpTuningParameters,
    HasGraspDetectionThresholdParameter,
    HasFingerStallToleranceParameters,
):
    """
    Let the robot pick up an object: take hold of it and lift it clear of its support.
    """

    @classmethod
    def from_above(cls, graspable: IsGraspable, arm: Arm) -> PickUpAction:
        """
        :param graspable: The annotation of an object standing upright.
        :param arm: The arm that picks it up.
        :return: The pick-up that comes down onto the origin of `graspable` along its
            z-axis.
        """
        grasp = GraspCandidate(
            graspable,
            Pose(
                orientation=RotationMatrix.from_rpy(pitch=np.pi / 2).quaternion,
                reference_frame=graspable.root,
            ),
        )
        return cls(grasp=grasp, arm=arm)

    def _grasp_steps(self) -> list[StatechartNode]:
        """
        A pick-up is a grasp the world is then told about: the object hangs off the tool
        frame afterwards, which is what makes it move with the arm.

        :return: The steps taking :attr:`grasp`, without lifting the object.
        """
        return [
            GraspingAction(
                grasp=self.grasp,
                arm=self.arm,
                approach_clearance=self.approach_clearance,
                retreat_distance=self.retreat_distance,
                pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                final_approach_linear_velocity=self.final_approach_linear_velocity,
                grasp_closing_velocity=self.grasp_closing_velocity,
                finger_stall_minimum_time=self.finger_stall_minimum_time,
                tolerate_finger_stall=self.tolerate_finger_stall,
                grasp_detection_threshold=self.grasp_detection_threshold,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            ),
            *self.reattach(
                self.grasp.graspable.root,
                self.arm.end_effector.tool_frame,
            ),
        ]

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper to touch the surroundings of its object,
            which it is around from reaching for it until it has lifted it clear, the
            moment the object is put under it included.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        lift_to_pose = self.grasp_pose_sequence(
            self.grasp.grasp_pose, self.arm.end_effector, self.grasp
        ).retreat
        return [
            *self._grasp_steps(),
            self.tool_center_point_goal(
                lift_to_pose,
                self.arm,
                max_linear_velocity=self.lift_linear_velocity,
            ),
        ]


@dataclass(eq=False, repr=False, kw_only=True)
class GraspingAction(
    MovesToolCenterPoint,
    OperatesGripper,
    SequentialAction,
    HasGraspCandidateParameter,
    HasArmParameter,
    HasGraspApproachParameters,
    HasReachTuningParameters,
    HasPickUpTuningParameters,
    HasGraspDetectionThresholdParameter,
    HasFingerStallToleranceParameters,
):
    """
    Let the robot take hold of an object: reach onto a grasp and close on it.

    What a pick-up does before it lifts, and the whole of it when the object is meant to
    stay where it is -- a handle being pulled, say.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper to touch the surroundings of the grasp it
            reaches and closes on.
        """
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        return [
            ReachAction(
                grasp=self.grasp,
                arm=self.arm,
                approach_clearance=self.approach_clearance,
                retreat_distance=self.retreat_distance,
                pre_approach_linear_velocity=self.pre_approach_linear_velocity,
                final_approach_linear_velocity=self.final_approach_linear_velocity,
                open_gripper_at_pre_pose=True,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
                grasp_detection_threshold=self.grasp_detection_threshold,
            ),
            self.closing_step(
                self.gripper_goal_type.moving_gripper_to(
                    self.arm.end_effector.default_configuration(
                        GripperState.CLOSE, finger_velocity=self.grasp_closing_velocity
                    ),
                    finger_stall_minimum_time=self.finger_stall_minimum_time,
                    tolerate_finger_stall=self.tolerate_finger_stall,
                )
            ),
        ]

    def closing_step(self, closing_goal: StatechartNode) -> Parallel:
        """
        :param closing_goal: A new node closing the gripper on the object.
        :return: A new step running `closing_goal` while the gripper is kept at the
            grasp, whatever else moves the robot meanwhile.
        """
        return Parallel(
            nodes=[
                closing_goal,
                self.tool_center_point_goal(
                    self.grasp_pose_sequence(
                        self.grasp.grasp_pose, self.arm.end_effector, self.grasp
                    ).grasp,
                    self.arm,
                ),
            ]
        )


@dataclass(eq=False, repr=False, kw_only=True)
class FixateObjectAction(ParallelAction):
    """
    Keeps what a gripper holds exactly where it is, by keeping the gripper where it is
    when the action starts, whether the object hangs off the gripper or not, such as the
    handle of a drawer the gripper holds open.

    Like the task it holds, it leaves its end to its owner: beside other nodes in a
    parallel it keeps the object in place for as long as the parallel runs, while a
    sequence ends it as soon as the gripper holds still.
    """

    end_effector: EndEffector
    """
    The gripper that holds the object.
    """

    @property
    def inherent_success_condition(self) -> Scalar:
        """
        Differs from :attr:`~coraplex.robot_plans.actions.base.ParallelAction.inherent_s
        uccess_condition` in never ending the action itself, because ending it lets the
        object go.
        """
        return Scalar.const_false()

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the gripper to touch the object it is around.
        """
        return [AllowCollisionForEndEffector(end_effector=self.end_effector)]

    def create_steps(self) -> list[StatechartNode]:
        tool_frame = self.end_effector.tool_frame
        return [
            CartesianPose(
                name="Keep the gripper in place",
                root_link=self.world.root,
                tip_link=tool_frame,
                goal=Pose(reference_frame=tool_frame),
                # Holding on has to matter more than keeping clear of what is held.
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
            )
        ]
