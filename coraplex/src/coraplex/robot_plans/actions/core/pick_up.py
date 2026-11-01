from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Parallel
from cramph.node import StatechartNode
from coraplex.robot_plans.mixins import (
    ReattachesBranches,
    HasApproachesGraspPoses,
    HasGraspDetectionThreshold,
    MovesToolCenterPoint,
    OperatesGripper,
    PickUpTuningParameters,
    ReachTuningParameters,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types import RotationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, eq=False)
class HasGraspChoice:
    """
    Adds to an action the grasp it takes hold by.

    Shared by every action that closes a gripper on something. The grasp names the
    object it is on, so that is not asked for separately.
    """

    grasp: GraspCandidate
    """
    The grasp to take hold by.

    One of the object's own
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasGraspCandidates.grasp_candidates`.
    """

    arm: Arm
    """
    The arm that should be used.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class ReachAction(
    MovesToolCenterPoint,
    OperatesGripper,
    SequentialAction,
    HasApproachesGraspPoses,
    ReachTuningParameters,
    HasGraspDetectionThreshold,
):
    """
    Let the robot reach a specific pose.
    """

    arm: Arm
    """
    The arm that should be used for pick up.
    """

    grasp: GraspCandidate
    """
    The grasp the fingers should close on, which also names the object it is on.
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
    HasGraspChoice,
    HasApproachesGraspPoses,
    PickUpTuningParameters,
    HasGraspDetectionThreshold,
):
    """
    Let the robot pick up an object: take hold of it and lift it clear of its support.
    """

    tolerate_grasp_stall: bool = False
    """
    Whether the CLOSE motion's completion also tolerates a stalled grasp (see
    :attr:`~giskardpy.motion_statechart.goals.gripper.MoveFingers.tolerate_stall`).

    Opt-in rather than always on: building the stall monitor needs a velocity variable
    for every one of the gripper's connections, which is not guaranteed for every robot
    -- it crashes on Tracy's real-execution gripper, whose connections do not all have
    one.
    """

    @classmethod
    def from_above(cls, graspable: HasGraspCandidates, arm: Arm) -> PickUpAction:
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
                grasp_stall_minimum_time=self.grasp_stall_minimum_time,
                tolerate_grasp_stall=self.tolerate_grasp_stall,
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
    HasGraspChoice,
    HasApproachesGraspPoses,
    PickUpTuningParameters,
    HasGraspDetectionThreshold,
):
    """
    Let the robot take hold of an object: reach onto a grasp and close on it.

    What a pick-up does before it lifts, and the whole of it when the object is meant to
    stay where it is -- a handle being pulled, say.
    """

    tolerate_grasp_stall: bool = False
    """
    Whether the CLOSE motion's completion also tolerates a stalled grasp (see
    :attr:`~giskardpy.motion_statechart.goals.gripper.MoveFingers.tolerate_stall`).
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
                    stall_minimum_time=self.grasp_stall_minimum_time,
                    tolerate_stall=self.tolerate_grasp_stall,
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
