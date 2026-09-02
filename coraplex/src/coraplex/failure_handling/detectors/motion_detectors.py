from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coraplex.plans.failures import MotionDidNotFinish
from coraplex.failure_handling.failure_refiner import FailureDetector
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.plans.failures import (
    BodyUnfetchable,
    EndEffectorDidNotReachTarget,
    NavigationGoalNotReachedError,
    PlanFailure,
)
from coraplex.robot_plans.parameter_mixins import (
    ObjectActedOn,
    TargetLocationMovedTo,
    TargetPoseReached,
    UsedArm,
    UsedEndEffector,
    UsedGraspDescription,
)

# %% motion detectors

# A motion can stop short for many reasons, so carrying the right parameters only makes
# a detector a candidate: each one still checks the world for the specific thing its
# failure type claims, and declines by returning the failure it was given when that
# claim does not hold. The refiner then offers the same failure to the next most
# specific detector.
#
# These detectors all consume the same failure type, so the refiner ranks them by the
# number of parameters they require. Two detectors requiring equally many parameters of
# the same action are ambiguous, which the refiner reports as
# :class:`~coraplex.exceptions.AmbiguousFailureDetector`; the fix is a detector declared
# for a more specific failure type or for more parameters, not a restriction on what an
# action may inherit.


@dataclass
class NavigationGoalDetector(FailureDetector):
    """
    Detects that a motion of an action driving somewhere left the robot short of its
    destination.

    Declines when the robot stands at its destination, because the motion then failed
    for a reason other than the driving itself.
    """

    input_failure_type = MotionDidNotFinish
    output_failure_type = NavigationGoalNotReachedError
    required_parameter_mixins = (TargetLocationMovedTo,)

    arrival_tolerance: float = 0.03
    """
    How close the robot has to be to its destination to count as having arrived.

    Matches the tolerance
    :meth:`~coraplex.robot_plans.actions.core.navigation.NavigateAction.post_condition`
    accepts, so the detector agrees with what navigating considers success.
    """

    def detect(self, failure: PlanFailure) -> PlanFailure:
        action = failure.action_node.action
        current_pose = failure.context.robot.root.global_pose
        if np.allclose(
            current_pose.to_np(),
            action.target_location.to_np(),
            atol=self.arrival_tolerance,
        ):
            return failure
        return NavigationGoalNotReachedError(
            node=failure.node,
            current_pose=current_pose,
            goal_pose=action.target_location,
        )


@dataclass
class EndEffectorTargetDetector(FailureDetector):
    """
    Detects that a motion of an action moving an end effector left it short of the pose
    it was sent to.

    Declines when the end effector holds that pose, because the motion then failed for a
    reason other than reaching it.
    """

    input_failure_type = MotionDidNotFinish
    output_failure_type = EndEffectorDidNotReachTarget
    required_parameter_mixins = (UsedEndEffector, TargetPoseReached)

    arrival_tolerance: float = 0.1
    """
    How close the end effector has to be to its target to count as having reached it.

    Matches the tolerance
    :meth:`~coraplex.robot_plans.actions.core.robot_body.MoveManipulatorAction.post_condition`
    accepts, so the detector agrees with what the motion considers success.
    """

    def detect(self, failure: PlanFailure) -> PlanFailure:
        action = failure.action_node.action
        if np.allclose(
            action.end_effector.tool_frame.global_pose.to_np(),
            action.target_pose.to_np(),
            atol=self.arrival_tolerance,
        ):
            return failure
        return EndEffectorDidNotReachTarget(
            node=failure.node,
            end_effector=action.end_effector,
            target=action.target_pose,
        )


@dataclass
class BodyUnfetchableDetector(FailureDetector):
    """
    Detects that a motion of an action grasping an object with an arm failed because the
    arm cannot reach the object.

    Reachability is decided over the grasp pose sequence the action would actually use,
    not over the object's own frame, which an end effector never has to occupy. The
    detector declines when that sequence is reachable, because the grasp then failed for
    a reason other than the object being out of reach.
    """

    input_failure_type = MotionDidNotFinish
    output_failure_type = BodyUnfetchable
    required_parameter_mixins = (ObjectActedOn, UsedArm, UsedGraspDescription)

    def detect(self, failure: PlanFailure) -> PlanFailure:
        action = failure.action_node.action
        is_reachable = IsObjectReachableBy(
            context=failure.context,
            arm=action.arm,
            object_designator=action.target_object,
            grasp_description=action.grasp_description,
        )
        if is_reachable():
            return failure
        return BodyUnfetchable(
            node=failure.node, body=action.target_object.root, arm=action.arm
        )
