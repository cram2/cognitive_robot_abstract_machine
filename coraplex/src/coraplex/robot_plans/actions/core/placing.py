from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    ConditionType,
    and_,
    not_,
    or_,
    variable_from,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose
from typing_extensions import Any, Dict, List, Tuple

from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import ObjectIsNotHeld
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperHolds
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.mixins import (
    HasApproachesGraspPoses,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
    PlaceTuningParameters,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)


@dataclass
class PlaceAction(
    ActionDescription,
    HasApproachesGraspPoses,
    PlaceTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
):
    """
    Places an object at a position with the arm that holds it.
    """

    object_designator: HasGraspCandidates
    """
    The annotation of the object that should be placed.
    """
    target_location: Pose
    """
    Pose in the world at which the object should be placed.
    """

    grasp_release_threshold: float = field(default=0.1, kw_only=True)
    """
    Maximum fraction of sampled rays between the gripper's fingers that may still hit
    :attr:`object_designator` for it to count as released (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """

    @property
    def _action_plan(self) -> PlanNode:
        arm, grasp = self._holding_arm_and_grasp()
        # A release runs the grasp backwards: down from above the target, then out
        # along the way the grasp was approached.
        poses = self.grasp_pose_sequence(
            grasp.moved_to(self.target_location), arm.end_effector, grasp
        )

        return sequential(
            [
                MoveToolCenterPointMotion(
                    poses.retreat,
                    arm,
                    allow_gripper_collision=True,
                    max_linear_velocity=self.transport_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveToolCenterPointMotion(
                    poses.grasp,
                    arm,
                    allow_gripper_collision=True,
                    max_linear_velocity=self.placing_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveGripperMotion(
                    configuration=arm.end_effector.default_configuration(
                        GripperState.OPEN,
                        finger_velocity=self.release_opening_velocity,
                    ),
                    allow_gripper_collision=True,
                ),
                ReAttachNode(
                    body=self.object_designator.root, new_parent=self.world.root
                ),
                MoveToolCenterPointMotion(
                    poses.pre_grasp,
                    arm,
                    max_linear_velocity=self.retract_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
            ],
            self.context,
        )

    def _holding_arm_and_grasp(self) -> Tuple[Arm, GraspCandidate]:
        """
        The arm that holds :attr:`object_designator`, and the grasp it holds it by.

        Read off the gripper while it holds the object; while the plan is still being
        built, taken from the latest pick-up before this place.

        :return: The arm and its grasp on the object.
        :raises ObjectIsNotHeld: If no arm holds the object and no pick-up precedes this
            place.
        """
        object_body = self.object_designator.root
        for arm in self.robot.all_arms:
            end_effector = arm.end_effector
            if GripperHolds(end_effector, object_body)():
                return arm, GraspCandidate(
                    self.object_designator, end_effector.held_body_T_grasp
                )
        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        if previous_pick is None:
            raise ObjectIsNotHeld(self.object_designator)
        return previous_pick.designator.arm, previous_pick.designator.grasp

    def _grasp_on_the_held_object(self) -> GraspCandidate:
        """
        :return: The grasp :attr:`object_designator` is held by, as
            :meth:`_holding_arm_and_grasp` finds it.
        """
        return self._holding_arm_and_grasp()[1]

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        An arm of the robot needs to hold the object, whether the object hangs off its
        gripper or merely lies between the fingers.

        A thin or rim grasp leaves too little of the object between the fingers for the
        ray test alone to see it, so a gripper the object hangs off counts too.
        """
        object_body = kwargs["object_designator"].root
        return or_(
            *[
                GripperHolds(arm.end_effector, object_body)
                for arm in context.robot.all_arms
            ],
            *PlaceAction._grips_of_every_arm(
                context, kwargs, kwargs["grasp_detection_threshold"]
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        No arm may hold the object any more and it needs to be at the target location.
        """
        return and_(
            *[
                not_(grip)
                for grip in PlaceAction._grips_of_every_arm(
                    context, kwargs, kwargs["grasp_release_threshold"]
                )
            ],
            allclose(
                variable_from(kwargs["object_designator"].root).global_pose,
                kwargs["target_location"],
                atol=0.03,
            ),
        )

    @staticmethod
    def _grips_of_every_arm(
        context: Context, kwargs: Dict[str, Any], threshold: float
    ) -> List[ConditionType]:
        """
        :param threshold: The fraction of rays between the fingers that has to hit the
            object for it to count as gripped.
        :return: For every arm of the robot, whether its gripper grips the object.
        """
        return [
            is_body_gripped(
                variable_from(kwargs["object_designator"].root),
                arm.end_effector,
                threshold=threshold,
            )
            for arm in context.robot.all_arms
        ]
