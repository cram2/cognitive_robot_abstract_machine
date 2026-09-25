from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional

from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.plan_node import PlanNode
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    or_,
    not_,
    and_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import ObjectIsNotHeld
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.base import ActionDescription
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
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses


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

    object_designator: HasGraspPoses
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
        arm = self._holding_arm()
        grasp = self._grasp_on_the_held_object()
        transport_pose, placing_pose, retract_pose = self.grasp_pose_sequence(
            grasp.moved_to(self.target_location),
            arm.end_effector,
            grasp,
            reverse=True,
        )

        return sequential(
            [
                MoveToolCenterPointMotion(
                    transport_pose,
                    arm,
                    allow_gripper_collision=True,
                    max_linear_velocity=self.transport_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveToolCenterPointMotion(
                    placing_pose,
                    arm,
                    allow_gripper_collision=True,
                    max_linear_velocity=self.placing_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveGripperMotion(
                    GripperState.OPEN,
                    arm.end_effector,
                    allow_gripper_collision=True,
                    finger_velocity=self.release_opening_velocity,
                ),
                ReAttachNode(
                    body=self.object_designator.root, new_parent=self.world.root
                ),
                MoveToolCenterPointMotion(
                    retract_pose,
                    arm,
                    max_linear_velocity=self.retract_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
            ],
            self.context,
        )


    def _holding_arm(self) -> Arm:
        """
        The arm that holds :attr:`object_designator`.

        A plan is built before it runs, so the object is usually still on its shelf at
        this point; then the arm is the one the preceding pick-up of it is going to take
        it with.

        :return: The arm holding the object, or about to hold it.
        :raises ObjectIsNotHeld: If no arm holds the object and no pick-up of it
            precedes this place.
        """
        for arm in self.robot.get_arms():
            if arm.end_effector.held_body is self.object_designator.root:
                return arm
        previous_pick = self._previous_pick_up_of_the_object()
        if previous_pick is None:
            raise ObjectIsNotHeld(self.object_designator)
        return previous_pick.arm


    def _previous_pick_up_of_the_object(self) -> Optional[PickUpAction]:
        """
        :return: The pick-up right before this place, if it picks up
            :attr:`object_designator`.
        """
        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        if previous_pick is None:
            return None
        if (
            previous_pick.designator.grasp.graspable.root
            is not self.object_designator.root
        ):
            return None
        return previous_pick.designator

    def _grasp_on_the_held_object(self) -> GraspPose:
        """
        The grasp the object is held by.

        Read off the gripper itself while it holds the object, since the transform
        between the two *is* the grasp, wherever on the object it sits. Before the
        object is held, the grasp the preceding pick-up was told to take says the same
        thing in advance.

        :return: The grasp on :attr:`object_designator`.
        :raises ObjectIsNotHeld: If no arm holds the object and no pick-up of it
            precedes this place.
        """
        for arm in self.robot.get_arms():
            held = arm.end_effector.grasp_on(self.object_designator.root)
            if held is not None:
                return GraspPose(self.object_designator, held)
        previous_pick = self._previous_pick_up_of_the_object()
        if previous_pick is None:
            raise ObjectIsNotHeld(self.object_designator)
        return previous_pick.grasp

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        An arm of the robot needs to hold the object.
        """
        return or_(
            *PlaceAction._grips_of_every_arm(
                context, kwargs, kwargs["grasp_detection_threshold"]
            )
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
            for arm in context.robot.get_arms()
        ]
