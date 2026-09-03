from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict

from coraplex.plans.attachment_nodes import DetachNode
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
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import GripperIsFree
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
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceAction(
    ActionDescription,
    HasApproachesGraspPoses,
    PlaceTuningParameters,
    HasGraspDetectionThreshold,
    HasTcpGoalThresholds,
):
    """
    Places an Object at a position using an arm.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be place
    """
    target_location: Pose
    """
    Pose in the world at which the object should be placed.
    """

    arm: Arms
    """
    Arm that is currently holding the object
    """

    grasp_release_threshold: float = field(default=0.1, kw_only=True)
    """
    Maximum fraction of sampled rays between the gripper's fingers that may still hit
    :attr:`object_designator` for it to count as released (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """

    def _retract_plan(self, retract_pose: Pose) -> PlanNode:
        """
        :return: The plan that re-parents the placed object back to the world and
            retracts the end effector away from it.
        """
        return sequential(
            [
                DetachNode(body=self.object_designator, new_parent=self.world.root),
                MoveToolCenterPointMotion(
                    retract_pose,
                    self.arm,
                    max_linear_velocity=self.retract_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
            ],
        )

    def _grasp_on_the_held_object(self) -> Pose:
        """
        The grasp the object is held by, in :attr:`object_designator`'s own frame.

        Read off the gripper itself while it holds the object, since the transform
        between the two *is* the grasp, wherever on the object it sits. A plan is built
        before it runs, though, so the object is usually still on its shelf at this
        point; then the grasp the preceding pick-up was told to take says the same thing
        in advance.

        :return: The grasp frame, in :attr:`object_designator`'s frame.
        """
        end_effector = ViewManager.get_arm_view(self.arm, self.robot).end_effector
        if end_effector.held_body is self.object_designator:
            return end_effector.held_body_T_grasp

        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        if previous_pick is None:
            return Pose(reference_frame=self.object_designator)
        return previous_pick.designator.grasp_pose

    def _grasp_pose_at(self, target_location: Pose) -> Pose:
        """
        The grasp frame the object would be released from, were it at
        ``target_location``.

        :param target_location: Where the object should end up.
        :return: The grasp frame, in ``target_location``'s frame.
        """
        return (
            target_location.to_homogeneous_matrix()
            @ self._grasp_on_the_held_object().to_homogeneous_matrix()
        ).to_pose()

    @property
    def _action_plan(self) -> PlanNode:
        transport_pose, placing_pose, retract_pose = self.grasp_pose_sequence(
            self._grasp_pose_at(self.target_location),
            ViewManager.get_arm_view(self.arm, self.robot).end_effector,
            self._grasp_on_the_held_object(),
            reverse=True,
        )

        return sequential(
            [
                MoveToolCenterPointMotion(
                    transport_pose,
                    self.arm,
                    allow_gripper_collision=True,
                    max_linear_velocity=self.transport_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveToolCenterPointMotion(
                    placing_pose,
                    self.arm,
                    allow_gripper_collision=True,
                    max_linear_velocity=self.placing_linear_velocity,
                    position_threshold=self.position_threshold,
                    orientation_threshold=self.orientation_threshold,
                ),
                MoveGripperMotion(
                    GripperState.OPEN,
                    self.arm,
                    allow_gripper_collision=True,
                    finger_velocity=self.release_opening_velocity,
                ),
                self._retract_plan(retract_pose),
            ],
            self.context,
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be in the gripper frame.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return or_(
            not_(GripperIsFree(end_effector)),
            is_body_gripped(
                variable_from(kwargs["object_designator"]),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper must be free again and the object needs to be at the target
        location.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return and_(
            GripperIsFree(end_effector),
            not_(
                is_body_gripped(
                    variable_from(kwargs["object_designator"]),
                    end_effector,
                    threshold=kwargs["grasp_release_threshold"],
                )
            ),
            allclose(
                variable_from(kwargs["object_designator"]).global_pose,
                kwargs["target_location"],
                atol=0.03,
            ),
        )
