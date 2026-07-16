from __future__ import annotations

from dataclasses import dataclass

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
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceAction(ActionDescription):
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

    @property
    def _action_plan(self) -> PlanNode:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        end_effector = arm.end_effector

        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        previous_grasp = (
            previous_pick.designator.grasp_description
            if previous_pick
            else GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
            )
        )

        _, _, retract_pose = previous_grasp.pose_sequence(
            self.target_location, self.object_designator, reverse=True
        )

        return sequential(
            [
                ReachAction(
                    self.target_location,
                    self.arm,
                    previous_grasp,
                    self.object_designator,
                    reverse_reach_order=True,
                ),
                MoveGripperMotion(GripperState.OPEN, self.arm),
                DetachNode(body=self.object_designator, new_parent=self.world.root),
                MoveToolCenterPointMotion(retract_pose, self.arm),
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
            is_body_in_gripper(variable_from(kwargs["object_designator"]), end_effector)
            > 0.9,
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
            is_body_in_gripper(variable_from(kwargs["object_designator"]), end_effector)
            < 0.1,
            allclose(
                variable_from(kwargs["object_designator"]).global_pose,
                kwargs["target_location"],
                atol=0.03,
            ),
        )
