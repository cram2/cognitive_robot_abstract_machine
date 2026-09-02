from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Dict

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.pose_validator import IsObjectReachableBy
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.motions.container import OpeningMotion, ClosingMotion
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.robot_plans.parameter_mixins import (
    HandleOperationParameters,
    UsedGraspingPreposeDistance,
)
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import and_, ConditionType
from krrood.entity_query_language.factories import (
    or_,
    variable_from,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass
class OpenAction(
    ActionDescription, HandleOperationParameters, UsedGraspingPreposeDistance
):
    """
    Opens a container like object.
    """

    @property
    def _action_plan(self) -> PlanNode:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        end_effector = arm.end_effector

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            end_effector,
        )

        return sequential(
            [
                GraspingAction(
                    target_object=self.handle,
                    arm=self.arm,
                    grasp_description=grasp_description,
                ),
                OpeningMotion(handle=self.handle, arm=self.arm),
                MoveGripperMotion(
                    motion=GripperState.OPEN,
                    arm=self.arm,
                    allow_gripper_collision=True,
                ),
            ]
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper with which to open the container has to be free and the handle has
        to be reachable.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return and_(
            GripperIsFree(end_effector),
            IsObjectReachableBy(
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                arm=kwargs["arm"],
                object_designator=kwargs["handle"],
                as_single_grasp=True,
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The handle has to be in the gripper of the robot and the container has to be
        open.
        """
        end_effector = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        handle_body = kwargs["handle"].root

        parent_connection = handle_body.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        return and_(
            or_(
                is_body_in_gripper(variable_from(handle_body), end_effector) > 0.9,
                allclose(
                    variable_from(handle_body).global_pose.to_position(),
                    variable_from(end_effector.tool_frame).global_pose.to_position(),
                    atol=3e-2,
                ),
            ),
            variable_from(parent_connection).position > 0.3,
        )


@dataclass
class CloseAction(
    ActionDescription, HandleOperationParameters, UsedGraspingPreposeDistance
):
    """
    Closes a container like object.
    """

    @property
    def _action_plan(self) -> PlanNode:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        end_effector = arm.end_effector

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            end_effector,
        )

        return sequential(
            [
                GraspingAction(
                    target_object=self.handle,
                    arm=self.arm,
                    grasp_description=grasp_description,
                ),
                ClosingMotion(handle=self.handle, arm=self.arm),
                MoveGripperMotion(
                    motion=GripperState.OPEN,
                    arm=self.arm,
                    allow_gripper_collision=True,
                ),
            ]
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The container has to be closed.
        """
        close_connection = kwargs["handle"].root.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        return variable_from(close_connection).position < 0.1
