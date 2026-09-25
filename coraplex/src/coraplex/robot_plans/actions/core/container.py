from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    and_,
    or_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.robot_plans.motions.container import OpeningMotion, ClosingMotion
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import GraspPose
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenAction(ActionDescription):
    """
    Opens a container like object.
    """

    handle: Handle
    """
    The handle of the container that should be opened.
    """
    arm: Arm
    """
    Arm that should be used for opening the container.
    """

    approach_clearance: float = HasApproachesGraspPoses.approach_clearance
    """
    The gap in meters between the handle and the gripper before it closes on it.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                GraspingAction(
                    GraspPose.from_body_origin(self.handle),
                    self.arm,
                    approach_clearance=self.approach_clearance,
                ),
                OpeningMotion(self.handle.root, self.arm),
                MoveGripperMotion(
                    GripperState.OPEN,
                    self.arm.end_effector,
                    allow_gripper_collision=True,
                ),
            ]
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper with which to open the container has to be free.
        """
        return GripperIsFree(variables["arm"].end_effector)

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The handle has to be in the gripper of the robot and the container has to be
        open.
        """
        end_effector = kwargs["arm"].end_effector
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
class CloseAction(ActionDescription):
    """
    Closes a container like object.
    """

    handle: Handle
    """
    The handle of the container that should be closed.
    """

    arm: Arm
    """
    Arm that should be used for closing.
    """

    approach_clearance: float = HasApproachesGraspPoses.approach_clearance
    """
    The gap in meters between the handle and the gripper before it closes on it.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                GraspingAction(
                    GraspPose.from_body_origin(self.handle),
                    self.arm,
                    approach_clearance=self.approach_clearance,
                ),
                ClosingMotion(self.handle.root, self.arm),
                MoveGripperMotion(
                    GripperState.OPEN,
                    self.arm.end_effector,
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
